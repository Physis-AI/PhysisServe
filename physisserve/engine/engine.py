"""Single and concurrent video streams share the same tick-by-tick execution."""
from datetime import timedelta
from pathlib import Path
from tempfile import mkdtemp

import torch
import torch.distributed as dist

from .model_runner import ModelRunner
from .strategies.placement import MappedLayout, plan_pool
from .transport import Distributed, Local


class Engine:
    def __init__(self, config, control_group=None):
        self.config = config
        self.distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.device = torch.device(config.device)
        if self.device.type == "cuda":
            self.device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(self.device)
        self.control_group = control_group
        if self.distributed:
            if dist.get_backend() == "nccl":
                # Initialize the full communicator before any subset starts P2P.
                dist.barrier(device_ids=[self.device.index])
            if self.control_group is None:
                self.control_group = (dist.group.WORLD if dist.get_backend() == "gloo" else
                                      dist.new_group(backend="gloo", timeout=timedelta(minutes=30)))
        self.dtype = getattr(torch, config.dtype)
        self.codec_dtype = getattr(torch, config.compute_dtype or config.dtype)
        backend = config.make_backend()
        self.blocks, self.first_frames = backend.dit.num_blocks, backend.dit.first_frames
        self.sampler = config.make_sampler()
        self.runners = {}
        self.requests, self.sessions, self.rejected = (), (), {}

    @property
    def is_output(self):
        return self.rank == 0

    def _barrier(self):
        if self.distributed:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dist.barrier(group=self.control_group)

    def prepare(self, requests, candidates=None, budgets=()):
        """Encode prompts and load independent stream weights before inference.

        Measured candidates use stringified input indices and budgets in GPU-rank
        order. Without a resource profile, every stream uses the configured layout
        over the whole pool. Matching runner slots retain weights across calls.
        """
        self.requests, self.sessions = tuple(requests), ()
        if candidates is None:
            layout = self.config.make_layout()
            layouts = dict.fromkeys(range(len(self.requests)), layout)
            self.rejected = {}
        else:
            placement = plan_pool(candidates, budgets)
            placements = tuple(placement.placements)
            layouts = {int(item.stream_id): MappedLayout(
                item.candidate.layout, item.rank_map, self.config.ranks) for item in placements}
            self.rejected = {int(index): reason for index, reason in placement.rejected.items()}
        plans = {index: tuple(tuple(layout.rank_blocks(rank, self.blocks))
                              for rank in range(self.config.ranks)) for index, layout in layouts.items()}
        # Release unused stream weights before loading a replacement or its encoder.
        self.runners = {index: item for index, item in self.runners.items()
                        if index in plans and item[0] == plans[index]}
        conditions = {}
        if layouts and self.is_output:
            encoder = self.config.make_backend().encoder
            encoder.load(self.config.encoder_device, self.codec_dtype)
            with torch.inference_mode():
                for index in sorted(layouts):
                    conditions[index] = {key: value.cpu() for key, value in
                                         encoder.encode(self.requests[index].prompt).items()}
            del encoder
        if self.distributed:
            payload = [conditions]
            dist.broadcast_object_list(payload, src=0, group=self.control_group)
            conditions = payload[0]
        sessions = []
        for index in sorted(layouts):
            layout, ownership = layouts[index], plans[index]
            if index not in self.runners:
                backend = self.config.make_backend()
                transport = Distributed(self.device) if self.distributed else Local()
                runner = ModelRunner(backend.dit, self.sampler, layout, device=self.device,
                                     dtype=self.dtype, transport=transport)
                self.runners[index] = ownership, runner
            runner = self.runners[index][1]
            runner.layout = layout
            sessions.append((index, runner, conditions[index]))
        self.sessions = tuple(sessions)
        self._barrier()

    def run(self, on_latent=None, on_batch=None):
        """Interleave one tick per active stream; return CPU latents on rank zero.

        Callbacks receive the input index followed by the runner's chunk/latent or
        rank/batch-size values. dit_only retains outputs on the device until all
        streams finish. Other modes buffer completed latents in CPU memory;
        online_decode additionally decodes each chunk before advancing this rank.
        Calling run again reuses weights and starts fresh KV state and seeded noise.
        """
        active = []
        chunks = {index: [] for index, *_ in self.sessions}
        decoders, videos = {}, {}
        try:
            for index, runner, condition in self.sessions:
                request = self.requests[index]
                if self.is_output and request.mode == "online_decode":
                    decoder = self.config.make_backend().decoder
                    decoder.load(self.device, self.codec_dtype)
                    decoder.reset()
                    decoders[index], videos[index] = decoder, []
                latent_cb = (None if on_latent is None else
                             lambda chunk, value, index=index: on_latent(index, chunk, value))
                batch_cb = (None if on_batch is None else
                            lambda rank, size, index=index: on_batch(index, rank, size))
                iterator = runner.run_ticks(condition, request.shape, request.chunks,
                                            seed=request.seed, fps=request.fps,
                                            on_latent=latent_cb, on_batch=batch_cb)
                active.append((index, iterator))
            while active:
                for session in tuple(active):
                    index, iterator = session
                    try:
                        outputs = next(iterator)
                    except StopIteration:
                        active.remove(session)
                        continue
                    for chunk, latent in outputs:
                        request = self.requests[index]
                        if index in decoders:
                            decoder = decoders[index]
                            with torch.inference_mode():
                                pixels = decoder.decode(latent.to(device=self.device, dtype=self.codec_dtype), chunk)
                            videos[index].append(self._video(pixels))
                        if request.mode != "dit_only":
                            latent = latent.cpu()
                        chunks[index].append((chunk, latent))
            self._barrier()
            if not self.is_output:
                return None
            results, directories = [], {}
            for index, request in enumerate(self.requests):
                result = {"prompt": request.prompt, "fps": request.fps}
                if index in self.rejected:
                    result.update(status="rejected", reason=self.rejected[index])
                else:
                    outputs = sorted(chunks[index], key=lambda item: item[0])
                    result.update(latents=torch.cat([latent for _, latent in outputs], dim=2).cpu(),
                                  chunks=request.chunks, first_frames=self.first_frames)
                    if index in videos:
                        result["video"] = torch.cat(videos[index], dim=0)
                    if request.latent_dir is not None or request.mode == "offline_decode":
                        root = Path(request.latent_dir or "latents").expanduser().resolve()
                        if root not in directories:
                            root.mkdir(parents=True, exist_ok=True)
                            directories[root] = Path(mkdtemp(prefix="generate-", dir=root))
                        self._save_latents(result, directories[root], index)
                results.append(result)
            return results
        finally:
            for _, iterator in active:
                iterator.close()

    def decode(self, outputs):
        """Decode completed CPU latents into uint8 [T,H,W,3] videos on rank zero."""
        # The large VAE replaces the DiT stage on this GPU; latent outputs are
        # already on the CPU. A following request reloads its runners in prepare.
        self.sessions, self.runners = (), {}
        if not self.is_output or outputs is None:
            return None
        outputs = list(outputs)
        if all(output.get("status") == "rejected" for output in outputs):
            return outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        decoder = self.config.make_backend().decoder
        decoder.load(self.device, self.codec_dtype)
        results = []
        for output in outputs:
            if output.get("status") == "rejected":
                results.append(dict(output))
                continue
            latent, count, first = output["latents"], output["chunks"], output["first_frames"]
            reset = getattr(decoder, "reset", None)
            if reset is not None:
                reset()
            with torch.inference_mode():
                sequence = getattr(decoder, "decode_sequence", None)
                if sequence is not None:
                    pixels = sequence(latent.to(device=self.device, dtype=self.codec_dtype))
                else:
                    pieces = latent.split([first] + [1] * (count - 1), dim=2)
                    decoded = []
                    for index, piece in enumerate(pieces):
                        pixels = decoder.decode(piece.to(device=self.device, dtype=self.codec_dtype), index)
                        decoded.append(pixels)
                    pixels = torch.cat(decoded, dim=2)
            results.append({**output, "video": self._video(pixels)})
        return results

    @staticmethod
    def _video(pixels):
        video = ((pixels[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        return video.permute(1, 2, 3, 0).cpu().contiguous().clone()

    @staticmethod
    def _save_latents(output, directory, index):
        from safetensors.torch import save_file

        path = directory / f"stream-{index:04d}.safetensors"
        metadata = {key: str(output[key]) for key in ("prompt", "fps", "chunks", "first_frames")}
        save_file({"latents": output["latents"]}, str(path), metadata=metadata)
        output["latent_path"] = str(path)
