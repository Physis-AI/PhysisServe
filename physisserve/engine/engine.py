"""Single and concurrent video streams share the same tick-by-tick execution."""
from datetime import timedelta
from pathlib import Path
from tempfile import mkdtemp

import torch
import torch.distributed as dist

from ..sampling_params import Request
from .model_runner import ModelRunner, _request_schedules, _request_ticks
from .strategies.placement import MappedLayout
from .transport import Distributed, Local


class Engine:
    def __init__(self, config):
        self.config = config
        self.distributed = dist.is_initialized()
        self.rank = dist.get_rank() if self.distributed else 0
        self.device = torch.device(config.device)
        if self.device.type == "cuda":
            self.device = torch.device("cuda", self.rank)
            torch.cuda.set_device(self.device)
        if (dist.get_world_size() if self.distributed else 1) != config.ranks:
            raise ValueError("initialize exactly config.ranks processes before constructing the engine")
        self.control_group = None
        if self.distributed:
            if dist.get_backend() == "nccl":
                # Initialize the full communicator before any subset starts P2P.
                dist.barrier(device_ids=[self.device.index])
            self.control_group = (dist.group.WORLD if dist.get_backend() == "gloo" else
                                  dist.new_group(backend="gloo", timeout=timedelta(minutes=30)))
        self.dtype = getattr(torch, config.dtype)
        self.codec_dtype = getattr(torch, config.compute_dtype or config.dtype)
        backend = config.make_backend(config.model)
        self.blocks, self.first_frames = backend.dit.num_blocks, backend.dit.first_frames
        self.supports_stream_decode = (callable(getattr(backend.decoder, "decode", None))
                                       and not callable(getattr(backend.decoder, "decode_sequence", None)))
        self.sampler = config.make_sampler()
        self.runners = {}
        self.requests, self.sessions, self.rejected = (), (), {}
        self.prepared = self.busy = self.failed = False

    @property
    def is_output(self):
        return self.rank == 0

    def _check_idle(self):
        if self.busy:
            raise RuntimeError("the engine is already running")
        if self.failed:
            raise RuntimeError("distributed execution failed; restart the process group")

    def _barrier(self):
        if self.distributed:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            dist.barrier(group=self.control_group)

    def prepare(self, requests, placement=None):
        """Encode prompts and load independent stream weights before inference.

        A placement uses stringified input indices as stream IDs. Omitted placement
        gives every stream the configured layout over the whole GPU pool. Matching
        runner slots retain their weights across repeated calls.
        """
        self._check_idle()
        self.prepared = False
        self.requests, self.sessions = tuple(requests), ()
        if any(not isinstance(request, Request) for request in self.requests):
            raise TypeError("requests must contain Request objects")
        if not self.supports_stream_decode and any(r.mode == "stream_decode" for r in self.requests):
            raise ValueError("this VAE requires whole-sequence decoding; use deferred_decode")
        ids = {str(index) for index in range(len(self.requests))}
        if placement is None:
            layout = self.config.make_layout()
            layouts = dict.fromkeys(range(len(self.requests)), layout)
            self.rejected = {}
        else:
            placements = tuple(placement.placements)
            placed_ids = [item.stream_id for item in placements]
            if (placement.num_ranks != self.config.ranks or len(set(placed_ids)) != len(placed_ids)
                    or set(placed_ids) & set(placement.rejected)
                    or set(placed_ids) | set(placement.rejected) != ids):
                raise ValueError("placement must cover every input index exactly once on the configured pool")
            layouts = {int(item.stream_id): MappedLayout(
                item.candidate.layout, item.rank_map, self.config.ranks) for item in placements}
            self.rejected = {int(index): reason for index, reason in placement.rejected.items()}
        plans = {}
        for index, layout in layouts.items():
            if layout.num_ranks(self.blocks) != self.config.ranks:
                raise ValueError("each stream layout must cover the configured GPU pool")
            schedules = _request_schedules(layout, self.blocks, self.requests[index].chunks)
            plans[index] = (_request_ticks(schedules, self.blocks),
                            tuple(tuple(layout.rank_blocks(rank, self.blocks))
                                  for rank in range(self.config.ranks)))
        # Release unused stream weights before loading a replacement or its encoder.
        self.runners = {index: item for index, item in self.runners.items()
                        if index in plans and item[0] == plans[index][1]}
        conditions = {}
        if layouts and self.is_output:
            encoder = self.config.make_backend(self.config.model).encoder
            encoder.load(self.config.encoder_device or self.device, self.codec_dtype)
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
            layout, (ticks, ownership) = layouts[index], plans[index]
            if index not in self.runners:
                backend = self.config.make_backend(self.config.model)
                transport = Distributed(self.device) if self.distributed else Local()
                runner = ModelRunner(backend.dit, self.sampler, layout, device=self.device,
                                     dtype=self.dtype, transport=transport, output_rank=0)
                self.runners[index] = ownership, runner
            runner = self.runners[index][1]
            runner.layout = layout
            sessions.append((index, ticks, runner, conditions[index]))
        self.sessions = tuple(sessions)
        self._barrier()
        self.prepared = True

    def run(self, on_latent=None, on_batch=None):
        """Interleave one tick per active stream; return CPU latents on rank zero.

        Callbacks receive the input index followed by the runner's chunk/latent or
        rank/batch-size values. dit_only retains outputs on the device until all
        streams finish. Other modes buffer completed latents in CPU memory;
        stream_decode additionally decodes each chunk before advancing this rank.
        Calling run again reuses weights and starts fresh KV state and seeded noise.
        """
        self._check_idle()
        if not self.prepared:
            raise RuntimeError("prepare requests before running inference")
        self.busy = True
        active, iterators = [], []
        chunks = {index: [] for index, *_ in self.sessions}
        decoders, videos = {}, {}
        completed = False
        try:
            for index, ticks, runner, condition in self.sessions:
                request = self.requests[index]
                if self.is_output and request.mode == "stream_decode":
                    decoder = self.config.make_backend(self.config.model).decoder
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
                iterators.append(iterator)
                active.append([index, ticks, iterator])
            while active:
                for session in tuple(active):
                    index, remaining, iterator = session
                    try:
                        outputs = next(iterator)
                    except StopIteration as error:
                        raise RuntimeError("runner ended before its scheduled final tick") from error
                    for chunk, latent in outputs:
                        request = self.requests[index]
                        shape = (*request.shape[:2], self.first_frames if chunk == 0 else 1, *request.shape[3:])
                        if tuple(latent.shape) != shape:
                            raise RuntimeError(f"stream {index} chunk {chunk} has incorrect latent shape")
                        if index in decoders:
                            if chunk != len(chunks[index]):
                                raise RuntimeError(f"stream {index} returned missing or duplicate chunks")
                            decoder = decoders[index]
                            with torch.inference_mode():
                                pixels = decoder.decode(latent.to(device=self.device, dtype=self.codec_dtype), chunk)
                                self._check_pixels(pixels, decoder.frames(chunk, shape))
                            videos[index].append(self._video(pixels))
                        if request.mode != "dit_only":
                            latent = latent.cpu()
                        chunks[index].append((chunk, latent))
                    session[1] -= 1
                    if remaining == 1:
                        try:
                            next(iterator)
                        except StopIteration:
                            active.remove(session)
                        else:
                            raise RuntimeError("runner produced more ticks than its schedule")
            self._barrier()
            completed = True
            if not self.is_output:
                return None
            results, directories = [], {}
            for index, request in enumerate(self.requests):
                result = {"prompt": request.prompt, "fps": request.fps}
                if index in self.rejected:
                    result.update(status="rejected", reason=self.rejected[index])
                else:
                    outputs = sorted(chunks[index], key=lambda item: item[0])
                    if [chunk for chunk, _ in outputs] != list(range(request.chunks)):
                        raise RuntimeError(f"stream {index} returned missing or duplicate chunks")
                    result.update(latents=torch.cat([latent for _, latent in outputs], dim=2).cpu(),
                                  chunks=request.chunks, first_frames=self.first_frames)
                    if index in videos:
                        result["video"] = torch.cat(videos[index], dim=0)
                    if request.latent_dir is not None or request.mode == "deferred_decode":
                        root = Path(request.latent_dir or "latents").expanduser().resolve()
                        if root not in directories:
                            root.mkdir(parents=True, exist_ok=True)
                            directories[root] = Path(mkdtemp(prefix="generate-", dir=root))
                        self._save_latents(result, directories[root], index)
                results.append(result)
            return results
        finally:
            for iterator in iterators:
                iterator.close()
            decoders.clear()
            self.busy = False
            if self.distributed and not completed:
                self.failed = True

    def decode(self, outputs):
        """Decode completed CPU latents into uint8 [T,H,W,3] videos on rank zero."""
        self._check_idle()
        # The large VAE replaces the DiT stage on this GPU; latent outputs are
        # already on the CPU. A following request reloads its runners in prepare.
        self.sessions, self.runners, self.prepared = (), {}, False
        if not self.is_output or outputs is None:
            return None
        outputs = list(outputs)
        if any(output.get("status") != "rejected" and "latents" not in output for output in outputs):
            raise ValueError("each output must contain latents or an explicit rejection")
        if not any("latents" in output for output in outputs):
            return outputs
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        decoder = self.config.make_backend(self.config.model).decoder
        decoder.load(self.device, self.codec_dtype)
        results = []
        for output in outputs:
            if output.get("status") == "rejected":
                results.append(dict(output))
                continue
            latent, count, first = output["latents"], output["chunks"], output["first_frames"]
            if (latent.ndim != 5 or latent.shape[0] != 1 or first != self.first_frames
                    or count < 1 or latent.shape[2] != first + count - 1):
                raise ValueError("invalid sequence latent shape or chunk metadata")
            pieces = latent.split([first] + [1] * (count - 1), dim=2)
            frame_counts = [decoder.frames(index, tuple(piece.shape)) for index, piece in enumerate(pieces)]
            reset = getattr(decoder, "reset", None)
            if reset is not None:
                reset()
            with torch.inference_mode():
                sequence = getattr(decoder, "decode_sequence", None)
                if sequence is not None:
                    pixels = sequence(latent.to(device=self.device, dtype=self.codec_dtype))
                else:
                    decoded = []
                    for index, piece in enumerate(pieces):
                        pixels = decoder.decode(piece.to(device=self.device, dtype=self.codec_dtype), index)
                        self._check_pixels(pixels, frame_counts[index])
                        decoded.append(pixels)
                    pixels = torch.cat(decoded, dim=2)
                self._check_pixels(pixels, sum(frame_counts))
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

    @staticmethod
    def _check_pixels(pixels, frames):
        if pixels.ndim != 5 or tuple(pixels.shape[:3]) != (1, 3, frames):
            raise ValueError(f"decoder must return pixels shaped [1,3,{frames},H,W]")
