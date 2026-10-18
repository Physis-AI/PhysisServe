import atexit
from datetime import timedelta
from multiprocessing.connection import wait
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import Config
from .engine.engine import Engine
from .sampling_params import SamplingParams


def _execute(engine, operation, args, kwargs):
    if isinstance(operation, str):
        return getattr(engine, operation)(*args, **kwargs)
    return operation(engine, *args, **kwargs)


def _worker(rank, config, rendezvous, connection):
    try:
        if config.device == "cuda":
            torch.cuda.set_device(rank)
        dist.init_process_group("nccl" if config.device == "cuda" else "gloo",
                                init_method=rendezvous, rank=rank, world_size=config.ranks,
                                timeout=timedelta(minutes=30))
        engine = Engine(config)
        connection.send((True, None))
        while True:
            message = connection.recv()
            if message is None:
                break
            operation, args, kwargs = message
            connection.send((True, _execute(engine, operation, args, kwargs)))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


class PhysisServe:
    """One prompt or a list of concurrent streams, on one model and GPU pool.

    Multi-GPU execution starts its own processes; run scripts under a
    ``if __name__ == '__main__':`` guard. Parallelism is over denoising stages.
    For multiple nodes, use ``torchrun ... --module physisserve.launch script.py``;
    the launcher supplies the GPU pool size without changing the script.
    """

    def __init__(self, model, *, pipeline_parallel_size=1, strategy="virtual_rank",
                 max_batch_size=None, device="cuda", encoder_device="cpu",
                 batch_costs=None, admission_window=None):
        from .launch import _workers

        if _workers is not None:
            world = dist.get_world_size()
            if type(pipeline_parallel_size) is not int or pipeline_parallel_size not in (1, world):
                raise ValueError("pipeline_parallel_size must be 1 or match the launch WORLD_SIZE")
            pipeline_parallel_size = world
        self.config = Config(model, ranks=pipeline_parallel_size, strategy=strategy,
                             max_batch=max_batch_size, device=device, encoder_device=encoder_device,
                             batch_costs=batch_costs, admission_window=admission_window)
        self._engine = None
        self._processes, self._connections = [], []
        self._rendezvous = None
        self._closed = False
        self._external = _workers
        if _workers is None and device == "cuda" and torch.cuda.device_count() < pipeline_parallel_size:
            raise ValueError(f"requested {pipeline_parallel_size} GPUs, found {torch.cuda.device_count()}")
        try:
            if _workers is not None:
                _workers.load(self, self.config, model)
            elif pipeline_parallel_size == 1:
                self._engine = Engine(self.config)
            else:
                self._rendezvous = TemporaryDirectory(prefix="physisserve-")
                address = Path(self._rendezvous.name, "rendezvous").as_uri()
                context = mp.get_context("spawn")
                for rank in range(pipeline_parallel_size):
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker, args=(rank, self.config, address, child))
                    process.start()
                    child.close()
                    self._connections.append(parent)
                    self._processes.append(process)
                self._receive()
        except BaseException:
            self.close()
            raise
        atexit.register(self.close)

    def _receive(self):
        pending = dict(enumerate(self._connections))
        results = [None] * len(pending)
        while pending:
            sentinels = [process.sentinel for process in self._processes]
            ready = wait([*pending.values(), *sentinels], timeout=1800)
            if not ready:
                raise TimeoutError("GPU workers did not respond within 30 minutes")
            for rank, connection in tuple(pending.items()):
                if connection in ready:
                    ok, result = connection.recv()
                    if not ok:
                        raise RuntimeError(f"worker {rank} failed:\n{result}")
                    results[rank] = result
                    del pending[rank]
            if pending and any(process.sentinel in ready for process in self._processes):
                raise RuntimeError("a GPU worker exited before completing the request")
        return results

    def _call(self, operation, *args, **kwargs):
        if self._closed:
            raise RuntimeError("PhysisServe is closed")
        try:
            if self._external is not None:
                return self._external.call(operation, *args, **kwargs)
            if self._engine is not None:
                return [_execute(self._engine, operation, args, kwargs)]
            for connection in self._connections:
                connection.send((operation, args, kwargs))
            return self._receive()
        except BaseException:
            self.close()
            raise

    def _requests(self, prompts, sampling_params):
        if not isinstance(prompts, (str, list, tuple)):
            raise TypeError("prompts must be a string or a list/tuple of strings")
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, (SamplingParams, list, tuple)):
            raise TypeError("sampling_params must be SamplingParams or a list/tuple of SamplingParams")
        params = ([sampling_params] * len(prompts) if isinstance(sampling_params, SamplingParams)
                  else list(sampling_params))
        if len(params) != len(prompts) or any(not isinstance(item, SamplingParams) for item in params):
            raise ValueError("provide one SamplingParams or one per prompt")
        return [param.request(prompt, self.config) for prompt, param in zip(prompts, params)]

    def generate(self, prompts, sampling_params=None):
        """Return ordered CPU latents and, for decoding modes, uint8 THWC videos.

        Every prompt has its own KV cache and seed.
        ``SamplingParams.mode`` selects DiT only, decoding after all streams finish,
        or decoding completed chunks during denoising. Offline decoding first
        saves each stream's latents to disk; ``latent_path`` identifies the file.
        """
        requests = self._requests(prompts, sampling_params)
        if not requests:
            return []
        if not self.config.supports_online_decode and any(r.mode == "online_decode" for r in requests):
            raise ValueError(f"{self.config.backend} uses a whole-sequence VAE; online_decode is unsupported")
        if any(r.mode != "dit_only" and r.chunks < self.config.min_decode_chunks for r in requests):
            raise ValueError(f"{self.config.backend} video decoding requires at least {self.config.min_decode_chunks} chunks")
        self._call("prepare", requests)
        outputs = self._call("run")[0]
        offline = [index for index, request in enumerate(requests)
                   if request.mode == "offline_decode" and outputs[index].get("status") != "rejected"]
        if offline:
            decoded = self.decode([outputs[index] for index in offline])
            for index, output in zip(offline, decoded):
                outputs[index] = output
        return outputs

    def decode(self, outputs):
        """Decode previously generated latents after inference or benchmarking."""
        return self._call("decode", outputs)[0]

    def close(self):
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        if self._external is not None:
            self._external.release(self)
            return
        for connection in self._connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = monotonic() + 5
        for process in self._processes:
            process.join(timeout=max(0, deadline - monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.terminate()
        for process in self._processes:
            process.join(timeout=1)
        for connection in self._connections:
            connection.close()
        self._connections.clear()
        self._processes.clear()
        self._engine = None
        if self._rendezvous is not None:
            self._rendezvous.cleanup()
        if self.config.device == "cuda":
            torch.cuda.empty_cache()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
