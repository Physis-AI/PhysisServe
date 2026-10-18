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
    """

    def __init__(self, model, *, pipeline_parallel_size=1, strategy="virtual_rank",
                 max_batch_size=4, device="cuda", encoder_device="cpu",
                 rope_backend="native", batch_costs=None, admission_window=None):
        self.config = Config(str(model), ranks=pipeline_parallel_size, strategy=strategy,
                             max_batch=max_batch_size, device=device, encoder_device=encoder_device,
                             rope_backend=rope_backend, batch_costs=batch_costs,
                             admission_window=admission_window)
        self._engine = None
        self._processes, self._connections = [], []
        self._rendezvous = None
        self._closed = False
        if device == "cuda" and torch.cuda.device_count() < pipeline_parallel_size:
            raise ValueError(f"requested {pipeline_parallel_size} GPUs, found {torch.cuda.device_count()}")
        try:
            if pipeline_parallel_size == 1:
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
            if self._engine is not None:
                return [_execute(self._engine, operation, args, kwargs)]
            for connection in self._connections:
                connection.send((operation, args, kwargs))
            return self._receive()
        except BaseException:
            self.close()
            raise

    def _requests(self, prompts, sampling_params):
        prompts = [prompts] if isinstance(prompts, str) else list(prompts)
        if sampling_params is None:
            sampling_params = SamplingParams()
        params = ([sampling_params] * len(prompts) if isinstance(sampling_params, SamplingParams)
                  else list(sampling_params))
        if len(params) != len(prompts) or any(not isinstance(item, SamplingParams) for item in params):
            raise ValueError("provide one SamplingParams or one per prompt")
        return [param.request(prompt, self.config) for prompt, param in zip(prompts, params)]

    def generate(self, prompts, sampling_params=None, *, placement=None):
        """Return ordered CPU latents and, for decoding modes, uint8 THWC videos.

        Every prompt has its own KV cache and seed. ``placement`` optionally accepts
        a measured resource plan from ``physisserve.engine.strategies.placement``.
        ``SamplingParams.mode`` selects DiT only, decoding after all streams finish,
        or decoding completed chunks during denoising. Deferred decoding first
        saves each stream's latents to disk; ``latent_path`` identifies the file.
        """
        requests = self._requests(prompts, sampling_params)
        if not requests:
            return []
        if self.config.backend in ("h3", "ltx2") and any(r.mode == "stream_decode" for r in requests):
            raise ValueError(f"{self.config.backend} uses a whole-sequence VAE; stream_decode is unsupported")
        if self.config.backend == "h3" and any(r.mode != "dit_only" and r.chunks < 3 for r in requests):
            raise ValueError("H3 video decoding requires at least 3 chunks")
        self._call("prepare", requests, placement=placement)
        outputs = self._call("run")[0]
        deferred = [index for index, request in enumerate(requests)
                    if request.mode == "deferred_decode" and outputs[index].get("status") != "rejected"]
        if deferred:
            decoded = self.decode([outputs[index] for index in deferred])
            for index, output in zip(deferred, decoded):
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
