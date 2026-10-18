r"""Run an unchanged inference script on a torchrun-managed GPU pool.

On each node, use the same command with its own --node-rank, for example:
  torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
    --master-addr=HOST --master-port=29500 --module physisserve.launch example.py

Only rank zero runs the script. The launch world size replaces the script's
default pipeline_parallel_size=1; an explicit non-default size must match it.
All nodes need the script/package and access to the model (a local directory or
a Hugging Face ID, resolved in each worker's own cache). Outputs stay on rank zero.
One model owns the pool at a time; close it before constructing another.
Caught worker errors stop the pool; native crashes still rely on backend timeouts.
"""
import argparse
from datetime import timedelta
from multiprocessing.connection import wait
import os
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import monotonic
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from .config import resolve_model
from .engine.engine import Engine


_workers = None


def ranks(requested):
    if _workers is None:
        return requested
    world = dist.get_world_size()
    if type(requested) is not int or requested not in (1, world):
        raise ValueError("pipeline_parallel_size must be 1 or match the launch WORLD_SIZE")
    return world


def load(config, source):
    workers = _workers if _workers is not None else Workers(config.device)
    workers.load(config, source)
    return workers


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
            connection.send((True, _execute(engine, *message)))
    except BaseException:
        connection.send((False, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        connection.close()


def _watch_failure(store, finished):
    # Static torchrun agents supervise only their own node. Notify peers even
    # when their main thread is blocked in a data collective after a Python error.
    while not finished.wait(0.1):
        try:
            if not store.check(["error"]):
                continue
            error = store.get("error").decode()
        except Exception as exc:
            error = f"lost launcher failure store: {exc}"
        if not finished.is_set():
            try:
                print(error, file=sys.stderr, flush=True)
            finally:
                os._exit(1)


class Workers:
    def __init__(self, device, group=None, store=None):
        self.device, self.group, self.store = device, group, store
        self.engine = None
        self.processes, self.connections = [], []
        self.rendezvous = None
        self.failed = False

    def load(self, config, source):
        if self.engine is not None or self.processes:
            raise RuntimeError("close the current PhysisServe before opening another on this launch")
        if config.device != self.device:
            raise ValueError("PhysisServe device must match the launch --device")
        if self.group is None and self.device == "cuda" and torch.cuda.device_count() < config.ranks:
            raise ValueError(f"requested {config.ranks} GPUs, found {torch.cuda.device_count()}")
        try:
            if self.group is not None:
                path = Path(source).expanduser()
                source = str(path.resolve()) if path.is_dir() else str(source)
                self.call("_load", config, source)
            elif config.ranks == 1:
                self.engine = Engine(config)
            else:
                self.rendezvous = TemporaryDirectory(prefix="physisserve-")
                address = Path(self.rendezvous.name, "rendezvous").as_uri()
                context = mp.get_context("spawn")
                for rank in range(config.ranks):
                    parent, child = context.Pipe()
                    process = context.Process(target=_worker, args=(rank, config, address, child))
                    process.start()
                    child.close()
                    self.connections.append(parent)
                    self.processes.append(process)
                self._receive()
        except BaseException:
            self.close()
            raise

    def execute(self, operation, args, kwargs):
        if operation == "_load":
            config, source = args
            if dist.get_rank() != 0:
                config.model = resolve_model(source)
            self.engine = Engine(config, control_group=self.group)
            return None
        if operation == "_close":
            self.engine = None
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return None
        return _execute(self.engine, operation, args, kwargs)

    def _receive(self):
        pending = dict(enumerate(self.connections))
        results = [None] * len(pending)
        while pending:
            sentinels = [process.sentinel for process in self.processes]
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
            if pending and any(process.sentinel in ready for process in self.processes):
                raise RuntimeError("a GPU worker exited before completing the request")
        return results

    def call(self, operation, *args, **kwargs):
        if self.group is None:
            if self.engine is not None:
                return [_execute(self.engine, operation, args, kwargs)]
            for rank, connection in enumerate(self.connections):
                remote_args = (None,) if operation == "decode" and rank != 0 else args
                connection.send((operation, remote_args, kwargs))
            return self._receive()
        if self.failed:
            raise RuntimeError("distributed workers failed; restart the launch")
        # Never issue cleanup collectives after a partial command: peers may
        # still be in data transfers. The failure monitor terminates the pool.
        self.failed = True
        try:
            remote_args = (None,) if operation == "decode" else args
            dist.broadcast_object_list([(operation, remote_args, kwargs)], src=0, group=self.group)
            result = self.execute(operation, args, kwargs)
            results = [None] * dist.get_world_size()
            # Keep rank zero's large latent/video results local, outside serialization.
            dist.gather_object(None, results, dst=0, group=self.group)
            results[0] = result
        except BaseException:
            self.report_failure()
            raise
        self.failed = False
        return results

    def listen(self):
        try:
            while True:
                self.failed = True
                message = [None]
                dist.broadcast_object_list(message, src=0, group=self.group)
                if message[0] is None:
                    self.failed = False
                    return
                result = self.execute(*message[0])
                dist.gather_object(result, dst=0, group=self.group)
                self.failed = False
        except BaseException:
            self.report_failure()
            raise

    def report_failure(self):
        self.store.compare_set("error", "", f"rank {os.environ['RANK']} failed:\n{traceback.format_exc()}")

    def close(self):
        if self.group is not None:
            if self.engine is not None and not self.failed:
                self.call("_close")
            self.engine = None
            return
        for connection in self.connections:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        deadline = monotonic() + 5
        for process in self.processes:
            process.join(timeout=max(0, deadline - monotonic()))
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=1)
        for connection in self.connections:
            connection.close()
        self.connections.clear()
        self.processes.clear()
        self.engine = None
        if self.rendezvous is not None:
            self.rendezvous.cleanup()
            self.rendezvous = None
        if self.device == "cuda":
            torch.cuda.empty_cache()


def main():
    global _workers
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("script", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not all(key in os.environ for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")):
        parser.error("start this module with torchrun --module physisserve.launch")
    script = args.script.resolve()
    if not script.is_file():
        parser.error(f"script does not exist: {script}")
    if args.device == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        if not 0 <= local_rank < torch.cuda.device_count():
            parser.error("LOCAL_RANK exceeds the number of visible GPUs on this node")
        torch.cuda.set_device(local_rank)
    timeout = timedelta(minutes=30)
    dist.init_process_group("nccl" if args.device == "cuda" else "gloo", init_method="env://", timeout=timeout)
    group = dist.new_group(backend="gloo", timeout=timeout) if args.device == "cuda" else dist.group.WORLD
    store = dist.TCPStore(os.environ["MASTER_ADDR"], int(os.environ["MASTER_PORT"]),
                          is_master=False, timeout=timedelta(seconds=10))
    prefix = f"physisserve/{os.environ.get('TORCHELASTIC_RUN_ID', 'run')}/{os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')}"
    store = dist.PrefixStore(prefix, store)
    finished = Event()
    Thread(target=_watch_failure, args=(store, finished), daemon=True).start()
    _workers = Workers(args.device, group, store)
    # Workers can import benchmark callbacks without running the user's script.
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args.args]
    try:
        if dist.get_rank() == 0:
            try:
                runpy.run_path(str(script), run_name="__main__")
            finally:
                if not _workers.failed:
                    _workers.close()
                    dist.broadcast_object_list([None], src=0, group=group)
        else:
            _workers.listen()
    finally:
        if not _workers.failed:
            finished.set()
            dist.barrier(group=group)
            dist.destroy_process_group()
        _workers = None


if __name__ == "__main__":
    # -m executes this file as __main__; keep serve's import on the same instance.
    sys.modules["physisserve.launch"] = sys.modules[__name__]
    main()
