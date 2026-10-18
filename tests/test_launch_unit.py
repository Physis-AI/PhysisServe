from types import SimpleNamespace

import pytest
import torch

from physisserve.engine.engine import Engine
from physisserve import launch
from tests.test_engine import FakeConfig


def test_engine_uses_local_cuda_device_not_global_rank(monkeypatch):
    selected, barriers = [], []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", selected.append)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 10)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 16)
    monkeypatch.setattr(torch.distributed, "get_backend", lambda: "nccl")
    monkeypatch.setattr(torch.distributed, "barrier", lambda **kwargs: barriers.append(kwargs))
    config = FakeConfig(ranks=16)
    config.device = "cuda"
    group = object()
    engine = Engine(config, control_group=group)
    assert engine.rank == 10 and engine.device == torch.device("cuda:2")
    assert engine.control_group is group and selected == [torch.device("cuda:2")]
    assert barriers == [{"device_ids": [2]}]


def test_worker_resolves_hugging_face_id_in_its_own_cache(monkeypatch):
    resolved = []

    def resolve(source):
        resolved.append(source)
        return "/node-1/cache/model"

    monkeypatch.setattr(launch.dist, "get_rank", lambda: 8)
    monkeypatch.setattr(launch, "resolve_model", resolve)
    monkeypatch.setattr(launch, "Engine", lambda config, control_group:
                        SimpleNamespace(config=config, group=control_group))
    config = SimpleNamespace(model="/node-0/cache/model")
    group = object()
    workers = launch.Workers("cpu", group, store=None)
    assert workers.execute("_load", (config, "org/model"), {}) is None
    assert resolved == ["org/model"]
    assert workers.engine.config.model == "/node-1/cache/model"
    assert workers.engine.group is group


def test_decode_payload_and_root_results_stay_local(monkeypatch):
    sent, gathered = [], []
    output = object()
    monkeypatch.setattr(launch.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(launch.dist, "broadcast_object_list", lambda payload, **kwargs: sent.append(payload))
    monkeypatch.setattr(launch.dist, "gather_object", lambda value, results, **kwargs: gathered.append(value))
    workers = launch.Workers("cpu", None, store=None)
    monkeypatch.setattr(workers, "execute", lambda operation, args, kwargs: args[0])
    assert workers.call("decode", output) == [output, None]
    assert sent == [[("decode", (None,), {})]]
    assert gathered == [None]


def test_launcher_requires_torchrun_environment(monkeypatch, capsys):
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr("sys.argv", ["launch.py", "example.py"])
    with pytest.raises(SystemExit, match="2"):
        launch.main()
    assert "start this module with torchrun" in capsys.readouterr().err


def test_failure_monitor_exits_even_when_stderr_is_broken(monkeypatch):
    def broken_print(*args, **kwargs):
        raise OSError("stderr is closed")

    def exit_process(code):
        raise SystemExit(code)

    monkeypatch.setattr("builtins.print", broken_print)
    monkeypatch.setattr(launch.os, "_exit", exit_process)
    store = SimpleNamespace(check=lambda keys: True, get=lambda key: b"worker failure")
    finished = SimpleNamespace(wait=lambda timeout: False, is_set=lambda: False)
    with pytest.raises(SystemExit, match="1"):
        launch._watch_failure(store, finished)
