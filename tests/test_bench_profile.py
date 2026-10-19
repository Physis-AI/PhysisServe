from math import ceil
from types import SimpleNamespace

import pytest
import torch

import bench
from physisserve import SamplingParams
from physisserve.engine.engine import Engine
from physisserve.engine.strategies.placement import DeviceBudget
from tests.test_api import fake_config
from tests.test_engine import FakeDiT


class Event:
    def __init__(self, **kwargs):
        pass

    def record(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 1.0


@pytest.fixture
def cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "Event", Event)
    for name in ("synchronize", "empty_cache", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda *args: None)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args: (100000, 100000))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *args: 1234)


def dit():
    model = FakeDiT(fake_config("fake"))
    model.load([0, 1], "cpu", torch.float32)
    model.device, model.dtype = torch.device("cpu"), torch.float32
    model.prepare({"text": torch.tensor(1.0)}, (1, 2, 1, 2, 2), 100)
    return model


def test_native_calibration_measures_every_batch_cell(cuda):
    costs = bench._block_costs(dit(), 0, (1, 2, 1, 2, 2),
                              [torch.tensor(value) for value in (1000, 750, 500, 250)],
                              seed=7, max_batch=4, warmups=1, samples=3)
    assert costs == {f"0:{size}": [1.0] * 3 for size in range(1, 5)}


@pytest.mark.parametrize("fail", [False, True])
def test_profile_counts_only_native_middle_calls_and_restores_methods(cuda, monkeypatch, fail):
    model = dit()
    runner = SimpleNamespace(dit=model)
    originals = {name: getattr(model, name) for name in ("embed", "block", "block_batch", "unembed")}
    monkeypatch.setattr(bench, "measure", lambda *args: ["uninstrumented"])

    def run(**kwargs):
        from physisserve.engine.kv_cache import Cache

        assert kwargs == {"return_outputs": False}
        hidden, time = torch.ones(1, 2, 1, 2, 2), torch.tensor(500)
        model.embed(hidden, time, 0)
        model.block_batch(0, [hidden] * 2, [time] * 2, [10, 11], [Cache([]), Cache([])])
        model.block_batch(0, [hidden] * 2, [time] * 2, [19, 20], [Cache([]), Cache([])])
        model.unembed(hidden, time, 19)
        if fail:
            raise RuntimeError("native failure")

    engine = SimpleNamespace(rank=0, device="cpu", sessions=[(0, runner, {})], run=run)
    arguments = (engine, ["test"], [SamplingParams(num_chunks=30)], None, (), 1, 1, 10)
    if fail:
        with pytest.raises(RuntimeError, match="native failure"):
            bench.profile_candidate(*arguments)
    else:
        result = bench.profile_candidate(*arguments)
        assert result["compute_ms_per_chunk"] == 2 / 10
        assert result["peak_bytes"] == 1234 and result["trials"] == ["uninstrumented"]
    assert all(getattr(model, name) == original for name, original in originals.items())


@pytest.mark.parametrize("case,expected", [("ready", "ready"), ("budget", "estimated_skip"),
                                          ("local", "oom"), ("peer", "oom")])
def test_preflight_distinguishes_estimated_memory_from_synchronized_oom(cuda, monkeypatch, case, expected):
    config = fake_config("fake", ranks=2, strategy="stage_equal")
    config.timesteps = config.make_sampler().timesteps
    engine = Engine(config)
    engine.distributed, engine._barrier = True, lambda: None
    engine._bench_conditions = {"test": {"text": torch.tensor(1.0)}}
    monkeypatch.setattr("physisserve.engine.transport.Distributed", lambda device: SimpleNamespace(rank=0))
    allocated = iter((100, 700))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args: next(allocated))
    if case == "local":
        def fail_load(*args):
            raise torch.cuda.OutOfMemoryError("weights do not fit")
        monkeypatch.setattr(FakeDiT, "load", fail_load)
    gathered = []

    def gather(values, value, **kwargs):
        gathered.append(value)
        values[:] = [value, ("oom", "peer weights do not fit") if case == "peer" else ("ready", None)]

    monkeypatch.setattr(bench.dist, "all_gather_object", gather)
    budget = 1000 if case == "budget" else 10000
    result = bench.prepare_candidate(engine, ["test"], [SamplingParams(num_chunks=33, height=2, width=2)],
                                     budgets=(DeviceBudget(budget),) * 2)
    assert gathered and result["status"] == expected and result["budget_bytes"] == budget
    assert bool(engine.runners) == (case == "ready")
    if case in ("ready", "budget"):
        assert result["estimated_bytes"] == ceil((600 + 4 * (19 * 8 + 2 * 8 + 64)) * 1.2)
    if expected == "oom":
        assert any("weights do not fit" in error for error in result["errors"] if error)
