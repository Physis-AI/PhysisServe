import pytest

import bench
from bench import summarize
from physisserve import PhysisServe
from tests.test_api import fake_config


def row(stream, times, batch):
    return {"times": {stream: list(enumerate(times))}, "max_batch": batch}


def test_benchmark_inputs_are_dit_only_for_both_strategies(monkeypatch):
    services = []

    def make_service(model, **kwargs):
        service = PhysisServe.__new__(PhysisServe)
        service.config = fake_config(model, **kwargs)

        def measure(operation, prompts, params, warmups, repeats):
            assert operation is bench.measure
            assert prompts == ["A corgi running along a beach at sunset."]
            assert [param.mode for param in params] == ["dit_only"]
            return [[row(0, [1, 2, 3], 1)]]

        service._call = measure
        service.close = lambda: None
        services.append(service)
        return service

    monkeypatch.setattr(bench, "PhysisServe", make_service)
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("sys.argv", ["bench.py", "fake", "--chunks", "3", "--skip-chunks", "0",
                                     "--warmups", "0", "--repeats", "1"])
    bench.main()
    assert [service.config.strategy for service in services] == ["stage_equal", "virtual_rank"]


def test_benchmark_aggregates_same_rank_gaps_and_discards_bootstrap():
    ranks = [[row(0, [10, 100, 110, 120, 130], 1), row(0, [20, 140, 160, 180, 200], 1)],
             [row(1, [50, 150, 170, 190, 210], 4), row(1, [60, 160, 190, 220, 250], 3)]]
    result = summarize(ranks, streams=2, chunks=5, repeats=2, skip_chunks=1)
    assert result["completion_ms"] == 230
    assert result["stream_gap_ms"] == [15, 25]
    assert result["chunk_gap_ms"] == 20
    assert result["chunks_per_second"] == pytest.approx(10 * 1000 / 230)
    assert result["max_batch"] == 4


def test_benchmark_requires_a_steady_interval_at_cli(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["bench.py", "fake", "--chunks", "3", "--skip-chunks", "2"])
    with pytest.raises(SystemExit, match="2"):
        bench.main()
    assert "chunks > skip-chunks + 1" in capsys.readouterr().err


def test_batched_completion_bursts_do_not_inflate_throughput():
    ranks = [[row(0, [10, 100, 100, 100, 100, 200, 200, 200, 200], 4)]]
    result = summarize(ranks, streams=1, chunks=9, repeats=1, skip_chunks=0)
    assert result["chunk_gap_ms"] == pytest.approx(190 / 8)
