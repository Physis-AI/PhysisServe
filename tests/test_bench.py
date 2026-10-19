import pytest

import bench
from bench import summarize
from physisserve import PhysisServe
from tests.test_api import fake_config


def row(*times, batch=1):
    return {"times": {index: list(enumerate(values)) for index, values in enumerate(times)},
            "max_batch": batch}


@pytest.mark.parametrize("blocks,chunks", [(30, 481), (40, 641), (48, 769), (50, 801)])
def test_default_long_run_is_model_sized_and_dit_only(monkeypatch, blocks, chunks):
    services = []

    def make_service(model, **kwargs):
        assert kwargs["max_batch_size"] == (1 if kwargs["strategy"] == "stage_equal" else 4)
        service = PhysisServe.__new__(PhysisServe)
        service.config = fake_config(model, **kwargs)
        service.config.num_blocks = blocks

        def measure(operation, prompts, params, warmups, repeats):
            assert operation is bench.measure
            assert prompts == ["A corgi running along a beach at sunset."]
            assert [(param.mode, param.num_chunks) for param in params] == [("dit_only", chunks)]
            assert (warmups, repeats) == (1, 3)
            return [[row(range(1, chunks + 1)) for _ in range(repeats)]]

        service._call = measure
        service.close = lambda: None
        services.append(service)
        return service

    monkeypatch.setattr(bench, "PhysisServe", make_service)
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("sys.argv", ["bench.py", "fake"])
    bench.main()
    assert [service.config.strategy for service in services] == ["stage_equal", "virtual_rank"]


def test_common_middle_excludes_fill_and_drain_and_aggregates_repeats():
    ranks = [[row(), row()], [
        row([0, 100, 200, 210, 220, 230, 240, 241, 242],
            [0, 100, 200, 220, 240, 260, 280, 281, 282], batch=4),
        row([0, 200, 300, 320, 340, 360, 380, 381, 382],
            [0, 200, 300, 340, 380, 420, 460, 461, 462], batch=3),
    ]]
    result = summarize(ranks, streams=2, chunks=9, repeats=2, trim=2)
    assert result["completion_ms"] == 372
    assert result["stream_gap_ms"] == [15, 30]
    assert result["chunk_gap_ms"] == 22.5
    assert result["steady_chunks_per_second"] == 112.5
    assert result["full_chunks_per_second"] == pytest.approx(18 * 1000 / 372)
    assert result["window_ms"] == 60
    assert result["max_gap_ms"] == 40
    assert result["half_drift"] == 0
    assert result["max_batch"] == 4


def test_early_finisher_cannot_inflate_other_streams_middle_rate():
    ranks = [[row([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 101, 102],
                  [0, 20, 40, 60, 80, 100, 120, 121, 122, 123, 124, 125, 126])]]
    result = summarize(ranks, streams=2, chunks=13, repeats=1, trim=2)
    assert result["window_ms"] == 60
    assert result["stream_gap_ms"] == [10, 20]
    assert result["steady_chunks_per_second"] == 150


def test_batched_completion_bursts_keep_zero_gaps_and_trim_chunk_boundaries():
    ranks = [[row([0, 100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300], batch=4)]]
    result = summarize(ranks, streams=1, chunks=13, repeats=1, trim=2)
    assert result["chunk_gap_ms"] == 25
    assert result["steady_chunks_per_second"] == 40
    assert result["max_gap_ms"] == 100


@pytest.mark.parametrize("slow", [
    [100, 110, 120, 130, 140, 150, 160, 170, 180],
    [0, 20, 40, 60, 61, 62, 63, 64, 65],
])
def test_missing_or_too_short_common_middle_does_not_fall_back_to_tail(slow):
    ranks = [[row([0, 10, 20, 30, 40, 50, 60, 61, 62], slow)]]
    with pytest.raises(ValueError, match="common middle window; increase --chunks"):
        summarize(ranks, streams=2, chunks=9, repeats=1, trim=2)


def test_short_run_is_rejected_before_inference_and_service_is_closed(monkeypatch, capsys):
    service = PhysisServe.__new__(PhysisServe)
    service.config = fake_config("fake")
    service.config.num_blocks = 30
    closed = []
    service.close = lambda: closed.append(True)
    service._call = lambda *args: pytest.fail("short benchmark must not run inference")
    monkeypatch.setattr(bench, "PhysisServe", lambda *args, **kwargs: service)
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("sys.argv", ["bench.py", "fake", "--chunks", "40"])
    with pytest.raises(SystemExit, match="2"):
        bench.main()
    assert "--chunks must be >= 481" in capsys.readouterr().err
    assert closed == [True]
