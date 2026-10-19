import json
from types import SimpleNamespace

import pytest

import bench
from bench import summarize
from physisserve import PhysisServe
from physisserve.engine.strategies.placement import DeviceBudget
from tests.test_api import fake_config


def row(*times, batch=1):
    return {"times": {index: list(enumerate(values)) for index, values in enumerate(times)},
            "max_batch": batch}


@pytest.mark.parametrize("blocks,chunks", [(30, 481), (40, 641), (48, 769), (50, 801)])
def test_default_long_run_profiles_every_layout_and_resumes_without_remeasurement(monkeypatch, tmp_path, blocks, chunks):
    calls, closed = [], []

    def make_service(model, **kwargs):
        service = SimpleNamespace(config=fake_config(model, ranks=kwargs["pipeline_parallel_size"]))
        service.config.num_blocks = blocks

        def call(operation, *args):
            calls.append((operation.__name__, args))
            if operation is bench.devices:
                return [dict(rank=rank, name="fake GPU", free_bytes=10000, total_bytes=10000) for rank in range(2)]
            if operation is bench.calibrate:
                params, prompt, batch = args
                assert (params.mode, params.num_chunks, batch) == ("dit_only", chunks, 4)
                return [{"rank": 0, "error": None,
                         "samples_ms": {f"{block}:{size}": [float(size)]
                                        for block in range(blocks) for size in range(1, batch + 1)}}]
            if operation is bench.release:
                return [None, None]
            prompts, params = args[:2]
            assert prompts == ["A corgi running along a beach at sunset."] * len(params)
            assert all((param.mode, param.num_chunks) == ("dit_only", chunks) for param in params)
            if operation is bench.prepare_candidate:
                return [{"rank": rank, "status": "ready"} for rank in range(2)]
            repeats = args[5] if operation is bench.profile_candidate else args[3]
            warmups = args[4] if operation is bench.profile_candidate else args[2]
            assert (warmups, repeats) == (1, 3)
            trials = [[row(*(range(1, chunks + 1) for _ in params)) for _ in range(repeats)],
                      [row() for _ in range(repeats)]]
            if operation is bench.profile_candidate:
                return [{"rank": rank, "trials": values, "peak_bytes": 100,
                         "compute_ms_per_chunk": 1.0} for rank, values in enumerate(trials)]
            assert operation is bench.measure
            return trials

        service._call = call
        service.close = lambda: closed.append(True)
        return service

    monkeypatch.setattr(bench, "PhysisServe", make_service)
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bench, "identity", lambda *args: {})
    arguments = ["bench.py", "fake", "--gpus", "2", "--slo-ms", "100", "--max-streams", "2"]
    output = tmp_path / "results"
    monkeypatch.setattr("sys.argv", arguments + ["--output", str(output)])
    bench.main()
    profiles = [args[2][0].candidates[0] for name, args in calls if name == "profile_candidate"]
    assert [candidate.name for candidate in profiles] == ["stage_equal-r2", "virtual_rank-r2",
                                                         "stage_equal-r1", "virtual_rank-r1"]
    assert sum(name == "calibrate" for name, _ in calls) == 1
    assert sum(name == "measure" for name, _ in calls) == 4
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "bounded_search_completed"
    assert report["protocol"]["initial_chunks"] == chunks
    assert report["protocol"]["profile_ranks"] == [1, 2]
    assert all(value["stop_reason"] == "stream_limit" for value in report["capacity"].values())
    calls.clear()
    monkeypatch.setattr("sys.argv", arguments + ["--resume", str(output)])
    bench.main()
    assert [name for name, _ in calls] == ["devices"]
    assert closed == [True, True]


def test_common_middle_excludes_fill_and_drain_and_aggregates_repeats():
    ranks = [[
        row([0, 100, 200, 210, 220, 230, 240, 241, 242],
            [0, 100, 200, 220, 240, 260, 280, 281, 282], batch=4),
        row([0, 200, 300, 320, 340, 360, 380, 381, 382],
            [0, 200, 300, 340, 380, 420, 460, 461, 462], batch=3),
    ], [row(), row()]]
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


def test_fixed_comparison_calibrates_and_keeps_each_stream_on_the_full_pool(monkeypatch, tmp_path):
    calls, closed = [], []
    service = SimpleNamespace(config=fake_config("fake", ranks=2))

    def call(operation, *args):
        calls.append((operation.__name__, args))
        if operation is bench.devices:
            return [{"rank": rank, "name": "fake", "free_bytes": 1000, "total_bytes": 1000}
                    for rank in range(2)]
        if operation is bench.calibrate:
            return [{"error": None, "samples_ms": {f"{block}:{size}": [float(size)]
                                                     for block in range(2) for size in range(1, 5)}}]
        if operation is bench.configure:
            return [None, None]
        prompts, params = args[:2]
        assert len(prompts) == len(params) == 3
        assert [param.seed for param in params] == [0, 1, 2]
        assert all((param.mode, param.num_chunks) == ("dit_only", 33) for param in params)
        if operation is bench.prepare_candidate:
            assert args[2] is None
            return [{"status": "ready"}, {"status": "ready"}]
        assert operation is bench.measure and args[4] is None
        assert args[2:4] == (1, 3)
        return [[row(*(range(33) for _ in params)) for _ in range(3)], [row() for _ in range(3)]]

    service._call, service.close = call, lambda: closed.append(True)
    monkeypatch.setattr(bench, "PhysisServe", lambda *args, **kwargs: service)
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(bench, "identity", lambda *args: {})
    output = tmp_path / "fixed"
    monkeypatch.setattr("sys.argv", ["bench.py", "fake", "--gpus", "2", "--streams", "3", "--output", str(output)])
    bench.main()
    assert sum(name == "calibrate" for name, _ in calls) == 1
    assert [args[0] for name, args in calls if name == "configure"] == ["stage_equal", "virtual_rank"]
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "completed" and report["steady_speedup"] == 1
    assert all(point["plan"] == "same full-pool layout per stream" for point in report["comparison"].values())
    assert closed == [True]


def test_calibration_costs_use_measured_medians_for_each_native_batch():
    rows = [{"error": None, "samples_ms": {"0:1": [1.0, 3.0], "0:2": [2.0, 4.0]}},
            {"error": None, "samples_ms": {"1:1": [4.0, 6.0], "1:2": [7.0, 9.0]}}]
    assert bench.cost_table(rows, blocks=2, max_batch=2) == [[2.0, 3.0], [5.0, 8.0]]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "empty", "zero", "nan", "inf", "error"])
def test_incomplete_or_invalid_calibration_is_not_replaced_with_estimates(failure):
    rows = [{"error": None, "samples_ms": {"0:1": [1.0], "0:2": [2.0]}}]
    if failure == "missing":
        del rows[0]["samples_ms"]["0:2"]
    elif failure == "duplicate":
        rows.append({"error": None, "samples_ms": {"0:1": [1.0]}})
    elif failure == "error":
        rows[0]["error"] = "out of memory"
    else:
        invalid = {"empty": [], "zero": [0.0], "nan": [float("nan")], "inf": [float("inf")]}
        rows[0]["samples_ms"]["0:2"] = invalid[failure]
    with pytest.raises(ValueError, match="incomplete block calibration"):
        bench.cost_table(rows, blocks=1, max_batch=2)


def test_startup_failure_cannot_be_replaced_by_stable_middle_slo_pass():
    traces = [[row([0, 200, 210, 220, 230, 240, 250, 260, 270])]]
    result = bench.qualify(traces, counts=[9], repeats=1, trim=2, slo_ms=100)
    assert result["startup_max_gap_ms"] == result["all_max_gap_ms"] == 200
    assert result["max_gap_ms"] == 10
    assert result["steady_slo_pass"] and result["stable"]
    assert not result["strict_slo_pass"] and not result["qualified"]


@pytest.mark.parametrize("times", [
    [0, 1, 2, 3, 4, 5, 6, 7],
    [0, 1, 2, 3, float("nan"), 5, 6, 7, 8],
    [0, 1, 2, 3, 4, 3, 6, 7, 8],
])
def test_invalid_delivery_trace_is_not_an_slo_measurement(times):
    with pytest.raises(ValueError, match="invalid stream delivery trace"):
        bench.qualify([[row(times)]], counts=[9], repeats=1, trim=2, slo_ms=10)


def test_window_extension_gives_fast_stream_more_chunks():
    traces = [[row(range(13), range(0, 52, 4))]]
    counts = bench.longer_counts(traces, [13, 13], trim=2)
    assert counts[0] > counts[1] >= 25
    extended = [[row(range(counts[0]), range(0, counts[1] * 4, 4))]]
    assert bench.qualify(extended, counts, repeats=1, trim=2, slo_ms=10)["qualified"]


def test_both_methods_use_measured_resource_demands_not_exclusive_gpu_reservations():
    args = SimpleNamespace(max_batch_size=4, slo_ms=100)
    profiles = [{"name": f"{method}-r2", "method": method, "ranks": 2, "status": "measured",
                 "demands": [{"peak_bytes": 10, "compute_ms_per_chunk": 25},
                             {"peak_bytes": 20, "compute_ms_per_chunk": 50}],
                 "summary": {"max_gap_ms": 10, "all_max_gap_ms": 200, "strict_slo_pass": False}}
                for method in ("stage_equal", "virtual_rank")]
    profiles.append({"name": "stage_equal-r1", "method": "stage_equal", "status": "unverified_memory"})
    for method in ("stage_equal", "virtual_rank"):
        candidates = bench.measured_candidates(profiles, method, [[1, 2, 3, 4]], args)
        assert len(candidates) == 1
        assert [demand.memory_bytes for demand in candidates[0].demands] == [12, 24]
        assert [demand.utilization for demand in candidates[0].demands] == pytest.approx([0.3, 0.6])
        assert candidates[0].max_gap_ms == 10


def test_capacity_search_does_not_assume_slo_passes_are_monotonic(monkeypatch, tmp_path):
    calls = []

    def point(service, params, problems, budgets, args, path, trim, physical_budgets=None):
        count = len(problems)
        calls.append((path.stem, count))
        return {"streams": count, "status": "measured", "qualified": count == 2,
                "steady_slo_pass": True}

    monkeypatch.setattr(bench, "run_point", point)
    args = SimpleNamespace(max_batch_size=4, max_streams=3, slo_ms=100)
    result = bench.search_capacity(None, None, [], [], [], args, tmp_path, trim=2)
    assert [count for _, count in calls] == [1, 2, 3, 1, 2, 3]
    for record in result.values():
        assert record["largest_verified_streams"] == 2
        assert record["largest_steady_only_streams"] == 3
        assert record["stop_reason"] == "stream_limit"


def test_placement_failure_is_unknown_not_a_measured_slo_failure(monkeypatch, tmp_path):
    service = SimpleNamespace(_call=lambda *args: pytest.fail("a rejected placement must not execute"))
    args = SimpleNamespace(max_batch_size=4, max_streams=3, slo_ms=100)
    result = bench.search_capacity(service, None, [DeviceBudget(100)], [], [], args, tmp_path, trim=2)
    for record in result.values():
        assert record["stop_reason"] == "not_admitted"
        assert record["largest_verified_streams"] == 0
        point, = record["points"]
        assert point["status"] == "not_admitted" and point["streams"] == 1
        assert "strict_slo_pass" not in point
        assert point["plan"]["diagnostics"]["global_slo_verified"] is False


def test_extending_overlap_does_not_erase_an_observed_slo_violation(tmp_path):
    attempts = []

    def call(operation, *args):
        if operation is bench.prepare_candidate:
            return [{"status": "ready"}]
        parameters = args[1]
        attempts.append([param.num_chunks for param in parameters])
        if len(attempts) == 1:
            return [[row([0, 20, 21, 22, 23, 24, 25, 26, 27], range(100, 109))]]
        return [[row(*(range(param.num_chunks) for param in parameters))]]

    service = SimpleNamespace(_call=call)
    args = SimpleNamespace(streams=2, prompt="test", warmups=1, repeats=1, slo_ms=10)
    params = bench.SamplingParams(num_chunks=9, mode="dit_only")
    point = bench.run_point(service, params, None, (), args, tmp_path / "point.json", trim=2)
    assert len(attempts) == 2
    assert point["status"] == "measured" and point["steady_slo_pass"]
    assert point["all_max_gap_ms"] == 20
    assert not point["strict_slo_pass"] and not point["qualified"]


def test_resume_mismatch_does_not_modify_existing_report(monkeypatch, tmp_path):
    report = tmp_path / "report.json"
    original = '{"status":"completed","settings":{},"identity":{}}\n'
    report.write_text(original)
    service = SimpleNamespace(config=fake_config("fake"), close=lambda: None,
                              _call=lambda *args: [{"name": "test", "free_bytes": 1000, "total_bytes": 1000}])
    monkeypatch.setattr(bench, "PhysisServe", lambda *args, **kwargs: service)
    monkeypatch.setattr(bench, "identity", lambda *args: {})
    monkeypatch.setattr(bench.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("sys.argv", ["bench.py", "fake", "--resume", str(tmp_path)])
    with pytest.raises(ValueError, match="resume requires"):
        bench.main()
    assert report.read_text() == original
