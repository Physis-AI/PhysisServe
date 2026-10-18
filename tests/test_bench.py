import pytest

from bench import summarize


def row(stream, times, batch):
    return {"times": {stream: list(enumerate(times))}, "max_batch": batch}


def test_benchmark_aggregates_same_rank_gaps_and_discards_bootstrap():
    ranks = [[row(0, [10, 100, 110, 120, 130], 1), row(0, [20, 140, 160, 180, 200], 1)],
             [row(1, [50, 150, 170, 190, 210], 4), row(1, [60, 160, 190, 220, 250], 3)]]
    result = summarize(ranks, streams=2, chunks=5, repeats=2, skip_chunks=1)
    assert result["completion_ms"] == 230
    assert result["stream_gap_ms"] == [15, 25]
    assert result["chunk_gap_ms"] == 20
    assert result["chunks_per_second"] == pytest.approx(10 * 1000 / 230)
    assert result["max_batch"] == 4


@pytest.mark.parametrize("ranks, message", [
    ([[row(0, [1, 2, 3], 1)], [row(0, [1, 2, 3], 1)]], "one owner"),
    ([[{"times": {}, "max_batch": 1}]], "missing stream"),
    ([[row(0, [1, 2], 1)]], "missing or duplicate"),
    ([[row(0, [1, 3, 2], 1)]], "nondecreasing"),
])
def test_benchmark_rejects_incomplete_or_invalid_timing(ranks, message):
    with pytest.raises(ValueError, match=message):
        summarize(ranks, streams=1, chunks=3, repeats=1, skip_chunks=0)


def test_benchmark_requires_a_steady_interval():
    with pytest.raises(ValueError, match="at least two"):
        summarize([], streams=1, chunks=3, repeats=1, skip_chunks=2)


def test_batched_completion_bursts_do_not_inflate_throughput():
    ranks = [[row(0, [10, 100, 100, 100, 100, 200, 200, 200, 200], 4)]]
    result = summarize(ranks, streams=1, chunks=9, repeats=1, skip_chunks=0)
    assert result["chunk_gap_ms"] == pytest.approx(190 / 8)
