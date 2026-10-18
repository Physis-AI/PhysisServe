import pytest

from physisserve.engine.strategies import StageEqual, VirtualRank
from physisserve.engine.strategies.placement import (
    Candidate, DeviceBudget, MappedLayout, RankDemand, StreamCandidates, plan_pool,
)


def candidate(name, demands, gap=10):
    return Candidate(name, StageEqual(len(demands)), tuple(RankDemand(*item) for item in demands), gap)


def problem(name, *candidates):
    return StreamCandidates(name, 2, 12, 20, candidates)


def test_candidate_allows_only_builtin_strategy_types():
    class CustomLayout:
        def num_ranks(self, blocks):
            pytest.fail("an unapproved layout must not be called")

        rank_blocks = schedule = num_ranks

    class CustomStageEqual(StageEqual):
        pass

    class CustomVirtualRank(VirtualRank):
        pass

    for layout in (CustomLayout(), CustomStageEqual(1), CustomVirtualRank(1)):
        with pytest.raises(ValueError):
            Candidate("custom", layout, (RankDemand(1, 0.1),), 1.0)


def test_unknown_or_failing_single_stream_gap_is_not_admitted():
    p = problem("p", candidate("unknown", [(1, 0.1)], None), candidate("slow", [(1, 0.1)], 21))
    plan = plan_pool([p], [DeviceBudget(10)])
    assert not plan.placements
    assert "verified" in plan.rejected["p"]
    assert plan.loads == (RankDemand(0, 0.0),)
    assert plan.diagnostics["global_slo_verified"] is False


def test_coalesced_ranks_charge_all_memory_and_utilization():
    a = problem("a", candidate("pair", [(3, 0.2), (3, 0.2)]))
    b = problem("b", candidate("one", [(3, 0.2)]))
    c = problem("c", candidate("one", [(3, 0.2)]))
    plan = plan_pool([a, b, c], [DeviceBudget(10)])
    assert [item.stream_id for item in plan.placements] == ["a", "b"]
    assert plan.placements[0].rank_map == (0, 0)
    assert set(plan.rejected) == {"c"}
    assert plan.loads[0].memory_bytes == 9
    assert plan.loads[0].utilization == pytest.approx(0.6)


def test_one_discrepancy_escapes_greedy_dead_end():
    p = problem("p", candidate("four", [(2, 0.4), (4, 0.2), (6, 0.4), (8, 0.2)]))
    budgets = [DeviceBudget(10), DeviceBudget(10)]
    assert plan_pool([p], budgets, lds_discrepancies=0).rejected
    searched = plan_pool([p], budgets, lds_discrepancies=1)
    assert not searched.rejected
    assert searched.placements[0].rank_map == (0, 1, 1, 0)
    assert [load.memory_bytes for load in searched.loads] == [10, 10]
    assert all(load.utilization <= 1 for load in searched.loads)
    assert searched == plan_pool([p], budgets, lds_discrepancies=1)


def test_best_fit_prefers_tightest_feasible_device():
    p = problem("p", candidate("one", [(4, 0.2)]))
    assert plan_pool([p], [DeviceBudget(10), DeviceBudget(5)]).placements[0].rank_map == (1,)


def test_failed_arrival_rolls_back_tentative_charges():
    a = problem("a", candidate("initial", [(2, 0.2)]))
    b = problem("b", candidate("too-large", [(6, 0.3), (6, 0.3)]))
    c = problem("c", candidate("following", [(8, 0.8)]))
    plan = plan_pool([a, b, c], [DeviceBudget(10)])
    assert [placement.stream_id for placement in plan.placements] == ["a", "c"]
    assert set(plan.rejected) == {"b"}
    assert plan.loads == (RankDemand(10, 1.0),)


def test_search_budget_exhaustion_does_not_poison_next_arrival():
    p = problem("p", candidate("two", [(2, 0.1), (2, 0.1)]))
    q = problem("q", candidate("one", [(10, 1)]))
    plan = plan_pool([p, q], [DeviceBudget(10)], search_limit=1)
    assert "feasibility unknown" in plan.rejected["p"]
    assert plan.diagnostics["search"]["p"] == dict(nodes=1, exhausted=True, tried_candidates=("two",))
    assert [placement.stream_id for placement in plan.placements] == ["q"]
    assert plan.loads == (RankDemand(10, 1),)


@pytest.mark.parametrize("layout", [StageEqual(3), VirtualRank(3)])
def test_mapping_preserves_batches_dependencies_and_admission_credits(layout):
    mapped = MappedLayout(layout, (2, 0, 2), total_ranks=4)
    original, actual = layout.schedule(3, 20), mapped.schedule(3, 20)
    assert mapped.num_ranks(3) == 4
    assert mapped.rank_blocks(1, 3) == mapped.rank_blocks(3, 3) == ()
    assert mapped.rank_blocks(2, 3) == tuple(sorted(set(layout.rank_blocks(0, 3)) | set(layout.rank_blocks(2, 3))))
    assert len(actual.tasks) == len(actual.task_of) == 20 * 4 * 3
    assert actual.admission_window == original.admission_window
    for source_tick, target_tick in zip(original.by_tick_batches, actual.by_tick_batches):
        assert len(source_tick) == len(target_tick)
        for source_batch, target_batch in zip(source_tick, target_tick):
            assert [task.key for task in source_batch] == [task.key for task in target_batch]
            for old, new in zip(source_batch, target_batch):
                assert new.rank == (2, 0, 2)[old.rank]
                assert old.tick == new.tick


def test_invalid_budgets_and_duplicate_streams_are_rejected():
    p = problem("p", candidate("one", [(1, 0.1)]))
    with pytest.raises(ValueError, match="unique"):
        plan_pool([p, p], [DeviceBudget(10)])
    with pytest.raises(ValueError):
        plan_pool([], [])
    for value in (-1, True, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            RankDemand(1, value)
    assert plan_pool([], [DeviceBudget(10)]).loads == (RankDemand(0, 0.0),)
