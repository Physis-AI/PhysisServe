import pytest

from physisserve.engine.strategies import StageEqual
from physisserve.engine.strategies.placement import (
    Candidate, DeviceBudget, RankDemand, StreamCandidates, plan_pool,
)


def candidate(name, demands, gap=10):
    return Candidate(name, StageEqual(len(demands)), tuple(RankDemand(*item) for item in demands), gap)


def problem(name, *candidates):
    return StreamCandidates(name, 20, candidates)


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


def test_no_arrivals_leave_pool_unused():
    empty = plan_pool([], [])
    assert empty.placements == empty.loads == ()
    assert empty.rejected == {}
    assert plan_pool([], [DeviceBudget(10)]).loads == (RankDemand(0, 0.0),)


def test_empty_pool_rejects_arrivals_without_searching():
    arrivals = [problem("p", candidate("one", [(1, 0.1)])),
                problem("q", candidate("unknown", [(1, 0.1)], None))]
    plan = plan_pool(arrivals, [])
    assert plan.placements == plan.loads == ()
    assert set(plan.rejected) == {"p", "q"}
    assert all(search["nodes"] == 0 for search in plan.diagnostics["search"].values())


@pytest.mark.parametrize("name", ["", "same"])
def test_candidate_names_do_not_restrict_search(name):
    p = problem("p", candidate(name, [(11, 0.1)]), candidate(name, [(1, 0.1)]))
    plan = plan_pool([p], [DeviceBudget(10)])
    assert not plan.rejected
    assert plan.placements[0].candidate is p.candidates[1]
    assert plan.diagnostics["search"]["p"]["tried_candidates"] == (name, name)


@pytest.mark.parametrize("ranks,discrepancies,search_limit", [
    (ranks, discrepancies, search_limit)
    for ranks in [1, 2, 5] for search_limit in [0, 1, 7, 10000]
    for discrepancies in ([0] if ranks == 1 or search_limit == 0 else [0, 1, 2])
])
def test_generated_placement_covers_arrivals_and_obeys_resource_and_search_budgets(
        ranks, discrepancies, search_limit):
    arrivals = [problem(str(index),
                        candidate("", [(2 + index % 3, 0.125)] * (1 + index % 4)),
                        candidate("", [(3 + index % 3, 0.25)] * (1 + index % 3)),
                        candidate("too-slow", [(1, 0.01)], 21))
                for index in range(9)]
    budgets = [DeviceBudget(8 + rank, 0.75) for rank in range(ranks)]
    plan = plan_pool(arrivals, budgets, lds_discrepancies=discrepancies, search_limit=search_limit)
    placed = [item.stream_id for item in plan.placements]
    assert len(set(placed)) == len(placed)
    assert set(placed).isdisjoint(plan.rejected)
    assert set(placed) | set(plan.rejected) == {item.stream_id for item in arrivals}
    expected_memory, expected_utilization = [0] * ranks, [0.] * ranks
    for item in plan.placements:
        source = arrivals[int(item.stream_id)]
        assert any(item.candidate is choice for choice in source.candidates)
        assert item.candidate.max_gap_ms <= source.slo_ms
        assert len(item.rank_map) == len(item.candidate.demands) == item.candidate.layout.num_ranks(3)
        for rank, demand in zip(item.rank_map, item.candidate.demands):
            assert 0 <= rank < ranks
            expected_memory[rank] += demand.memory_bytes
            expected_utilization[rank] += demand.utilization
    assert len(plan.loads) == ranks
    for rank, (load, budget) in enumerate(zip(plan.loads, budgets)):
        assert load.memory_bytes == expected_memory[rank] <= budget.memory_bytes
        assert load.utilization == expected_utilization[rank] <= budget.utilization
    for search in plan.diagnostics["search"].values():
        assert 0 <= search["nodes"] <= search_limit
