# Required Notice: Copyright 2026 Physis AI, Inc. and/or its affiliates
#
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
#
# Licensed under the PolyForm Noncommercial License 1.0.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://polyformproject.org/licenses/noncommercial/1.0.0
#
# This software is provided "AS IS", without warranties or conditions.
# See the License for permissions, limitations, and liability terms.

import pytest

from physisserve.engine.strategies import StageEqual
from physisserve.engine.strategies.placement import (
    Candidate, DeviceBudget, RankDemand, StreamCandidates, plan_pool,
)


def candidate(name, demands, gap=10, mean=None):
    return Candidate(name, StageEqual(len(demands)), tuple(RankDemand(*item) for item in demands), gap, mean)


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


def test_two_streams_use_seven_gpus_when_measured_layouts_benefit():
    choices = tuple(candidate(f"r{ranks}", [(6, 0.25)] * ranks, mean=mean)
                    for ranks, mean in ((1, 12), (3, 4.2), (4, 3.4)))
    arrivals = [problem(str(index), *choices) for index in range(2)]
    budgets = [DeviceBudget(10)] * 7
    plan = plan_pool(arrivals, budgets)
    assert not plan.rejected
    assert [len(item.rank_map) for item in plan.placements] == [4, 3]
    assert sorted(rank for item in plan.placements for rank in item.rank_map) == list(range(7))
    assert all(load == RankDemand(6, 0.25) for load in plan.loads)
    assert plan == plan_pool(arrivals, budgets)


@pytest.mark.parametrize("gpus,accepted", [(1, 1), (3, 2)])
def test_fast_layout_never_evicts_an_admitted_stream(gpus, accepted):
    choices = (candidate("small", [(6, 0.2)], mean=10),
               candidate("fast", [(6, 0.2)] * 3, mean=2))
    plan = plan_pool([problem(str(index), *choices) for index in range(2)], [DeviceBudget(10)] * gpus)
    assert len(plan.placements) == accepted and len(plan.rejected) == 2 - accepted
    assert all(item.candidate is choices[0] for item in plan.placements)


def test_spare_gpus_spread_measured_coalesced_and_shared_ranks():
    choice = candidate("two", [(2, 0.1)] * 2, mean=10)
    plan = plan_pool([problem(str(index), choice) for index in range(2)], [DeviceBudget(10)] * 4)
    assert sorted(rank for item in plan.placements for rank in item.rank_map) == list(range(4))
    assert all(item.candidate is choice for item in plan.placements)
    assert plan.loads == (RankDemand(2, 0.1),) * 4


def test_spare_gpu_search_exhaustion_keeps_the_admitted_plan():
    small = candidate("small", [(6, 0.2)], mean=10)
    fast = candidate("fast", [(6, 0.2)] * 2, mean=2)
    plan = plan_pool([problem("p", small, fast)], [DeviceBudget(10)] * 3, search_limit=1)
    assert not plan.rejected and plan.placements[0].candidate is small
    assert plan.loads == (RankDemand(6, 0.2), RankDemand(0, 0.0), RankDemand(0, 0.0))
    assert plan.diagnostics["spare_gpu_search"] == {"nodes": 1, "exhausted": True}


@pytest.mark.parametrize("case", ["memory", "slo", "slower", "unknown", "unprofiled"])
def test_spare_gpus_do_not_justify_an_unverified_or_worse_upgrade(case):
    small = candidate("small", [(6, 0.2)], mean=None if case == "unprofiled" else 10)
    larger = candidate("larger", [(11 if case == "memory" else 6, 0.2)] * 2,
                       gap=21 if case == "slo" else 10,
                       mean=None if case == "unknown" else 11 if case == "slower" else 2)
    plan = plan_pool([problem("p", small, larger)], [DeviceBudget(10)] * 3)
    assert plan.placements[0].candidate is small
    assert plan.placements[0].rank_map == (0,)
    assert plan.loads == (RankDemand(6, 0.2), RankDemand(0, 0.0), RankDemand(0, 0.0))


@pytest.mark.parametrize("ranks,discrepancies,search_limit", [
    (ranks, discrepancies, search_limit)
    for ranks in [1, 2, 5] for search_limit in [0, 1, 7, 10000]
    for discrepancies in ([0] if ranks == 1 or search_limit == 0 else [0, 1, 2])
])
def test_generated_placement_covers_arrivals_and_obeys_resource_and_search_budgets(
        ranks, discrepancies, search_limit):
    arrivals = [problem(str(index),
                        candidate("", [(2 + index % 3, 0.125)] * (1 + index % 4), mean=8),
                        candidate("", [(3 + index % 3, 0.25)] * (1 + index % 3), mean=4),
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
    assert 0 <= plan.diagnostics["spare_gpu_search"]["nodes"] <= search_limit
