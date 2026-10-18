"""Static placement and schedule invariants with synthetic costs, not GPU timing claims."""
from collections import Counter

import pytest

from tests.test_scheduler import assert_dag
from physisserve.engine.strategies import StageEqual, VirtualRank
from physisserve.engine.strategies.virtual_rank import _group_costs, _improve, _plan


def test_interleaved_stream_layouts_do_not_replan_between_chunks(monkeypatch):
    calls = []
    solve = VirtualRank._solve

    def counted(layout, blocks):
        calls.append((layout, blocks))
        return solve(layout, blocks)

    monkeypatch.setattr(VirtualRank, "_solve", counted)
    layouts = (VirtualRank(2), VirtualRank(3))
    for _ in range(3):
        for layout in layouts:
            layout.schedule(4, 6)
    assert calls == [(layouts[0], 4), (layouts[1], 4)]


def assert_plan(layout, blocks, chunks):
    placement = layout.placement(blocks)
    assert set(placement) == {"stage_owners", "rank_groups", "rank_compute_work", "max_rank_compute_work"}
    owners, groups = placement["stage_owners"], placement["rank_groups"]
    assert set(owners) == set(range(layout.num_ranks(blocks)))
    seen = Counter()
    for rank, rows in enumerate(groups):
        work = 0.
        for block, steps in rows:
            assert 1 <= len(steps) <= layout.max_batch
            work += layout.batch_costs[block][len(steps) - 1]
            for step in steps:
                assert owners[step * blocks + block] == rank
                seen[step, block] += 1
        assert work == pytest.approx(placement["rank_compute_work"][rank])
    assert seen == Counter({(step, block): 1 for step in range(4) for block in range(blocks)})
    assert placement["max_rank_compute_work"] == max(placement["rank_compute_work"])
    plan = assert_dag(layout, blocks, chunks)
    assert plan.admission_window == (layout.admission_window or 4 * blocks)
    for task in plan.tasks:
        assert task.rank == owners[task.step * blocks + task.block]
        if task.step == task.block == 0 and task.chunk >= plan.admission_window:
            assert plan.output(task.chunk - plan.admission_window).tick < task.tick
    for tick in plan.by_tick_batches:
        for batch in tick:
            assert [task.chunk for task in batch] == sorted({task.chunk for task in batch})
    return placement, plan


@pytest.mark.parametrize("max_batch", [1, 2, 3, 4])
@pytest.mark.parametrize("blocks", [1, 2, 3, 5])
def test_small_rank_counts_cover_dag_credits_and_local_kv(blocks, max_batch):
    costs = [[1 + block / 10, 1.1 + block / 10, 3.5 + block / 10, 4.2 + block / 10]
             for block in range(blocks)]
    for ranks in range(1, 4 * blocks + 1):
        assert_plan(VirtualRank(ranks, max_batch, costs), blocks, 4 * blocks + 3)


@pytest.mark.parametrize("ranks", [1, 20])
def test_one_gpu_and_one_stage_per_gpu(ranks):
    placement, _ = assert_plan(VirtualRank(ranks, batch_costs=[[1, 1.4, 1.7, 2]] * 5), 5, 23)
    assert placement["max_rank_compute_work"] == pytest.approx(10 if ranks == 1 else 1)


def test_per_block_grouping_uses_costs_without_requiring_full_batches():
    layout = VirtualRank(1, batch_costs=[[1, 3, 6, 10], [1, 1.2, 3, 4]])
    placement, _ = assert_plan(layout, 2, 12)
    sizes = [[len(steps) for block, steps in placement["rank_groups"][0] if block == index]
             for index in range(2)]
    assert sizes == [[1, 1, 1, 1], [2, 2]]
    assert placement["max_rank_compute_work"] == pytest.approx(6.4)


def test_unprofiled_default_has_deterministic_same_block_batches():
    layout = VirtualRank(2)
    assert layout.batch_costs is None
    expected = VirtualRank(2, batch_costs=[[1, 1, 1, 1]] * 3)
    assert layout.placement(3) == expected.placement(3)
    assert layout.schedule(3, 24).by_tick_batches == expected.schedule(3, 24).by_tick_batches
    assert max(len(batch) for tick in layout.schedule(3, 24).by_tick_batches for batch in tick) == 4


def test_noncontiguous_step_groups_keep_identities_and_sorted_chunks():
    costs = [[2., 3.9, 4., 5.]] * 2
    work, parts = _group_costs(costs, 4)
    fixed = _plan((0, 1, 1, 0, 0, 1, 1, 0), 2, work, parts)
    assert fixed["rank_groups"][0] == ((0, (0, 2)), (1, (1, 3)))

    class Probe(VirtualRank):
        def _solution(self, blocks):
            return fixed

    _, plan = assert_plan(Probe(2, batch_costs=costs), 2, 20)
    assert any({task.step for task in batch} == {0, 2} for tick in plan.by_tick_batches for batch in tick)


def test_local_moves_improve_tied_bottlenecks_but_reject_equal_cost_plateaus():
    work, parts = _group_costs([[2., 3.9, 4., 5.]] * 2, 4)
    owners = (0, 0, 0, 0, 1, 1, 1, 1)
    initial = _plan(owners, 2, work, parts)
    result = _improve(owners, 2, work, parts)
    assert result["max_rank_compute_work"] < initial["max_rank_compute_work"]
    assert result["max_rank_compute_work"] == pytest.approx(5)
    work, parts = _group_costs([[1., 2., 3., 4.]] * 3, 4)
    owners = (0, 1, 2) * 4
    assert _improve(owners, 3, work, parts)["stage_owners"] == owners


def test_scale_invariance_determinism_and_mutation_isolation():
    costs = [[1., 1.1, 1.2, 1.3]] * 5
    layout = VirtualRank(2, batch_costs=costs)
    expected = layout.placement(5)
    costs[0][0] = 99
    for scale in (1e-15, 1., 1e15):
        other = VirtualRank(2, batch_costs=[[scale * value for value in row]
                                             for row in layout.batch_costs])
        assert other.placement(5)["stage_owners"] == expected["stage_owners"]
    changed = layout.placement(5)
    changed["stage_owners"] = (99,) * 20
    assert layout.placement(5) == expected
    before = layout.schedule(5, 23).tasks
    layout.schedule(5, 2)
    assert layout.schedule(5, 23).tasks == before


@pytest.mark.parametrize("costs", [[], "1234", iter([[1, 2, 3, 4]]), [None], ["1234"],
                                   [iter([1, 2, 3, 4])], [[1, 2]], [[0, 1, 2, 3]], [[True, 1, 2, 3]],
                                   [[float("inf")] * 4], [[float("nan")] * 4], [[1e308] * 4]])
def test_invalid_costs(costs):
    with pytest.raises(ValueError):
        VirtualRank(batch_costs=costs)


def test_tuple_cost_curves():
    assert VirtualRank(batch_costs=((1, 2, 3, 4),)).placement(1) == VirtualRank(
        batch_costs=[[1, 2, 3, 4]]).placement(1)


def test_invalid_strategy_options():
    for field in ("ranks", "max_batch", "admission_window"):
        for value in (0, -1, True, 1.5):
            with pytest.raises(ValueError):
                VirtualRank(batch_costs=[[1, 2, 3, 4]], **{field: value})
    with pytest.raises(ValueError, match="max_batch"):
        VirtualRank(max_batch=5)
    for ranks in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="ranks"):
            StageEqual(ranks)


def test_model_dependent_bounds():
    layout = VirtualRank(batch_costs=[[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="one row"):
        layout.placement(2)
    with pytest.raises(ValueError, match="number of stages"):
        VirtualRank(5, batch_costs=[[1, 2, 3, 4]]).placement(1)
    with pytest.raises(ValueError, match="stage count"):
        VirtualRank(1, batch_costs=[[1, 2, 3, 4]], admission_window=3).placement(1)
    with pytest.raises(ValueError, match="ranks <= four times blocks"):
        StageEqual(5).num_ranks(1)
