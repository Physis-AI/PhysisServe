"""Same-Step KV read sets and lifetimes for static layouts."""

import pytest
import gc
import weakref
import torch

from physisserve.engine.kv_cache import KVCache, history_chunks
from physisserve.engine.sampler import DENOISING_STEPS
from physisserve.engine.strategies import VirtualRank, StageEqual
from physisserve.engine.scheduler import Schedule, Task


def assert_dag(layout, blocks, chunks):
    schedule = layout.schedule(blocks, chunks)
    assert isinstance(schedule, Schedule)
    expected = {(chunk, step, block) for chunk in range(chunks)
                for step in range(DENOISING_STEPS) for block in range(blocks)}
    assert set(schedule.task_of) == expected
    assert len(schedule.tasks) == len(expected)
    assert schedule.ticks == max(task.tick for task in schedule.tasks) + 1
    owners = {}
    for task in schedule.tasks:
        assert task.block in layout.rank_blocks(task.rank, blocks)
        assert owners.setdefault((task.step, task.block), task.rank) == task.rank
        following = schedule.successor(task)
        if following is not None:
            assert following.tick > task.tick
        if task.chunk:
            assert schedule.task(task.chunk - 1, task.step, task.block).tick < task.tick
    for tick, batches in enumerate(schedule.by_tick_batches):
        assert [task for batch in batches for task in batch] == schedule.by_tick[tick]
        for batch in batches:
            assert len({(task.rank, task.block) for task in batch}) == 1
            assert len({task.step for task in batch}) == len(batch)
            assert all(task.tick == tick for task in batch)
            assert 1 <= len(batch) <= getattr(layout, "max_batch", 1)
    kv = KVCache(schedule, lambda rank: True)
    assert all(schedule.task(*key).rank == task.rank
               for task, keys in kv.reads.items() for key in keys)
    for chunk in range(chunks):
        assert schedule.output(chunk).key == (chunk, DENOISING_STEPS - 1, blocks - 1)
    assert layout.schedule(blocks, chunks).by_tick_batches == schedule.by_tick_batches
    return schedule


LAYOUTS = [StageEqual(), StageEqual(3),
           VirtualRank(2, batch_costs=[[1, 1.2, 1.4, 1.6]] * 3),
           VirtualRank(2, max_batch=1, batch_costs=[[1]] * 3)]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_static_kv_cache_read_sets_and_lifetimes(layout):
    blocks, chunks = 3, 26
    schedule = layout.schedule(blocks, chunks)
    kv = KVCache(schedule, lambda rank: True)
    last_use = {}
    expired = set()
    published = {}
    for tick in range(schedule.ticks):
        for task in schedule.by_tick[tick]:
            assert task.block in layout.rank_blocks(task.rank, blocks)
            assert task.key not in expired
            expected = [(chunk, task.step, task.block) for chunk in history_chunks(task.chunk)]
            assert kv.reads[task] == expected
            last_use[task.key] = tick
            for key in kv.reads[task]:
                assert key not in expired
                assert schedule.task(*key).tick < tick
                assert schedule.task(*key).rank == task.rank
                last_use[key] = tick
            cache = kv.context(task)
            assert all(page is published[key] for page, key in zip(cache.history, expected))
            label = task.chunk * 100 + task.step * 10 + task.block
            page = torch.full((1, 1, 1, 1), float(label))
            _, values = cache(page, page)
            torch.testing.assert_close(values.flatten(), torch.tensor([
                float(chunk * 100 + step * 10 + block) for chunk, step, block in [*expected, task.key]
            ]))
            kv.publish(task, cache)
            published[task.key] = cache.page
            following = schedule.successor(task)
            if following is None:
                assert (task.step, task.block) == (DENOISING_STEPS - 1, blocks - 1)
            else:
                assert following.tick > tick
                assert following.chunk == task.chunk
        for key in kv.expiry[tick]:
            assert key not in expired
            assert last_use[key] == tick
            expired.add(key)
        kv.release(tick)
        assert set(kv.pages) == set(published) - expired
    assert expired == set(schedule.task_of)
    assert sum(len(entries) for entries in kv.expiry) == len(schedule.tasks)
    kv.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_fixed_sink_keeps_chunk_zero_readable(layout):
    blocks, chunks = 3, 26
    schedule = layout.schedule(blocks, chunks)
    kv = KVCache(schedule, lambda rank: True)
    for task in schedule.tasks:
        chunks_read = [key[0] for key in kv.reads[task]]
        assert chunks_read == history_chunks(task.chunk)
        if task.chunk == 0:
            last_reader = schedule.task(chunks - 1, task.step, task.block)
            assert task.key in kv.expiry[last_reader.tick]


def test_fixed_sink_history_has_no_configuration_surface():
    assert history_chunks(0) == []
    assert history_chunks(1) == [0]
    assert history_chunks(18) == list(range(18))
    assert history_chunks(19) == list(range(19))
    assert history_chunks(25) == [0, *range(7, 25)]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_fixed_home_kv_planner_keeps_only_rank_local_histories(layout):
    schedule = layout.schedule(3, 26)
    full = KVCache(schedule, lambda rank: True)
    for rank in range(layout.num_ranks(3)):
        local = KVCache(schedule, lambda other: other == rank)
        assert local.reads == {task: keys for task, keys in full.reads.items() if task.rank == rank}
        assert local.expiry == [[key for key in tick if schedule.task(*key).rank == rank]
                                for tick in full.expiry]


@pytest.mark.parametrize("local", [lambda rank: True, lambda rank: rank == 3], ids=["simulation", "one-rank"])
def test_fixed_home_cache_does_not_retain_global_schedule(local):
    schedule = StageEqual().schedule(3, 12)
    reference = weakref.ref(schedule)
    kv = KVCache(schedule, local)
    expected_keys = {task.key for task in schedule.tasks if local(task.rank)}
    del schedule
    gc.collect()
    assert reference() is None
    assert {task.key for task in kv.reads} == expected_keys


@pytest.mark.parametrize("local", [lambda rank: True, lambda rank: rank == 0, lambda rank: False],
                         ids=["simulation", "one-rank", "unowned-stage"])
def test_chunk_dependent_ownership_is_rejected(local):
    schedule = Schedule([Task(chunk * 4 + step, chunk % 2, chunk, step, 0)
                         for chunk in range(6) for step in range(4)], blocks=1)
    with pytest.raises(ValueError, match=r"fixed rank for each \(step, block\)"):
        KVCache(schedule, local)
