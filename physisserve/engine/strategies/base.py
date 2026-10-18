"""Batch schedule storage and static-placement helpers for single-stream strategies."""

from collections.abc import Sequence

from ..scheduler import Schedule, Task


class BatchSchedule(Schedule):
    """Schedule-compatible cells plus explicit sequential batches per tick.

    ``by_tick_batches[tick]`` is a tuple of nonempty tuples of existing Task
    objects. Batches on the same rank execute in the given order. All producers
    must finish in a *prior* tick, even when their consumers share that rank.
    ``by_tick`` remains the flattened cell view for KV lifetime planning.
    An optional ``admission_window`` adds control edges from output chunk
    ``i - window`` to input chunk ``i``. Executors must enforce these credits;
    increasing a task's tick alone does not implement admission backpressure.
    """

    def __init__(self, by_tick_batches: Sequence[Sequence[Sequence[Task]]],
                 blocks: int, chunks: int, admission_window: int | None = None):
        self.blocks = blocks
        self.chunks = chunks
        self.admission_window = admission_window
        self.by_tick_batches = tuple(tuple(tuple(batch) for batch in tick) for tick in by_tick_batches)
        self.ticks = len(self.by_tick_batches)
        self.by_tick = [[task for batch in tick for task in batch] for tick in self.by_tick_batches]
        self.tasks = sorted(task for tick in self.by_tick for task in tick)
        self.task_of = {task.key: task for task in self.tasks}


class _StaticPlacement:
    def num_ranks(self, blocks: int) -> int:
        return max(self._stage_owners(blocks)) + 1

    def rank_blocks(self, rank: int, blocks: int) -> tuple[int, ...]:
        owners = self._stage_owners(blocks)
        return tuple(sorted({stage % blocks for stage, owner in enumerate(owners) if owner == rank}))
