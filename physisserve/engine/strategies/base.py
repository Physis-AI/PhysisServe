"""Shared plan validation and static-placement helpers for single-stream strategies."""

from collections.abc import Sequence

from ...contract import DENOISING_STEPS
from ..scheduler import Schedule, Task


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _ranks(ranks: int | None) -> None:
    if ranks is not None:
        _positive_int(ranks, "ranks")


def _batch_limit(max_batch: int) -> None:
    if type(max_batch) is not int or not 1 <= max_batch <= DENOISING_STEPS:
        raise ValueError(f"max_batch must be an integer from 1 to {DENOISING_STEPS}")


def _equal_parts(length: int, count: int) -> tuple[range, ...]:
    quotient, remainder = divmod(length, count)
    cuts = [0]
    for rank in range(count):
        cuts.append(cuts[-1] + quotient + (rank < remainder))
    return tuple(range(left, right) for left, right in zip(cuts, cuts[1:]))


def _owners(parts: tuple[range, ...]) -> tuple[int, ...]:
    return tuple(rank for rank, part in enumerate(parts) for _ in part)


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
        _positive_int(blocks, "blocks")
        _positive_int(chunks, "chunks")
        if admission_window is not None:
            _positive_int(admission_window, "admission_window")
        self.blocks = blocks
        self.chunks = chunks
        self.admission_window = admission_window
        self.by_tick_batches = tuple(tuple(tuple(batch) for batch in tick) for tick in by_tick_batches)
        self.ticks = len(self.by_tick_batches)
        self.by_tick = [[task for batch in tick for task in batch] for tick in self.by_tick_batches]
        self.tasks = sorted(task for tick in self.by_tick for task in tick)
        self.task_of = {}
        for tick, batches in enumerate(self.by_tick_batches):
            for batch in batches:
                if not 1 <= len(batch) <= DENOISING_STEPS:
                    raise ValueError("a batch must have between one and four cells")
                if len({(task.rank, task.block) for task in batch}) != 1:
                    raise ValueError("all batch members must share a rank and block")
                for task in batch:
                    if any(type(value) is not int for value in
                           (task.tick, task.rank, task.chunk, task.step, task.block)):
                        raise ValueError("task coordinates must be integers")
                    if (task.tick != tick or task.rank < 0 or not 0 <= task.chunk < chunks
                            or not 0 <= task.step < DENOISING_STEPS or not 0 <= task.block < blocks):
                        raise ValueError("task coordinates are outside the schedule")
                    if task.key in self.task_of:
                        raise ValueError(f"duplicate scheduled cell: {task.key}")
                    self.task_of[task.key] = task
        if len(self.tasks) != chunks * DENOISING_STEPS * blocks:
            raise ValueError("schedule must cover every (chunk, step, block) exactly once")
        for task in self.tasks:
            stage = task.step * blocks + task.block
            predecessors = []
            if stage:
                step, block = divmod(stage - 1, blocks)
                predecessors.append(self.task(task.chunk, step, block))
            if task.chunk:
                predecessors.append(self.task(task.chunk - 1, task.step, task.block))
            if (admission_window is not None and stage == 0
                    and task.chunk >= admission_window):
                predecessors.append(self.output(task.chunk - admission_window))
            if any(previous.tick >= task.tick for previous in predecessors):
                raise ValueError(f"cell {task.key} requires all predecessors in prior ticks")


class _StaticPlacement:
    def _stage_owners(self, blocks: int) -> tuple[int, ...]:
        raise NotImplementedError

    def num_ranks(self, blocks: int) -> int:
        return max(self._stage_owners(blocks)) + 1

    def rank_blocks(self, rank: int, blocks: int) -> tuple[int, ...]:
        owners = self._stage_owners(blocks)
        ranks = max(owners) + 1
        if type(rank) is not int or not 0 <= rank < ranks:
            raise ValueError(f"rank {rank} is outside the {ranks} ranks of this layout")
        return tuple(sorted({stage % blocks for stage, owner in enumerate(owners) if owner == rank}))
