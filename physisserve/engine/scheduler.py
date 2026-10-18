"""Cell identities and schedule lookup shared by vertical strategies.

Ticks specify dependency order, not equal GPU execution durations. Concrete
stage placement and batch dispatch policies live in the strategies package.
"""
from dataclasses import dataclass

from .sampler import DENOISING_STEPS


@dataclass(frozen=True, order=True)
class Task:
    tick: int
    rank: int
    chunk: int
    step: int
    block: int

    @property
    def key(self) -> tuple[int, int, int]:
        """(chunk, step, block): the cell's identity and the key of the KV page it writes."""
        return self.chunk, self.step, self.block


class Schedule:
    """Every cell of a request placed on (tick, rank); identical on every rank."""

    def __init__(self, tasks: list[Task], blocks: int):
        self.tasks = sorted(tasks)
        self.blocks = blocks
        self.task_of = {t.key: t for t in self.tasks}
        self.ticks = self.tasks[-1].tick + 1
        self.by_tick: list[list[Task]] = [[] for _ in range(self.ticks)]
        for task in self.tasks:
            self.by_tick[task.tick].append(task)
        if any(len({t.rank for t in batch}) != len(batch) for batch in self.by_tick):
            raise AssertionError("two cells scheduled on one rank in the same tick")

    def task(self, chunk: int, step: int, block: int) -> Task:
        return self.task_of[chunk, step, block]

    def output(self, chunk: int) -> Task:
        """The cell after which chunk's latent is final: the last block of the last denoising pass."""
        return self.task(chunk, DENOISING_STEPS - 1, self.blocks - 1)

    def successor(self, task: Task) -> Task | None:
        """The cell that consumes this cell's output: next block, else block 0 of the next stage."""
        if task.block + 1 < self.blocks:
            return self.task(task.chunk, task.step, task.block + 1)
        if task.step + 1 < DENOISING_STEPS:
            return self.task(task.chunk, task.step + 1, 0)
        return None
