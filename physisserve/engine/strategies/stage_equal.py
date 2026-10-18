"""Equal contiguous stage ownership with singleton execution."""

from ...contract import DENOISING_STEPS
from ..scheduler import Task
from .base import BatchSchedule, _StaticPlacement, _equal_parts, _owners, _positive_int, _ranks


class StageEqual(_StaticPlacement):
    """Equal contiguous portions of the flattened four-pass virtual stages.

    Each rank runs one cell per logical tick, without batching. Local order is
    chunk first, then stage; placing several stages on a rank already provides
    basic time multiplexing. Dependency-safe order is compiled offline, not
    selected from an online ready queue. No costs or delivery target are used.

    ``ranks=None`` assigns
    each (step, block) its own rank and emits one completed chunk per tick after
    pipeline fill. Indivisible stage counts differ by at most one per rank.
    """

    def __init__(self, ranks: int | None = None):
        _ranks(ranks)
        self.ranks = ranks

    def _stage_owners(self, blocks: int) -> tuple[int, ...]:
        _positive_int(blocks, "blocks")
        stages = DENOISING_STEPS * blocks
        ranks = stages if self.ranks is None else self.ranks
        if ranks > stages:
            raise ValueError("stage-equal placement requires ranks <= four times blocks")
        return _owners(_equal_parts(stages, ranks))

    def schedule(self, blocks: int, chunks: int) -> BatchSchedule:
        _positive_int(chunks, "chunks")
        owners = self._stage_owners(blocks)
        ends = [-1] * (max(owners) + 1)
        previous_chunk = [-1] * len(owners)
        by_tick = []
        for chunk in range(chunks):
            previous = -1
            for stage, rank in enumerate(owners):
                tick = max(previous, previous_chunk[stage], ends[rank]) + 1
                step, block = divmod(stage, blocks)
                while len(by_tick) <= tick:
                    by_tick.append([])
                by_tick[tick].append((Task(tick, rank, chunk, step, block),))
                previous = previous_chunk[stage] = ends[rank] = tick
        return BatchSchedule(by_tick, blocks, chunks)
