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

"""Equal contiguous stage ownership with singleton execution."""

from ..sampler import DENOISING_STEPS
from ..scheduler import Task
from .base import BatchSchedule, _StaticPlacement


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
        self.ranks = ranks

    def _stage_owners(self, blocks: int) -> tuple[int, ...]:
        stages = DENOISING_STEPS * blocks
        ranks = stages if self.ranks is None else self.ranks
        quotient, remainder = divmod(stages, ranks)
        return tuple(rank for rank in range(ranks) for _ in range(quotient + (rank < remainder)))

    def schedule(self, blocks: int, chunks: int) -> BatchSchedule:
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
