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
