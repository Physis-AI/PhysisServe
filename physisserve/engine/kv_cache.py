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

"""Same-Step KV cache scheduling: which pages each cell reads, and where they live and for how long.

Every cell (chunk, step, block) writes one KV page, so a chunk has one page version per stage and
block; a page is keyed by that (chunk, step, block) triple. Every cell reads each history chunk's
version from the same denoising step, ``(c', s, b)``, from the block schedule rather than arrival order.
The history chunks a cell reads are the ``history`` chunks before it plus the first chunk of the video,
which stays readable however far the window has moved (the fixed attention sink).
Each (step, block) has one fixed rank across chunks. KVCache plans only this process's pages and
releases each page after its last local read; historical KV never transfers between ranks.
"""
import torch

from .scheduler import Schedule, Task

ATTENTION_SINK_CHUNKS = 1
HISTORY_CHUNKS = 18


def history_chunks(chunk: int) -> list[int]:
    """Chunks read by ``chunk``: fixed sink chunk 0, then the bounded recent history."""
    first = max(0, chunk - HISTORY_CHUNKS)
    return [*range(min(ATTENTION_SINK_CHUNKS, first)), *range(first, chunk)]


class Cache:
    """What a block sees: history pages in chunk order, then its own page.

    The model computes this block's K/V and calls ``cache(k, v)`` once, receiving the concatenated
    K/V for attention. The page it wrote is published by KVCache afterwards and must not be modified
    in place by the model. Pages are [B, heads, tokens, head_dim]. Positional backends deliberately use
    page order to construct the fixed effective window ``[sink | recent | current]``.
    """

    def __init__(self, history: list[tuple[torch.Tensor, torch.Tensor]]):
        self.history = history
        self.page: tuple[torch.Tensor, torch.Tensor] | None = None

    def __call__(self, key: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.page = (key, value)
        if not self.history:
            return key, value
        parts = [*self.history, (key, value)]
        return tuple(torch.cat([p[i] for p in parts], dim=2) for i in (0, 1))


class BatchCache:
    """Stack equal-length windows, retaining only each member's own pre-RoPE pages."""

    def __init__(self, caches, batch_size):
        self.caches, self.batch_size = caches, batch_size

    def __call__(self, key, value):
        caches, batch = self.caches, self.batch_size
        history_tokens = sum(page[0].shape[2] for page in caches[0].history)
        window_shape = (*key.shape[:2], history_tokens + key.shape[2], key.shape[3])
        packed = (key.new_empty(window_shape), value.new_empty(window_shape)) if history_tokens else (key, value)
        for member, (cache, k, v) in enumerate(zip(caches, key.split(batch), value.split(batch))):
            # A sink can outlive its batch peers; retain no views into the packed storage.
            cache.page = (k.clone(), v.clone())
            if history_tokens:
                parts = [*cache.history, cache.page]
                for component in (0, 1):
                    torch.cat([page[component] for page in parts], dim=2,
                              out=packed[component][member * batch:(member + 1) * batch])
        return packed


class KVCache:
    """Local Same-Step page lifetimes for a schedule with fixed stage ownership.

    ``offset`` and ``sink`` serve the chunks after a separately finished first chunk: the schedule's chunk ``i`` is
    the request's chunk ``i + offset``, and every read of a request chunk below ``offset`` takes the fixed page
    ``sink[(chunk, step, block)]`` (held for the whole request; with a clean sink the same page for every step).
    """

    def __init__(self, schedule: Schedule, is_local, offset: int = 0, sink=None):
        tasks = [task for task in schedule.tasks if is_local(task.rank)]
        self.sink = {} if sink is None else sink
        self.reads = {
            task: [(chunk - offset, task.step, task.block)
                   for chunk in history_chunks(task.chunk + offset)]
            for task in tasks
        }
        # A page has exactly one home, including when one process simulates
        # several ranks. No remote holders or global schedule need to survive.
        last_use = {task.key: task.tick for task in tasks}
        for task, keys in self.reads.items():
            for key in keys:
                if key[0] < 0:
                    continue
                last_use[key] = max(last_use[key], task.tick)
        self.offset = offset
        self.expiry = [[] for _ in range(schedule.ticks)]
        for key, tick in sorted(last_use.items()):
            self.expiry[tick].append(key)
        self.pages = {}

    def context(self, task: Task) -> Cache:
        keys = self.reads[task]
        return Cache([self.sink[key[0] + self.offset, key[1], key[2]] if key[0] < 0 else self.pages[key]
                      for key in keys])

    def publish(self, task: Task, cache: Cache):
        self.pages[task.key] = cache.page

    def release(self, tick: int):
        for key in self.expiry[tick]:
            del self.pages[key]
