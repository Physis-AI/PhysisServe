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

from ..contract import ATTENTION_SINK_CHUNKS, HISTORY_CHUNKS
from .scheduler import Schedule, Task


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
        if self.page is not None:
            raise RuntimeError("a block forward may write its KV only once")
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError("KV pages must be [B, heads, tokens, head_dim] with matching shapes")
        self.page = (key, value)
        if not self.history:
            return key, value
        parts = [*self.history, (key, value)]
        return tuple(torch.cat([p[i] for p in parts], dim=2) for i in (0, 1))


class KVCache:
    """Local Same-Step page lifetimes for a schedule with fixed stage ownership.

    ``offset`` and ``sink`` serve the chunks after a separately finished first chunk: the schedule's chunk ``i`` is
    the request's chunk ``i + offset``, and every read of a request chunk below ``offset`` takes the fixed page
    ``sink[(chunk, step, block)]`` (held for the whole request; with a clean sink the same page for every step).
    """

    def __init__(self, schedule: Schedule, is_local, offset: int = 0, sink=None):
        homes, tasks = {}, []
        for task in schedule.tasks:
            stage = task.step, task.block
            if homes.setdefault(stage, task.rank) != task.rank:
                raise ValueError("KVCache requires a fixed rank for each (step, block)")
            if is_local(task.rank):
                tasks.append(task)
        if type(offset) is not int or offset < 0 or (offset and sink is None):
            raise ValueError("offset must be a nonnegative integer, and a positive offset needs sink pages")
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
                    if (key[0] + offset, *key[1:]) not in self.sink:
                        raise ValueError(f"{task} reads sink page {key[0] + offset, *key[1:]}, which is missing")
                    continue
                if schedule.task(*key).tick >= task.tick:
                    raise AssertionError(f"{task} reads a page that is not finished before its tick")
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
        if cache.page is None:
            raise RuntimeError(f"block {task.block} did not write its K/V into the cache")
        self.pages[task.key] = cache.page

    def release(self, tick: int):
        for key in self.expiry[tick]:
            del self.pages[key]

    def close(self):
        if self.pages:
            raise AssertionError("KV pages are still held at the end of the request")
