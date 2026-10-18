"""Static virtual-rank placement and a periodic same-block batch schedule."""

from collections.abc import Sequence
from functools import lru_cache

from ..sampler import DENOISING_STEPS
from ..scheduler import Task
from .base import BatchSchedule, _StaticPlacement


def _group_costs(costs, limit):
    work, parts = [], []
    for curve in costs:
        values, groups = [0.0], [()]
        for height in range(1, DENOISING_STEPS + 1):
            size = min(range(1, min(height, limit) + 1),
                       key=lambda size: (values[height - size] + curve[size - 1], -size))
            values.append(values[height - size] + curve[size - 1])
            groups.append((*groups[height - size], size))
        work.append(tuple(values))
        parts.append(tuple(groups))
    return tuple(work), tuple(parts)


def _edges(owners):
    return sum(left != right for left, right in zip(owners, owners[1:]))


def _plan(owners, ranks, work, parts):
    blocks = len(work)
    groups, counts, loads = [], [], []
    for rank in range(ranks):
        rows, row, total = [], [], 0.0
        for block in range(blocks):
            steps = tuple(step for step in range(DENOISING_STEPS) if owners[step * blocks + block] == rank)
            row.append(len(steps))
            total += work[block][len(steps)]
            offset = 0
            for size in parts[block][len(steps)]:
                rows.append((block, steps[offset:offset + size]))
                offset += size
        groups.append(tuple(rows))
        counts.append(tuple(row))
        loads.append(total)
    return dict(stage_owners=tuple(owners), rank_groups=tuple(groups), step_counts=tuple(counts),
                rank_compute_work=tuple(loads), max_rank_compute_work=max(loads))


def _rectangle_seed(work, ranks):
    """Minimax recursive cuts of contiguous step-by-block rectangles."""
    blocks = len(work)
    prefix = [[0.0] for _ in range(DENOISING_STEPS + 1)]
    for block_work in work:
        for height in range(1, DENOISING_STEPS + 1):
            prefix[height].append(prefix[height][-1] + block_work[height])
    cuts = {}

    @lru_cache(None)
    def solve(height, left, right, count):
        if count == 1:
            return prefix[height][right] - prefix[height][left]
        best, cut = float("inf"), None
        # Block cuts come first to retain step co-location on equal costs.
        candidates = [("block", middle, height * (middle - left),
                       height * (right - middle)) for middle in range(left + 1, right)]
        candidates.extend(("step", middle, middle * (right - left),
                           (height - middle) * (right - left)) for middle in range(1, height))
        for axis, middle, first_area, second_area in candidates:
            for first_count in range(max(1, count - second_area), min(count - 1, first_area) + 1):
                first = ((height, left, middle) if axis == "block" else (middle, left, right))
                second = ((height, middle, right) if axis == "block" else (height - middle, left, right))
                first_work = solve(*first, first_count)
                if first_work > best:
                    continue
                value = max(first_work, solve(*second, count - first_count))
                if value < best:
                    best, cut = value, (axis, middle, first_count)
        cuts[height, left, right, count] = cut
        return best

    solve(DENOISING_STEPS, 0, blocks, ranks)
    owners = [-1] * (DENOISING_STEPS * blocks)
    rank = 0

    def recover(step, height, left, right, count):
        nonlocal rank
        if count == 1:
            for member in range(step, step + height):
                for block in range(left, right):
                    owners[member * blocks + block] = rank
            rank += 1
            return
        axis, middle, first_count = cuts[height, left, right, count]
        if axis == "block":
            recover(step, height, left, middle, first_count)
            recover(step, height, middle, right, count - first_count)
        else:
            recover(step, middle, left, right, first_count)
            recover(step + middle, height - middle, left, right, count - first_count)

    recover(0, DENOISING_STEPS, 0, blocks, ranks)
    return tuple(owners)


def _boundary_seed(work, ranks):
    """Minimax contiguous segments of (block, step), allowing a split block."""
    blocks, stages = len(work), DENOISING_STEPS * len(work)

    @lru_cache(None)
    def segment(left, right):
        return sum(work[block][min(right, DENOISING_STEPS * (block + 1)) - max(left, DENOISING_STEPS * block)]
                   for block in range(left // DENOISING_STEPS, (right - 1) // DENOISING_STEPS + 1))

    @lru_cache(None)
    def solve(right, count):
        if count == 1:
            return segment(0, right), (0, right)
        return min((max(solve(left, count - 1)[0], segment(left, right)),
                    (*solve(left, count - 1)[1], right)) for left in range(count - 1, right))

    _, cuts = solve(stages, ranks)
    owners = [-1] * stages
    for rank, (left, right) in enumerate(zip(cuts, cuts[1:])):
        for position in range(left, right):
            block, step = divmod(position, DENOISING_STEPS)
            owners[step * blocks + block] = rank
    return tuple(owners)


def _subsets(owners, source, blocks):
    for block in range(blocks):
        for first in range(DENOISING_STEPS):
            for end in range(first + 1, DENOISING_STEPS + 1):
                cells = tuple(step * blocks + block for step in range(first, end))
                if all(owners[cell] == source for cell in cells):
                    yield cells


def _improve(initial, ranks, work, parts):
    current = _plan(initial, ranks, work, parts)
    blocks = len(work)
    for _ in range(64):
        owners, counts, loads = current["stage_owners"], current["step_counts"], current["rank_compute_work"]
        ceiling = current["max_rank_compute_work"]
        tolerance = ceiling * 1e-12  # Preserve the supplied cost unit.
        best = None
        for source in (rank for rank in range(ranks) if ceiling - loads[rank] <= tolerance):
            for target in range(ranks):
                if target == source:
                    continue
                targets = tuple(cell for cell, rank in enumerate(owners) if rank == target)
                for cells in _subsets(owners, source, blocks):
                    block, amount = cells[0] % blocks, len(cells)
                    exchanges = [None] if sum(counts[source]) > amount else []
                    if amount == 1:
                        exchanges.extend(cell for cell in targets if cell % blocks != block)
                    for other in exchanges:
                        deltas = {(source, block): -amount, (target, block): amount}
                        if other is not None:
                            deltas[source, other % blocks] = 1
                            deltas[target, other % blocks] = -1
                        proposed_loads = list(loads)
                        for (rank, index), delta in deltas.items():
                            height = counts[rank][index]
                            proposed_loads[rank] += work[index][height + delta] - work[index][height]
                        maximum = max(proposed_loads)
                        if maximum >= ceiling - tolerance:
                            continue
                        proposed = list(owners)
                        for cell in cells:
                            proposed[cell] = target
                        if other is not None:
                            proposed[other] = source
                        proposed = tuple(proposed)
                        edges = _edges(proposed)
                        if best is None or maximum < best[0] - tolerance or (
                                abs(maximum - best[0]) <= tolerance and (edges, proposed) < (best[1], best[2])):
                            best = maximum, edges, proposed
        if best is None:
            break
        following = _plan(best[2], ranks, work, parts)
        if following["max_rank_compute_work"] >= ceiling - tolerance:
            break
        current = following
    return current


class VirtualRank(_StaticPlacement):
    """Map single-stream virtual ranks to GPUs and plan their batch execution.

    A virtual rank is one (step, block), reused by all chunks with fixed KV home.

    ``batch_costs[block][size - 1]`` is a positive duration in one common unit.
    Without measured costs, each batch has equal weight: a deterministic,
    unprofiled heuristic that favors co-locating steps of the same block.
    Two partition DPs seed a bounded local search: move same-block steps or
    swap two stages to reduce maximum rank compute work. Each block chooses
    its cheapest local batch partition. Equal-cost placements prefer fewer
    cross-rank stage edges. This compute-only heuristic does not model memory,
    communication time or execution overhead, and is not a latency optimum.

    All chunks reuse the same homes and keep their stage's KV there. The
    runtime repeats a dependency-safe plan; it does not migrate KV or regroup
    ready tasks online. Ticks are dependency labels, not global barriers.
    """

    def __init__(self, ranks: int | None = None, max_batch: int = 4,
                 batch_costs: Sequence[Sequence[float]] | None = None,
                 admission_window: int | None = None):
        self.ranks, self.max_batch = ranks, max_batch
        self.batch_costs = None if batch_costs is None else tuple(tuple(row[:max_batch]) for row in batch_costs)
        self.admission_window = admission_window
        self._solutions = {}

    def _solution(self, blocks):
        if blocks not in self._solutions:
            self._solutions[blocks] = self._solve(blocks)
        return self._solutions[blocks]

    def _solve(self, blocks):
        stages = DENOISING_STEPS * blocks
        ranks = blocks if self.ranks is None else self.ranks
        costs = self.batch_costs or ((1.0,) * self.max_batch,) * blocks
        work, parts = _group_costs(costs, self.max_batch)
        if ranks in (1, stages):
            owners = (0,) * stages if ranks == 1 else tuple(
                block * DENOISING_STEPS + step for step in range(DENOISING_STEPS) for block in range(blocks))
            return _plan(owners, ranks, work, parts)
        seeds = (("rectangular", _rectangle_seed(work, ranks)),
                 ("block_major_boundary", _boundary_seed(work, ranks)))
        candidates = [(name, _improve(owners, ranks, work, parts)) for name, owners in seeds]
        return min(candidates, key=lambda item: (item[1]["max_rank_compute_work"],
                   _edges(item[1]["stage_owners"]), item[0]))[1]

    def placement(self, blocks: int) -> dict:
        """Fixed homes and batch groups, with per-rank work in input cost units."""
        result = self._solution(blocks)
        return {key: result[key] for key in ("stage_owners", "rank_groups", "rank_compute_work",
                                             "max_rank_compute_work")}

    def _stage_owners(self, blocks: int) -> tuple[int, ...]:
        return self._solution(blocks)["stage_owners"]

    def schedule(self, blocks: int, chunks: int) -> BatchSchedule:
        result = self._solution(blocks)
        owners, groups = result["stage_owners"], result["rank_groups"]
        stages = DENOISING_STEPS * blocks
        offsets = [0]
        for stage in range(1, stages):
            boundary = owners[stage] != owners[stage - 1] or stage % blocks == 0
            offsets.append(offsets[-1] + int(boundary))
        positions = [{block: index for index, block in enumerate(sorted({block for block, _ in rows}))}
                     for rows in groups]
        width = max(map(len, positions))
        ticks = [[] for _ in range((chunks + offsets[-1]) * width)]
        for wave in range(chunks + offsets[-1]):
            for rank, rank_groups in enumerate(groups):
                for block, steps in rank_groups:
                    tick = wave * width + positions[rank][block]
                    # Reverse steps to order batch members by increasing chunk.
                    batch = tuple(Task(tick, rank, wave - offsets[step * blocks + block], step, block)
                                  for step in reversed(steps) if 0 <= wave - offsets[step * blocks + block] < chunks)
                    if batch:
                        ticks[tick].append(batch)
        while not ticks[-1]:
            ticks.pop()
        return BatchSchedule(ticks, blocks, chunks, admission_window=self.admission_window or stages)
