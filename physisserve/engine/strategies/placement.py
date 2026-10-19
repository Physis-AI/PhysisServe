"""Best-Fit placement with bounded discrepancy search over fixed stream layouts."""
from collections.abc import Sequence
from dataclasses import dataclass, field

from ..scheduler import Task
from .base import BatchSchedule
from .stage_equal import StageEqual
from .virtual_rank import VirtualRank


@dataclass(frozen=True)
class RankDemand:
    memory_bytes: int
    utilization: float


@dataclass(frozen=True)
class DeviceBudget:
    memory_bytes: int
    utilization: float = 1.0


@dataclass(frozen=True)
class Candidate:
    name: str
    layout: StageEqual | VirtualRank
    demands: tuple[RankDemand, ...]
    max_gap_ms: float | None = None
    mean_gap_ms: float | None = None


@dataclass(frozen=True)
class StreamCandidates:
    stream_id: str
    slo_ms: float
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class StreamPlacement:
    stream_id: str
    candidate: Candidate
    rank_map: tuple[int, ...]


@dataclass(frozen=True)
class PoolPlan:
    placements: tuple[StreamPlacement, ...]
    rejected: dict[str, str]
    loads: tuple[RankDemand, ...]
    diagnostics: dict = field(default_factory=dict)


def _fits(load, demand, budget):
    # No tolerance can make a physically over-budget allocation acceptable.
    return (load.memory_bytes + demand.memory_bytes <= budget.memory_bytes
            and load.utilization + demand.utilization <= budget.utilization)


def _add(load, demand):
    return RankDemand(load.memory_bytes + demand.memory_bytes, load.utilization + demand.utilization)


def _eligible(problem):
    return sorted((candidate for candidate in problem.candidates
                   if candidate.max_gap_ms is not None and candidate.max_gap_ms <= problem.slo_ms),
                  key=lambda candidate: (len(candidate.demands), candidate.name))


class _Search:
    def __init__(self, limit):
        self.limit, self.nodes, self.exhausted = limit, 0, False

    def visit(self):
        if self.nodes >= self.limit:
            self.exhausted = True
            return False
        self.nodes += 1
        return True


def _pooled_map(candidate, loads, budgets, discrepancies, search, distinct=False):
    """Best-Fit ordering with LDS over selected option positions.

    At each item, option 0 is the current best-fit choice. Option j consumes
    j discrepancies. Prior streams are immutable; all tentative current-stream
    work stays in this function and is discarded on failure.
    """
    if not budgets or (distinct and len(candidate.demands) > len(budgets)):
        return None
    work = list(loads)
    rank_map = [-1] * len(candidate.demands)
    largest_memory = max(budget.memory_bytes for budget in budgets)
    largest_utilization = max(budget.utilization for budget in budgets)
    order = sorted(range(len(candidate.demands)), key=lambda index: (
        -max(candidate.demands[index].memory_bytes / largest_memory,
             candidate.demands[index].utilization / largest_utilization), index))

    def place(position, remaining):
        if position == len(order):
            return tuple(rank_map)
        local_rank = order[position]
        demand = candidate.demands[local_rank]
        options = [rank for rank, budget in enumerate(budgets)
                   if (not distinct or rank not in rank_map) and _fits(work[rank], demand, budget)]
        options.sort(key=lambda rank: (
            (budgets[rank].memory_bytes - work[rank].memory_bytes - demand.memory_bytes)
            / budgets[rank].memory_bytes
            + (budgets[rank].utilization - work[rank].utilization - demand.utilization)
            / budgets[rank].utilization, rank))
        for difference, rank in enumerate(options):
            if difference > remaining:
                break
            if not search.visit():
                return None
            previous = work[rank]
            work[rank] = _add(previous, demand)
            rank_map[local_rank] = rank
            result = place(position + 1, remaining - difference)
            if result is not None:
                return result
            work[rank] = previous
            rank_map[local_rank] = -1
            if search.exhausted:
                return None
        return None

    return place(0, discrepancies)


def _use_spare_gpus(problems, placements, budgets, discrepancies, search):
    """Improve a complete plan before loading weights, without displacing peers.

    Isolate measured layouts first, then greedily buy measured throughput per
    additional GPU. Standalone timings only rank choices: pooled SLO and speed
    still need measurement, particularly after changing physical topology.
    """
    choices = {problem.stream_id: _eligible(problem) for problem in problems}
    while not search.exhausted:
        best, best_score = None, None
        for index, item in enumerate(placements):
            current = item.candidate
            if current.mean_gap_ms is None:
                continue
            occupied = {rank for peer in placements if peer is not item for rank in peer.rank_map}
            free = [rank for rank in range(len(budgets)) if rank not in occupied]
            own = set(item.rank_map)
            isolated = len(own) == len(current.demands) and own.isdisjoint(occupied)
            for candidate in choices[item.stream_id]:
                spread = candidate is current and not isolated
                if candidate.mean_gap_ms is None or (candidate.mean_gap_ms >= current.mean_gap_ms and not spread):
                    continue
                extra = len(candidate.demands) - len(own - occupied)
                gain = 1 / candidate.mean_gap_ms - 1 / current.mean_gap_ms
                score = (spread, extra <= 0, gain / max(1, extra), gain)
                if best_score is not None and score <= best_score:
                    continue
                mapping = _pooled_map(candidate, [RankDemand(0, 0.0)] * len(free),
                                      [budgets[rank] for rank in free], discrepancies, search, distinct=True)
                if mapping is not None:
                    best = index, StreamPlacement(item.stream_id, candidate, tuple(free[rank] for rank in mapping))
                    best_score = score
                if search.exhausted:
                    break
            if search.exhausted:
                break
        if best is None:
            break
        index, replacement = best
        placements[index] = replacement


def plan_pool(problems: Sequence[StreamCandidates], budgets: Sequence[DeviceBudget],
              lds_discrepancies=2, search_limit=10000) -> PoolPlan:
    """Admit streams compactly, then spend spare GPUs on measured faster layouts.

    Admission is unchanged by optimization; no accepted stream is evicted.
    Everything happens before inference, not through live KV migration. Search
    is bounded per arrival and for the final whole-plan improvement pass.
    """
    budgets = tuple(budgets)
    loads = [RankDemand(0, 0.0) for _ in budgets]
    placements, rejected, searches = [], {}, {}
    for problem in problems:
        candidates = _eligible(problem)
        search = _Search(search_limit)
        selected, rank_map = None, None
        tried = []
        for candidate in candidates:
            tried.append(candidate.name)
            rank_map = _pooled_map(candidate, loads, budgets, lds_discrepancies, search)
            if rank_map is not None:
                selected = candidate
                break
            if search.exhausted:
                break
        searches[problem.stream_id] = dict(nodes=search.nodes, exhausted=search.exhausted,
                                          tried_candidates=tuple(tried))
        if selected is None:
            rejected[problem.stream_id] = (
                "no individually verified candidate satisfies the requested SLO" if not candidates
                else "search budget exhausted; feasibility unknown" if search.exhausted
                else "no placement found by this bounded policy; infeasibility not proved")
            continue
        placements.append(StreamPlacement(problem.stream_id, selected, rank_map))
        for demand, rank in zip(selected.demands, rank_map):
            loads[rank] = _add(loads[rank], demand)
    improvement = _Search(search_limit)
    if not rejected:
        _use_spare_gpus(problems, placements, budgets, lds_discrepancies, improvement)
        loads = [RankDemand(0, 0.0) for _ in budgets]
        for item in placements:
            for demand, rank in zip(item.candidate.demands, item.rank_map):
                loads[rank] = _add(loads[rank], demand)
    return PoolPlan(tuple(placements), rejected, tuple(loads), {
        "global_slo_verified": False,
        "search_limit_per_stream": search_limit,
        "lds_discrepancies": lds_discrepancies,
        "search": searches,
        "spare_gpu_search": dict(nodes=improvement.nodes, exhausted=improvement.exhausted),
    })


class MappedLayout:
    """Map existing local homes to pool GPUs without changing batch groups.

    Multiple local ranks may share a physical GPU. Formerly parallel groups on
    that GPU are now sequential in their original tick order; no new batch or
    latency guarantee is invented. Idle pool ranks return an empty block set.
    """
    def __init__(self, layout, rank_map, total_ranks):
        self.rank_map = tuple(rank_map)
        self.layout, self.total_ranks = layout, total_ranks
        self.max_batch = getattr(layout, "max_batch", 1)
        self.admission_window = getattr(layout, "admission_window", None)

    def num_ranks(self, blocks):
        return self.total_ranks

    def rank_blocks(self, rank, blocks):
        return tuple(sorted({block for local, physical in enumerate(self.rank_map) if physical == rank
                             for block in self.layout.rank_blocks(local, blocks)}))

    def schedule(self, blocks, chunks):
        schedule = self.layout.schedule(blocks, chunks)
        batches = schedule.by_tick_batches
        remapped = [tuple(tuple(Task(task.tick, self.rank_map[task.rank], task.chunk, task.step, task.block)
                               for task in batch) for batch in tick) for tick in batches]
        return BatchSchedule(remapped, blocks, chunks,
                             admission_window=getattr(schedule, "admission_window", None))
