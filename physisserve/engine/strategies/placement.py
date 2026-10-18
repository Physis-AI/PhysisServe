"""Best-Fit placement with bounded discrepancy search over fixed stream layouts."""
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real

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


@dataclass(frozen=True)
class StreamCandidates:
    stream_id: str
    blocks: int
    chunks: int
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

    @property
    def num_ranks(self):
        return len(self.loads)

    def validate(self, ranks, blocks, streams):
        if (not isinstance(self.placements, (list, tuple)) or not isinstance(self.rejected, dict)
                or not isinstance(self.loads, (list, tuple))
                or any(not isinstance(reason, str) for reason in self.rejected.values())):
            raise ValueError("placement requires placements, string rejection reasons and RankDemand loads")
        if self.num_ranks != ranks:
            raise ValueError("placement must use the configured GPU pool")
        for load in self.loads:
            if (type(load) is not RankDemand or type(load.memory_bytes) is not int or load.memory_bytes < 0
                    or isinstance(load.utilization, bool) or not isinstance(load.utilization, Real)
                    or not isfinite(load.utilization) or load.utilization < 0):
                raise ValueError("loads must contain nonnegative integer memory and finite nonnegative utilization")
        if any(type(item) is not StreamPlacement or not isinstance(item.stream_id, str)
               or not item.stream_id for item in self.placements):
            raise ValueError("placements must contain StreamPlacement objects with nonempty stream IDs")
        placed = [item.stream_id for item in self.placements]
        if (len(set(placed)) != len(placed) or set(placed) & set(self.rejected)
                or set(placed) | set(self.rejected) != {str(index) for index in range(streams)}):
            raise ValueError("placement must cover every input index exactly once")
        for item in self.placements:
            _validate_candidates((item.candidate,), blocks)
            if not isinstance(item.rank_map, (list, tuple)) or len(item.rank_map) != len(item.candidate.demands):
                raise ValueError("rank_map must cover exactly the single-stream local ranks")
            if any(type(rank) is not int or not 0 <= rank < ranks for rank in item.rank_map):
                raise ValueError("mapped ranks must be integers inside the pool")


def _validate_candidates(candidates, blocks):
    # Shared by the two public boundaries; internal search only handles validated data.
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("candidates must be a list or tuple")
    names = set()
    for candidate in candidates:
        if type(candidate) is not Candidate or not isinstance(candidate.name, str) or not candidate.name:
            raise ValueError("candidates must contain Candidate objects with nonempty names")
        if candidate.name in names:
            raise ValueError("candidate names must be unique within a stream")
        names.add(candidate.name)
        if type(candidate.layout) not in (StageEqual, VirtualRank):
            raise ValueError("layout must be StageEqual or VirtualRank")
        if not isinstance(candidate.demands, (list, tuple)) or not candidate.demands:
            raise ValueError("demands must contain one RankDemand per local rank")
        for demand in candidate.demands:
            if (type(demand) is not RankDemand or type(demand.memory_bytes) is not int or demand.memory_bytes < 0
                    or isinstance(demand.utilization, bool) or not isinstance(demand.utilization, Real)
                    or not isfinite(demand.utilization) or demand.utilization < 0):
                raise ValueError("demands must contain nonnegative integer memory and finite nonnegative utilization")
        gap = candidate.max_gap_ms
        if gap is not None and (isinstance(gap, bool) or not isinstance(gap, Real) or not isfinite(gap) or gap <= 0):
            raise ValueError("max_gap_ms must be a finite positive number")
        if candidate.layout.num_ranks(blocks) != len(candidate.demands):
            raise ValueError("demands must cover exactly the layout's local ranks")


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
        if self.nodes == self.limit:
            self.exhausted = True
            return False
        self.nodes += 1
        return True


def _pooled_map(candidate, loads, budgets, discrepancies, search):
    """Best-Fit ordering with LDS over selected option positions.

    At each item, option 0 is the current best-fit choice. Option j consumes
    j discrepancies. Prior streams are immutable; all tentative current-stream
    work stays in this function and is discarded on failure.
    """
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
        options = [rank for rank, budget in enumerate(budgets) if _fits(work[rank], demand, budget)]
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


def plan_pool(problems: Sequence[StreamCandidates], budgets: Sequence[DeviceBudget],
              lds_discrepancies=2, search_limit=10000) -> PoolPlan:
    """Place arrivals without moving accepted streams. Demands are summed per GPU.

    Validate all input records before searching. Input max_gap_ms must be measured
    for each candidate; pooled execution must still satisfy the caller's delivery
    target. Search is bounded per arrival.
    """
    if type(lds_discrepancies) is not int or lds_discrepancies < 0:
        raise ValueError("lds_discrepancies must be a nonnegative integer")
    if type(search_limit) is not int or search_limit < 1:
        raise ValueError("search_limit must be a positive integer")
    budgets = tuple(budgets)
    if not budgets:
        raise ValueError("budgets must contain at least one DeviceBudget")
    for budget in budgets:
        if (type(budget) is not DeviceBudget or type(budget.memory_bytes) is not int or budget.memory_bytes <= 0
                or isinstance(budget.utilization, bool) or not isinstance(budget.utilization, Real)
                or not isfinite(budget.utilization) or budget.utilization <= 0):
            raise ValueError("budgets must contain positive integer memory and finite positive utilization")
    problems = tuple(problems)
    for problem in problems:
        if type(problem) is not StreamCandidates or not isinstance(problem.stream_id, str) or not problem.stream_id:
            raise ValueError("problems must contain StreamCandidates with nonempty stream IDs")
        if any(type(value) is not int or value < 1 for value in (problem.blocks, problem.chunks)):
            raise ValueError("blocks and chunks must be positive integers")
        if (isinstance(problem.slo_ms, bool) or not isinstance(problem.slo_ms, Real)
                or not isfinite(problem.slo_ms) or problem.slo_ms <= 0):
            raise ValueError("slo_ms must be a finite positive number")
        _validate_candidates(problem.candidates, problem.blocks)
    if len({problem.stream_id for problem in problems}) != len(problems):
        raise ValueError("stream IDs must be unique")
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
    return PoolPlan(tuple(placements), rejected, tuple(loads), {
        "global_slo_verified": False,
        "search_limit_per_stream": search_limit,
        "lds_discrepancies": lds_discrepancies,
        "search": searches,
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
