"""Best-Fit placement with bounded discrepancy search over fixed stream layouts."""
from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite
from numbers import Real
from typing import Any

from ..scheduler import Task
from .base import BatchSchedule


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _number(value, name, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value)
            or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be a finite {'positive' if positive else 'nonnegative'} number")


@dataclass(frozen=True)
class RankDemand:
    memory_bytes: int
    utilization: float

    def __post_init__(self):
        _integer(self.memory_bytes, "memory_bytes")
        _number(self.utilization, "utilization")


@dataclass(frozen=True)
class DeviceBudget:
    memory_bytes: int
    utilization: float = 1.0

    def __post_init__(self):
        _integer(self.memory_bytes, "memory_bytes", 1)
        _number(self.utilization, "utilization", positive=True)


@dataclass(frozen=True)
class Candidate:
    name: str
    layout: Any
    demands: tuple[RankDemand, ...]
    max_gap_ms: float | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("candidate name must be a nonempty string")
        if any(not callable(getattr(self.layout, name, None))
               for name in ("num_ranks", "rank_blocks", "schedule")):
            raise ValueError("layout must implement the single-stream layout interface")
        object.__setattr__(self, "demands", tuple(self.demands))
        if not self.demands or any(not isinstance(item, RankDemand) for item in self.demands):
            raise ValueError("demands must contain one RankDemand per local rank")
        if self.max_gap_ms is not None:
            _number(self.max_gap_ms, "max_gap_ms", positive=True)


@dataclass(frozen=True)
class StreamCandidates:
    stream_id: str
    blocks: int
    chunks: int
    slo_ms: float
    candidates: tuple[Candidate, ...]

    def __post_init__(self):
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise ValueError("stream_id must be a nonempty string")
        _integer(self.blocks, "blocks", 1)
        _integer(self.chunks, "chunks", 1)
        _number(self.slo_ms, "slo_ms", positive=True)
        object.__setattr__(self, "candidates", tuple(self.candidates))
        names = set()
        for candidate in self.candidates:
            if not isinstance(candidate, Candidate):
                raise ValueError("candidates must contain Candidate objects")
            if candidate.name in names:
                raise ValueError("candidate names must be unique within a stream")
            names.add(candidate.name)
            if candidate.layout.num_ranks(self.blocks) != len(candidate.demands):
                raise ValueError("demands must cover exactly the layout's local ranks")


@dataclass(frozen=True)
class StreamPlacement:
    stream_id: str
    candidate: Candidate
    rank_map: tuple[int, ...]

    def __post_init__(self):
        if not isinstance(self.stream_id, str) or not self.stream_id:
            raise ValueError("stream_id must be a nonempty string")
        if not isinstance(self.candidate, Candidate):
            raise ValueError("candidate must be a Candidate")
        object.__setattr__(self, "rank_map", tuple(self.rank_map))
        if len(self.rank_map) != len(self.candidate.demands):
            raise ValueError("rank_map must cover every candidate local rank")
        for rank in self.rank_map:
            _integer(rank, "physical rank")


@dataclass(frozen=True)
class PoolPlan:
    placements: tuple[StreamPlacement, ...]
    rejected: dict[str, str]
    loads: tuple[RankDemand, ...]
    diagnostics: dict = field(default_factory=dict)

    @property
    def num_ranks(self):
        return len(self.loads)


def _budgets(budgets):
    budgets = tuple(budgets)
    if not budgets or any(not isinstance(budget, DeviceBudget) for budget in budgets):
        raise ValueError("budgets must contain at least one DeviceBudget")
    return budgets


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

    Input max_gap_ms must be measured for each candidate; pooled execution must
    still satisfy the caller's delivery target. Search is bounded per arrival.
    """
    _integer(lds_discrepancies, "lds_discrepancies")
    _integer(search_limit, "search_limit", 1)
    budgets = _budgets(budgets)
    problems = tuple(problems)
    if any(not isinstance(problem, StreamCandidates) for problem in problems):
        raise ValueError("problems must contain StreamCandidates")
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
        _integer(total_ranks, "total_ranks", 1)
        self.rank_map = tuple(rank_map)
        if not self.rank_map:
            raise ValueError("rank_map must not be empty")
        for rank in self.rank_map:
            _integer(rank, "physical rank")
            if rank >= total_ranks:
                raise ValueError("mapped rank is outside the pool")
        if any(not callable(getattr(layout, name, None))
               for name in ("num_ranks", "rank_blocks", "schedule")):
            raise ValueError("layout must implement the single-stream layout interface")
        self.layout, self.total_ranks = layout, total_ranks
        self.max_batch = getattr(layout, "max_batch", 1)
        self.admission_window = getattr(layout, "admission_window", None)

    def _validate(self, blocks):
        if self.layout.num_ranks(blocks) != len(self.rank_map):
            raise ValueError("rank_map must cover exactly the single-stream local ranks")

    def num_ranks(self, blocks):
        self._validate(blocks)
        return self.total_ranks

    def rank_blocks(self, rank, blocks):
        self._validate(blocks)
        _integer(rank, "physical rank")
        if rank >= self.total_ranks:
            raise ValueError("rank is outside the pool")
        return tuple(sorted({block for local, physical in enumerate(self.rank_map) if physical == rank
                             for block in self.layout.rank_blocks(local, blocks)}))

    def schedule(self, blocks, chunks):
        self._validate(blocks)
        schedule = self.layout.schedule(blocks, chunks)
        batches = schedule.by_tick_batches
        remapped = [tuple(tuple(Task(task.tick, self.rank_map[task.rank], task.chunk, task.step, task.block)
                               for task in batch) for batch in tick) for tick in batches]
        return BatchSchedule(remapped, blocks, chunks,
                             admission_window=getattr(schedule, "admission_window", None))
