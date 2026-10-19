"""Profile and compare StageEqual/VirtualRank capacity; --streams selects a fixed-count comparison."""
import argparse
from bisect import bisect_left, bisect_right
import gc
import hashlib
import json
import os
import socket
from dataclasses import replace
from importlib.metadata import version
from math import ceil, isfinite
from pathlib import Path
from statistics import median
from tempfile import mkdtemp

import torch
import torch.distributed as dist

from physisserve import PhysisServe, SamplingParams
from physisserve.engine.sampler import DENOISING_STEPS
from physisserve.engine.strategies import StageEqual, VirtualRank
from physisserve.engine.strategies.placement import Candidate, DeviceBudget, RankDemand, StreamCandidates, plan_pool


def release(engine):
    engine.sessions, engine.runners = (), {}
    gc.collect()
    torch.cuda.empty_cache()
    engine._barrier()


def devices(engine):
    release(engine)
    free, total = torch.cuda.mem_get_info(engine.device)
    return {"rank": engine.rank, "name": torch.cuda.get_device_name(engine.device),
            "host": socket.gethostname(), "device": str(engine.device),
            "uuid": str(getattr(torch.cuda.get_device_properties(engine.device), "uuid", "unavailable")),
            "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "free_bytes": free, "total_bytes": total}


def _conditions(engine, prompts):
    cache = getattr(engine, "_bench_conditions", {})
    missing = set(prompts) - cache.keys()
    if missing and engine.is_output:
        encoder = engine.config.make_backend().encoder
        encoder.load("cpu", engine.codec_dtype)
        with torch.inference_mode():
            cache.update({prompt: {key: value.cpu().contiguous() for key, value in encoder.encode(prompt).items()}
                          for prompt in sorted(missing)})
        del encoder
        gc.collect()
    if missing and engine.distributed:
        payload = [{prompt: cache[prompt] for prompt in missing} if engine.is_output else None]
        dist.broadcast_object_list(payload, src=0, group=engine.control_group)
        cache.update(payload[0])
    engine._bench_conditions = cache
    return [cache[prompt] for prompt in prompts]


def calibrate(engine, params, prompt, max_batch, warmups=3, samples=12):
    """Native B1..Bmax costs: real blocks, prompt and seeded full-history fixtures.

    Each worker loads one block at a time. Missing/OOM cells remain missing;
    isolated-block costs are planning inputs, not end-to-end measurements.
    """
    from physisserve.engine.kv_cache import HISTORY_CHUNKS

    release(engine)
    config, condition = engine.config, _conditions(engine, [prompt])[0]
    shape = (1, config.channels, 1, params.height // config.spatial_scale, params.width // config.spatial_scale)
    times = [torch.tensor(value, dtype=torch.float32) for value in config.timesteps]
    chunks = HISTORY_CHUNKS + 2 + (warmups + samples + 1) * DENOISING_STEPS
    row = {"rank": engine.rank, "samples_ms": {}, "error": None}
    for block in range(engine.rank, config.num_blocks, config.ranks):
        dit = None
        try:
            dit = config.make_backend().dit
            dit.load([block], engine.device, engine.dtype)
            with torch.inference_mode():
                dit.prepare(condition, shape, chunks, fps=config.fps)
                dit.prepare_timesteps(times)
                row["samples_ms"].update(_block_costs(dit, block, shape, times, params.seed,
                                                       max_batch, warmups, samples))
        except torch.cuda.OutOfMemoryError as exc:
            row["error"] = f"block {block}: {exc}"
        finally:
            del dit
            gc.collect()
            torch.cuda.empty_cache()
        if row["error"]:
            break
    return row


def _block_costs(dit, block, shape, times, seed, max_batch, warmups, samples):
    from physisserve.engine.kv_cache import Cache, HISTORY_CHUNKS

    rng = torch.Generator(device=dit.device).manual_seed(seed + block * 104729)

    def random(size, dtype):
        return torch.randn(size, generator=rng, device=dit.device, dtype=dtype)

    hidden = [random(dit.hidden_shape(shape), getattr(dit, "hidden_dtype", dit.dtype))
              for _ in range(max_batch)]
    dtype = getattr(dit, "page_dtype", dit.dtype)
    first_shape = (*shape[:2], dit.first_frames, *shape[3:])
    sink = tuple(random(dit.kv_shape(first_shape), dtype) for _ in range(2))
    histories = [[sink] + [tuple(random(dit.kv_shape(shape), dtype) for _ in range(2))
                          for _ in range(HISTORY_CHUNKS)] for _ in range(max_batch)]
    costs = {}
    for size in range(1, max_batch + 1):
        def call(ordinal):
            chunks = [HISTORY_CHUNKS + 2 + ordinal * DENOISING_STEPS + member for member in range(size)]
            selected = [times[(ordinal + member) % DENOISING_STEPS] for member in range(size)]
            caches = [Cache(histories[member]) for member in range(size)]
            outputs = ([dit.block(block, hidden[0], selected[0], chunks[0], caches[0])] if size == 1 else
                       dit.block_batch(block, hidden[:size], selected, chunks, caches))
            return outputs, caches

        for ordinal in range(warmups):
            call(ordinal)
        torch.cuda.synchronize(dit.device)
        events = []
        for ordinal in range(warmups, warmups + samples):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            outputs, caches = call(ordinal)
            end.record()
            events.append((start, end))
        events[-1][1].synchronize()
        if any(cache.page is None for cache in caches) or not all(
                bool(torch.isfinite(value).all()) for value in
                [*outputs, *(value for cache in caches for value in cache.page)]):
            raise RuntimeError(f"invalid calibration output: block {block}, batch {size}")
        costs[f"{block}:{size}"] = [start.elapsed_time(end) for start, end in events]
        del outputs, caches
    return costs


def prepare_candidate(engine, prompts, params, candidates=None, budgets=(), memory_margin=1.2, memory_budgets=None):
    """Load locally, then conservatively screen KV/state memory before any P2P.

    Estimated skips are unverified, not measured memory failures. The estimate
    cannot guarantee runtime fit: communication buffers, native workspaces and
    allocator fragmentation still vary. Runtime OOM aborts the distributed run.
    """
    from math import ceil, prod

    from physisserve.engine.kv_cache import HISTORY_CHUNKS
    from physisserve.engine.model_runner import ModelRunner
    from physisserve.engine.strategies.placement import MappedLayout, plan_pool
    from physisserve.engine.transport import Distributed, Local

    release(engine)
    conditions = _conditions(engine, prompts)
    free, _ = torch.cuda.mem_get_info(engine.device)
    memory_budgets = budgets if memory_budgets is None else memory_budgets
    budget = min(free, memory_budgets[engine.rank].memory_bytes) if memory_budgets else int(free * 0.8)
    before = torch.cuda.memory_allocated(engine.device)
    torch.cuda.reset_peak_memory_stats(engine.device)
    if candidates is None:
        layouts = dict.fromkeys(range(len(prompts)), engine.config.make_layout())
    else:
        plan = plan_pool(candidates, budgets)
        layouts = {int(item.stream_id): MappedLayout(item.candidate.layout, item.rank_map,
                   engine.config.ranks) for item in plan.placements}
    error, status, estimated = None, "ready", 0
    try:
        for index, layout in layouts.items():
            ownership = tuple(tuple(layout.rank_blocks(rank, engine.blocks)) for rank in range(engine.config.ranks))
            backend = engine.config.make_backend()
            transport = Distributed(engine.device) if engine.distributed else Local()
            runner = ModelRunner(backend.dit, engine.sampler, layout, device=engine.device,
                                 dtype=engine.dtype, transport=transport)
            engine.runners[index] = ownership, runner
            param, dit = params[index], runner.dit
            shape = (1, engine.config.channels, 1, param.height // engine.config.spatial_scale,
                     param.width // engine.config.spatial_scale)
            with torch.inference_mode():
                dit.prepare(conditions[index], shape, param.num_chunks, fps=engine.config.fps)
                dit.prepare_timesteps([torch.tensor(value, dtype=torch.float32)
                                       for value in engine.config.timesteps])
            first_shape = (*shape[:2], dit.first_frames, *shape[3:])
            stages = sum(task.rank == engine.rank for task in layout.schedule(engine.blocks, 1).tasks)
            page_bytes = 2 * prod(dit.kv_shape(shape)) * runner.page_dtype.itemsize
            sink_bytes = 2 * prod(dit.kv_shape(first_shape)) * runner.page_dtype.itemsize
            state_bytes = prod(shape) * engine.dtype.itemsize + prod(dit.hidden_shape(shape)) * runner.hidden_dtype.itemsize
            estimated += stages * ((HISTORY_CHUNKS + 1) * page_bytes + 2 * sink_bytes + state_bytes)
            del backend, runner
        estimated = ceil((torch.cuda.memory_allocated(engine.device) - before + estimated) * memory_margin)
        if estimated > budget:
            status, error = "estimated_skip", f"estimated {estimated} bytes exceeds budget {budget} bytes"
    except torch.cuda.OutOfMemoryError as exc:
        status, error = "oom", str(exc)
    outcomes = [(status, error)]
    if engine.distributed:
        outcomes = [None] * engine.config.ranks
        dist.all_gather_object(outcomes, (status, error), group=engine.control_group)
    failed = [value for value, _ in outcomes if value != "ready"]
    row = {"rank": engine.rank, "status": "oom" if "oom" in failed else "estimated_skip" if failed else "ready",
           "errors": [error for _, error in outcomes], "estimated_bytes": estimated,
           "budget_bytes": budget, "free_bytes": free}
    if failed:
        release(engine)
    return row


def _instrument(runner, first, stop):
    originals, events, depth = {}, [], 0
    for name in ("embed", "block", "block_batch", "unembed"):
        original = getattr(runner.dit, name)
        originals[name] = original

        def measured(*args, _original=original, _name=name, **kwargs):
            nonlocal depth
            chunks = args[3] if _name == "block_batch" else [args[3] if _name == "block" else args[2]]
            timed = depth == 0 and first <= min(chunks) and max(chunks) < stop
            depth += 1
            if timed:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            try:
                return _original(*args, **kwargs)
            finally:
                depth -= 1
                if timed:
                    end.record()
                    events.append((start, end))

        setattr(runner.dit, name, measured)
    return originals, events


def profile_candidate(engine, prompts, params, candidates, budgets, warmups, repeats, trim):
    """Uninstrumented long-run traces plus separately measured native compute.

    Peak allocated bytes include the loaded stage, live KV and workspaces.
    No decoding or retained output tensors are included. Margins are applied
    by the planner, not hidden in these raw measurements.
    """
    trials = measure(engine, prompts, params, warmups, repeats, candidates, budgets)
    first, stop = trim, params[0].num_chunks - trim
    wrapped = [(runner, *_instrument(runner, first, stop)) for _, runner, _ in engine.sessions]
    try:
        engine.run(return_outputs=False)
        torch.cuda.synchronize(engine.device)
        compute = sum(start.elapsed_time(end) for _, _, events in wrapped for start, end in events)
        return {"rank": engine.rank, "trials": trials,
                "peak_bytes": torch.cuda.max_memory_allocated(engine.device),
                "compute_ms_per_chunk": compute / (stop - first)}
    finally:
        for runner, originals, _ in wrapped:
            for name, original in originals.items():
                setattr(runner.dit, name, original)


def measure(engine, prompts, params, warmups, repeats, candidates=None, budgets=()):
    """Worker-side DiT timing. Prompt encoding, loading, CPU copies and VAE are excluded."""
    engine.prepare(prompts, params, candidates, budgets)
    for _ in range(warmups):
        engine.run(return_outputs=False)
    measurements = []
    for _ in range(repeats):
        engine._barrier()
        torch.cuda.synchronize(engine.device)
        start = torch.cuda.Event(enable_timing=True)
        events, largest_batch = {}, 0

        def completed(index, chunk, latent):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            events.setdefault(index, []).append((chunk, event))

        def batched(index, rank, size):
            nonlocal largest_batch
            largest_batch = max(largest_batch, size)

        start.record()
        engine.run(on_output=completed, on_batch=batched, return_outputs=False)
        torch.cuda.synchronize(engine.device)
        measurements.append({
            "times": {index: [(chunk, start.elapsed_time(event)) for chunk, event in values]
                      for index, values in events.items()},
            "max_batch": largest_batch,
        })
    return measurements


def summarize(rank_results, streams, chunks, repeats, trim):
    """Measure the common middle, excluding fill, drain and early stream completion.

    Delivery events are recorded on rank zero, after output transfers, so
    even mapped multi-node streams use one CUDA clock.
    """
    counts = [chunks] * streams if isinstance(chunks, int) else chunks
    durations, windows, rates = [], [], []
    gaps, max_batch, max_gap, half_drift = [[] for _ in range(streams)], 0, 0.0, 0.0
    for repeat in range(repeats):
        owners, = (rank[repeat]["times"] for rank in rank_results if rank[repeat]["times"])
        max_batch = max(max_batch, *(rank[repeat]["max_batch"] for rank in rank_results))
        if set(owners) != set(range(streams)) or any(
                [chunk for chunk, _ in sorted(owners[index])] != list(range(count))
                for index, count in enumerate(counts)):
            raise ValueError("incomplete stream delivery trace")
        times = [[time for _, time in sorted(owners[index])] for index in range(streams)]
        start = max(values[trim] for values in times)
        end = min(values[count - 1 - trim] for count, values in zip(counts, times))
        rate = 0.0
        for index, values in enumerate(times):
            first = max(trim, bisect_left(values, start))
            last = min(counts[index] - 1 - trim, bisect_right(values, end) - 1)
            if last - first < max(trim, 2) or values[last] <= values[first]:
                raise ValueError("no sufficiently long common middle window; increase --chunks")
            # Keep zero gaps inside native batches; average elapsed time over
            # all retained gaps instead of taking the median of individual gaps.
            gap = (values[last] - values[first]) / (last - first)
            gaps[index].append(gap)
            rate += 1000 / gap
            max_gap = max(max_gap, *(b - a for a, b in zip(values[first:last], values[first + 1:last + 1])))
            middle = (first + last) // 2
            left = (values[middle] - values[first]) / (middle - first)
            right = (values[last] - values[middle]) / (last - middle)
            half_drift = max(half_drift, abs(left - right) / max(left, right))
        durations.append(max(values[-1] for values in times))
        windows.append(end - start)
        rates.append(rate)
    completion_ms = median(durations)
    per_stream = [median(values) for values in gaps]
    return {"completion_ms": completion_ms, "chunk_gap_ms": median(per_stream),
            "stream_gap_ms": per_stream, "steady_chunks_per_second": median(rates),
            "full_chunks_per_second": sum(counts) * 1000 / completion_ms,
            "window_ms": median(windows), "max_gap_ms": max_gap, "half_drift": half_drift,
            "max_batch": max_batch}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def archive(path):
    if path.exists():
        count = len(list(path.parent.glob(f"{path.stem}-attempt-*.json")))
        path.replace(path.with_name(f"{path.stem}-attempt-{count}.json"))


def cost_table(rows, blocks, max_batch):
    cells = [key for row in rows for key in row["samples_ms"]]
    samples = {key: values for row in rows for key, values in row["samples_ms"].items()}
    expected = {f"{block}:{size}" for block in range(blocks) for size in range(1, max_batch + 1)}
    if (any(row["error"] for row in rows) or len(cells) != len(set(cells)) or set(samples) != expected
            or any(not values or any(not isfinite(x) or x <= 0 for x in values) for values in samples.values())):
        raise ValueError("incomplete block calibration; inspect calibration.json (no estimated cells are substituted)")
    return [[median(samples[f"{block}:{size}"]) for size in range(1, max_batch + 1)] for block in range(blocks)]


def make_layout(method, ranks, costs, max_batch):
    return StageEqual(ranks) if method == "stage_equal" else VirtualRank(ranks, max_batch, costs)


def problems_for(candidates, count, slo_ms):
    return tuple(StreamCandidates(str(index), slo_ms, tuple(candidates)) for index in range(count))


def configure(engine, method, costs, max_batch):
    release(engine)
    engine.config.strategy = method
    engine.config.batch_costs = costs if method == "virtual_rank" else None
    engine.config.max_batch = max_batch if method == "virtual_rank" else 1


def placement_json(plan):
    return {"placements": [{"stream": item.stream_id, "candidate": item.candidate.name,
                            "rank_map": item.rank_map} for item in plan.placements],
            "rejected": plan.rejected, "diagnostics": plan.diagnostics}


def qualify(rows, counts, repeats, trim, slo_ms):
    """Strict SLO includes bootstrap; steady numbers never replace that verdict."""
    gaps, startup, first = [], [], []
    for repeat in range(repeats):
        owners, = (rank[repeat]["times"] for rank in rows if rank[repeat]["times"])
        if set(owners) != set(range(len(counts))):
            raise ValueError("incomplete stream delivery trace")
        for index, count in enumerate(counts):
            values = sorted(owners[index])
            times = [time for _, time in values]
            if ([chunk for chunk, _ in values] != list(range(count)) or
                    any(not isfinite(t) or t < 0 for t in times) or times != sorted(times)):
                raise ValueError("invalid stream delivery trace")
            gaps.extend(b - a for a, b in zip(times, times[1:]))
            startup.append(times[1] - times[0])
            first.append(times[0])
    result = {"all_max_gap_ms": max(gaps), "startup_max_gap_ms": max(startup),
              "first_output_max_ms": max(first),
              "strict_slo_pass": slo_ms is not None and max(gaps) <= slo_ms}
    try:
        result.update(summarize(rows, len(counts), counts, repeats, trim))
    except ValueError as exc:
        if "common middle window" not in str(exc):
            raise
        return {**result, "status": "insufficient_overlap", "qualified": False}
    result["stable"] = result["half_drift"] <= 0.05
    result["steady_slo_pass"] = slo_ms is not None and result["max_gap_ms"] <= slo_ms and result["stable"]
    result["qualified"] = result["strict_slo_pass"] and result["steady_slo_pass"]
    result["status"] = "measured" if result["stable"] else "unstable"
    return result


def longer_counts(rows, counts, trim):
    """Extend fast streams as well as the window; do not average their drained tails."""
    starts, means = [], []
    for index, count in enumerate(counts):
        traces = [[time for _, time in sorted(trial["times"][index])]
                  for rank in rows for trial in rank if index in trial["times"]]
        starts.append(max(values[trim] for values in traces))
        means.append(median((values[count - 1 - trim] - values[trim]) / (count - 1 - 2 * trim)
                            for values in traces))
    end = max(starts) + 4 * trim * max(means)
    return [max(2 * count - 1, ceil((end - start) / gap) + 2 * trim + 1)
            for count, start, gap in zip(counts, starts, means)]


def run_point(service, params, problems, budgets, args, path, trim, physical_budgets=None):
    from bench import measure, prepare_candidate

    plan = plan_pool(problems, budgets) if problems is not None else None
    mapping = placement_json(plan) if plan is not None else "same full-pool layout per stream"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["status"] in ("measured", "not_admitted") and saved["plan"] == json.loads(json.dumps(mapping)):
            return saved
        archive(path)
    count = len(problems) if problems is not None else args.streams
    point = {"status": "running", "streams": count,
             "plan": mapping, "attempts": []}
    save(path, point)
    if plan is not None and plan.rejected:
        point["status"] = "not_admitted"
    else:
        prompts = [args.prompt] * count
        parameters = [replace(params, seed=params.seed + index) for index in range(count)]
        point["preflight"] = service._call(prepare_candidate, prompts, parameters, problems, budgets, 1.2, physical_budgets)
        if any(row["status"] != "ready" for row in point["preflight"]):
            point["status"] = "unverified_memory"
        else:
            for attempt in range(3):
                counts = [param.num_chunks for param in parameters]
                rows = service._call(measure, prompts, parameters, args.warmups, args.repeats, problems, budgets)
                result = qualify(rows, counts, args.repeats, trim, args.slo_ms)
                point["attempts"].append({"chunks": counts, "raw": rows, "result": result})
                point.update(result)
                point["strict_slo_pass"] = all(item["result"]["strict_slo_pass"] for item in point["attempts"])
                point["qualified"] = point["strict_slo_pass"] and result.get("steady_slo_pass", False)
                for key in ("all_max_gap_ms", "startup_max_gap_ms", "first_output_max_ms"):
                    point[key] = max(item["result"][key] for item in point["attempts"])
                save(path, point)
                if result["status"] == "measured":
                    break
                counts = longer_counts(rows, counts, trim)
                parameters = [replace(param, num_chunks=count) for param, count in zip(parameters, counts)]
    save(path, point)
    return point


def candidate_profiles(service, params, budgets, costs, args, directory, trim):
    from bench import prepare_candidate, profile_candidate

    profiles = []
    # Large models may not fit on few ranks. Keep measured larger layouts first.
    for ranks in range(service.config.ranks, 0, -1):
        for method in ("stage_equal", "virtual_rank"):
            name = f"{method}-r{ranks}"
            path = directory / f"profile-{name}.json"
            if path.exists() and (row := json.loads(path.read_text()))["status"] == "measured":
                profiles.append(row)
                continue
            archive(path)
            layout = make_layout(method, ranks, costs, args.max_batch_size)
            candidate = Candidate(name, layout, (RankDemand(0, 1.0),) * ranks, 0.0)
            problems = problems_for([candidate], 1, args.slo_ms)
            plan = plan_pool(problems, budgets)
            row = {"name": name, "method": method, "ranks": ranks, "status": "running"}
            save(path, row)
            print(f"Profiling {name}", flush=True)
            ready = service._call(prepare_candidate, [args.prompt], [params], problems, budgets)
            row["preflight"] = ready
            if any(item["status"] != "ready" for item in ready):
                row["status"] = "unverified_memory"
            else:
                raw = service._call(profile_candidate, [args.prompt], [params], problems, budgets,
                                    args.warmups, args.repeats, trim)
                summary = qualify([item["trials"] for item in raw], [params.num_chunks], args.repeats, trim, args.slo_ms)
                mapping = plan.placements[0].rank_map
                by_rank = {item["rank"]: item for item in raw}
                row.update(status=summary["status"], summary=summary, raw=raw,
                           idle_peak_bytes=max((item["peak_bytes"] for item in raw if item["rank"] not in mapping), default=0),
                           demands=[{key: by_rank[rank][key] for key in ("peak_bytes", "compute_ms_per_chunk")}
                                    for rank in mapping])
            save(path, row)
            profiles.append(row)
    return profiles


def measured_candidates(profiles, method, costs, args, idle_bytes=0):
    return [Candidate(profile["name"], make_layout(method, profile["ranks"], costs, args.max_batch_size),
                      tuple(RankDemand(max(1, ceil((row["peak_bytes"] - idle_bytes) * 1.2)),
                                       row["compute_ms_per_chunk"] * 1.2 / args.slo_ms)
                            for row in profile["demands"]), profile["summary"]["max_gap_ms"])
            for profile in profiles if profile["method"] == method and profile["status"] == "measured"]


def equal_problems(profiles, ranks, count, slo_ms, idle_bytes=0):
    """Equal GPU counts per stream, with no sharing between streams.

    Utilization 1 reserves an exclusive GPU; it is not a compute estimate.
    Memory comes from the same profiles/margins as pooling. Do not screen by
    isolated latency: the concurrent run must determine whether the SLO holds.
    None means a required profile is unverified, not an observed failure.
    """
    if count > ranks:
        return problems_for([], count, slo_ms)
    quotient, remainder = divmod(ranks, count)
    sizes = [quotient + (index < remainder) for index in range(count)]
    measured = {profile["ranks"]: profile for profile in profiles
                if profile["method"] == "stage_equal" and profile["status"] == "measured"}
    if not set(sizes) <= measured.keys():
        return None
    problems = []
    for index, size in enumerate(sizes):
        demands = tuple(RankDemand(max(1, ceil((row["peak_bytes"] - idle_bytes) * 1.2)), 1.0)
                        for row in measured[size]["demands"])
        candidate = Candidate(f"stage_equal-exclusive-r{size}", StageEqual(size), demands, 0.0)
        problems.append(StreamCandidates(str(index), slo_ms, (candidate,)))
    return tuple(problems)


def search_capacity(service, params, budgets, profiles, costs, args, directory, trim):
    # Even an unmapped rank loads the stream's small shared modules. Reserve
    # that measured footprint on every GPU, then charge active stages extra.
    idle_bytes = max((profile.get("idle_peak_bytes", 0) for profile in profiles), default=0)
    methods = ("stage_equal_exclusive", "stage_equal", "virtual_rank")
    candidates = {method: measured_candidates(profiles, method, costs, args, idle_bytes)
                  for method in methods[1:]}
    result = {method: {"largest_verified_streams": 0, "largest_steady_only_streams": 0,
                       "stop_reason": "stream_limit", "points": [], "ceiling_reached": False,
                       "idle_bytes_per_stream_per_gpu": idle_bytes,
                       "unverified_profiles": [p["name"] for p in profiles
                           if p["method"] == ("stage_equal" if method == "stage_equal_exclusive" else method)
                           and p["status"] != "measured"], "unverified_points": []}
              for method in methods}
    for count in range(1, args.max_streams + 1):
        # Interleave methods at each count; reverse by count and seed so one
        # method is not always measured after the others have warmed the GPUs.
        order = methods[::-1] if (count + args.seed) % 2 else methods
        available = tuple(DeviceBudget(max(1, item.memory_bytes - ceil(idle_bytes * 1.2) * count), item.utilization)
                          for item in budgets)
        for method in order:
            record = result[method]
            if record["stop_reason"] != "stream_limit":
                continue
            print(f"Testing {method}, {count} stream(s)", flush=True)
            path = directory / f"{method}-streams{count}.json"
            exclusive = method == "stage_equal_exclusive"
            problems = (equal_problems(profiles, len(budgets), count, args.slo_ms, idle_bytes) if exclusive
                        else problems_for(candidates[method], count, args.slo_ms))
            if problems is None:
                point = {"streams": count, "status": "unverified_profile",
                         "reason": "the equal GPU split requires an unverified StageEqual profile"}
            else:
                point = run_point(service, params, problems, available, args, path, trim, physical_budgets=budgets)
            point["evaluation_order"] = list(order)
            if exclusive and count > len(budgets):
                point["reason"] = "exclusive equal splitting requires at least one GPU per stream"
                record["stop_reason"] = "not_admitted"
            elif not exclusive and not candidates[method] and point["status"] == "not_admitted":
                record["stop_reason"] = "not_admitted"
            # Other failures are not monotonic: changing the stream count can
            # change both the available memory and the selected layouts.
            save(path, point)
            record["points"].append({key: value for key, value in point.items() if key not in ("attempts", "preflight")})
            record["largest_verified_streams"] = max((p["streams"] for p in record["points"] if p.get("qualified")), default=0)
            record["largest_steady_only_streams"] = max((p["streams"] for p in record["points"] if p.get("steady_slo_pass")), default=0)
            record["ceiling_reached"] = count == args.max_streams and record["stop_reason"] == "stream_limit"
            record["unverified_points"] = [p["streams"] for p in record["points"] if p["status"] in
                                           ("unverified_profile", "unverified_memory", "unstable", "insufficient_overlap")]
            save(directory / "capacity.json", result)
    for method, record in result.items():
        print(f"{method}: largest verified = {record['largest_verified_streams']}, stop = {record['stop_reason']}", flush=True)
    return result


def identity(config, params, hardware):
    root = Path(__file__).resolve().parent
    return {"model": config.model, "height": params.height, "width": params.width,
            "devices": [{key: item.get(key) for key in ("name", "host", "device", "uuid", "visible_devices", "total_bytes")}
                        for item in hardware], "torch": torch.__version__,
            "nccl_environment": {key: os.environ.get(key) for key in
                                 ("NCCL_ALGO", "NCCL_PROTO", "NCCL_IB_DISABLE", "NCCL_P2P_DISABLE", "NCCL_SOCKET_IFNAME")},
            "cuda": torch.version.cuda, "diffusers": version("diffusers"), "transformers": version("transformers"),
            "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [root / "bench.py", *sorted((root / "physisserve").rglob("*.py"))]},
            "model_config_sha256": {name: hashlib.sha256(Path(config.model, name).read_bytes()).hexdigest()
                                    for name in ("model_index.json", "transformer/config.json")}}


def main():
    from bench import calibrate, devices

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Diffusers directory or Hugging Face repository")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--streams", type=int, help="run a fixed-count comparison instead of capacity search")
    parser.add_argument("--slo-ms", type=float, default=1600, help="delivery-gap deadline in ms, including bootstrap (default: 1600)")
    parser.add_argument("--max-streams", type=int, default=64, help="capacity search ceiling, not a claimed maximum (default: 64)")
    parser.add_argument("--chunks", type=int, help="initial chunks per stream; default/minimum: 1 + 16 * block count")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="A corgi running along a beach at sunset.")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output", type=Path, help="new report directory (default: a fresh directory under bench-results)")
    outputs.add_argument("--resume", type=Path, help="reuse completed measurements from an identical interrupted run")
    args = parser.parse_args()
    if (min(args.max_streams, args.repeats, args.warmups) < 1
            or (args.streams is not None and args.streams < 1)
            or not isfinite(args.slo_ms) or args.slo_ms <= 0):
        parser.error("require positive streams, repeats, warmups and finite positive SLO")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this benchmark")
    service = PhysisServe(args.model, pipeline_parallel_size=args.gpus, strategy="stage_equal")
    directory, report = None, None
    try:
        config = service.config
        trim = DENOISING_STEPS * config.num_blocks
        chunks = args.chunks if args.chunks is not None else 4 * trim + 1
        if chunks < 4 * trim + 1:
            parser.error(f"--chunks must be >= {4 * trim + 1} for this model's long-run benchmark")
        params = SamplingParams(num_chunks=chunks, height=args.height, width=args.width,
                                seed=args.seed, mode="dit_only").resolve(config)
        hardware = service._call(devices)
        settings = {key: value for key, value in vars(args).items() if key not in ("output", "resume")}
        provenance = identity(config, params, hardware)
        if args.resume:
            loaded = json.loads((args.resume.resolve() / "report.json").read_text())
            if loaded["settings"] != settings or loaded["identity"] != provenance:
                raise ValueError("resume requires the same model, hardware, software and benchmark settings")
            directory, report = args.resume.resolve(), loaded
            hardware = report["hardware"]
        else:
            if args.output:
                directory = args.output.resolve()
                directory.mkdir(parents=True, exist_ok=False)
            else:
                root = Path("bench-results")
                root.mkdir(exist_ok=True)
                directory = Path(mkdtemp(prefix="run-", dir=root)).resolve()
            report = {"identity": provenance, "settings": settings, "hardware": hardware,
                      "scope": "DiT delivery on rank-zero clock; no retained latents/VAE. Capacity is the largest verified point under measured candidates, bounded placement and stream ceiling, not a global optimum.",
                      "protocol": {"trim_chunks": trim, "initial_chunks": chunks, "memory_fraction": 0.8,
                                   "demand_margin": 1.2, "max_half_drift": 0.05, "extension_attempts": 3,
                                   "profile_ranks": list(range(1, config.ranks + 1)),
                                   "capacity_order": "exclusive StageEqual, pooled StageEqual, pooled VirtualRank; reversed when (stream count + seed) is odd",
                                   "exclusive_baseline": "divmod(pool GPUs, streams), disjoint GPUs; utilization=1 is reservation, not measured compute",
                                   "calibration": "real native blocks, random full-history KV fixtures; 3 warmups, 12 samples/cell",
                                   "weight_identity": "model path and config hashes; tensor payloads are not hashed"}}
        report["status"] = "running"
        report.pop("error", None)
        save(directory / "report.json", report)
        print(f"Reports: {directory}", flush=True)
        budgets = tuple(DeviceBudget(int(item["free_bytes"] * 0.8)) for item in hardware)
        calibration = directory / "calibration.json"
        if calibration.exists():
            raw = json.loads(calibration.read_text())
            try:
                cost_table(raw, config.num_blocks, args.max_batch_size)
            except ValueError:
                save(directory / f"calibration-failed-{len(list(directory.glob('calibration-failed-*')))}.json", raw)
                raw = None
        else:
            raw = None
        if raw is None or any(row["error"] for row in raw):
            raw = service._call(calibrate, params, args.prompt, args.max_batch_size)
            save(calibration, raw)
        costs = cost_table(raw, config.num_blocks, args.max_batch_size)
        save(directory / "costs.json", costs)
        if args.streams is None:
            profiles = candidate_profiles(service, params, budgets, costs, args, directory, trim)
            report["capacity"] = search_capacity(service, params, budgets, profiles, costs, args, directory, trim)
        else:
            comparisons = {}
            for method in ("stage_equal", "virtual_rank"):
                from bench import configure
                service._call(configure, method, costs, args.max_batch_size)
                point = run_point(service, params, None, budgets, args, directory / f"{method}.json", trim)
                comparisons[method] = {key: value for key, value in point.items() if key != "attempts"}
                if "steady_chunks_per_second" in point:
                    print(f"{method}: {point['steady_chunks_per_second']:.2f} steady chunks/s; "
                          f"max gap {point['max_gap_ms']:.2f} ms", flush=True)
            report["comparison"] = comparisons
            if all(p.get("status") == "measured" for p in comparisons.values()):
                se, vr = comparisons["stage_equal"], comparisons["virtual_rank"]
                report["steady_speedup"] = vr["steady_chunks_per_second"] / se["steady_chunks_per_second"]
                print(f"VirtualRank steady throughput: {report['steady_speedup']:.3f}x", flush=True)
        if args.streams is None:
            uncertain = any(item["unverified_profiles"] or item["unverified_points"] for item in report["capacity"].values())
            report["status"] = "partial_search" if uncertain else "bounded_search_completed"
        else:
            report["status"] = "completed" if all(p["status"] == "measured" for p in comparisons.values()) else "incomplete"
    except Exception as exc:
        if report is not None:
            report.update(status="incomplete", error=str(exc))
        raise
    finally:
        if directory is not None and report is not None:
            save(directory / "report.json", report)
        service.close()


if __name__ == "__main__":
    main()
