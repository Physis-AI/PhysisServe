"""Compare StageEqual and VirtualRank on identical streams in dit_only mode."""
import argparse
from bisect import bisect_left, bisect_right
import json
from pathlib import Path
from statistics import median

import torch

from physisserve import PhysisServe, SamplingParams
from physisserve.engine.sampler import DENOISING_STEPS


def measure(engine, prompts, params, warmups, repeats):
    """Worker-side DiT timing. Prompt encoding, loading, CPU copies and VAE are excluded."""
    engine.prepare(prompts, params)
    for _ in range(warmups):
        engine.run()
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
        engine.run(on_latent=completed, on_batch=batched)
        torch.cuda.synchronize(engine.device)
        measurements.append({
            "times": {index: [(chunk, start.elapsed_time(event)) for chunk, event in values]
                      for index, values in events.items()},
            "max_batch": largest_batch,
        })
    return measurements


def summarize(rank_results, streams, chunks, repeats, trim):
    """Measure the common middle, excluding fill, drain and early stream completion.

    bench uses the same full-pool layout for every stream, so all final stages
    share one rank and one CUDA clock, including in a multi-node launch.
    """
    durations, windows, rates = [], [], []
    gaps, max_batch, max_gap, half_drift = [[] for _ in range(streams)], 0, 0.0, 0.0
    for repeat in range(repeats):
        owners, = (rank[repeat]["times"] for rank in rank_results if rank[repeat]["times"])
        max_batch = max(max_batch, *(rank[repeat]["max_batch"] for rank in rank_results))
        times = [[time for _, time in sorted(owners[index])] for index in range(streams)]
        start = max(values[trim] for values in times)
        end = min(values[chunks - 1 - trim] for values in times)
        rate = 0.0
        for index, values in enumerate(times):
            first = max(trim, bisect_left(values, start))
            last = min(chunks - 1 - trim, bisect_right(values, end) - 1)
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
            "full_chunks_per_second": streams * chunks * 1000 / completion_ms,
            "window_ms": median(windows), "max_gap_ms": max_gap, "half_drift": half_drift,
            "max_batch": max_batch}


def main():
    # Worker callbacks must be importable when only rank zero runs this script.
    from bench import measure

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Diffusers directory or Hugging Face repository")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--chunks", type=int,
                        help="chunks per stream; default/minimum: 1 + 16 * model block count")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-costs", type=Path, help="JSON per-block measured costs for batch sizes 1..max-batch-size")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="A corgi running along a beach at sunset.")
    args = parser.parse_args()
    if (args.streams < 1 or args.repeats < 1 or args.warmups < 0
            or (args.chunks is not None and args.chunks < 1)):
        parser.error("require positive streams/repeats/chunks and nonnegative warmups")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this benchmark")
    batch_costs = json.loads(args.batch_costs.read_text()) if args.batch_costs else None
    print(f"VirtualRank costs: {args.batch_costs}" if args.batch_costs else
          "VirtualRank costs: unprofiled equal-block batch weights")
    prompts = [args.prompt] * args.streams
    reports = {}
    for strategy in ("stage_equal", "virtual_rank"):
        service = PhysisServe(args.model, pipeline_parallel_size=args.gpus, strategy=strategy,
                              max_batch_size=1 if strategy == "stage_equal" else args.max_batch_size,
                              batch_costs=batch_costs if strategy == "virtual_rank" else None)
        try:
            trim = DENOISING_STEPS * service.config.num_blocks
            chunks = args.chunks if args.chunks is not None else 4 * trim + 1
            if chunks < 4 * trim + 1:
                parser.error(f"--chunks must be >= {4 * trim + 1} for this model's long-run benchmark")
            params = [SamplingParams(num_chunks=chunks, height=args.height, width=args.width,
                                     seed=args.seed + index, mode="dit_only")
                      for index in range(args.streams)]
            print(f"{strategy}: {chunks} chunks/stream, trim {trim} at each end, "
                  f"{args.warmups} warmup(s), {args.repeats} repeat(s)", flush=True)
            inputs = service._inputs(prompts, params)
            values = service._call(measure, *inputs, args.warmups, args.repeats)
            reports[strategy] = summarize(values, args.streams, chunks, args.repeats, trim)
        finally:
            service.close()
        report = reports[strategy]
        print(f"{strategy:12s}  completion {report['completion_ms']:.2f} ms  "
              f"full-run {report['full_chunks_per_second']:.2f} chunks/s  "
              f"observed batch {report['max_batch']}")
        print(f"  common middle {report['window_ms']:.2f} ms: "
              f"steady mean gap {report['chunk_gap_ms']:.2f} ms  "
              f"{report['steady_chunks_per_second']:.2f} chunks/s  "
              f"max gap {report['max_gap_ms']:.2f} ms  half-window drift {report['half_drift']:.1%}")
        if args.streams > 1:
            print("  per-stream steady mean gaps (ms):", ", ".join(f"{value:.2f}" for value in report["stream_gap_ms"]))
    baseline, candidate = reports["stage_equal"], reports["virtual_rank"]
    print(f"VirtualRank speedup: steady gap {baseline['chunk_gap_ms'] / candidate['chunk_gap_ms']:.3f}x, "
          f"steady throughput {candidate['steady_chunks_per_second'] / baseline['steady_chunks_per_second']:.3f}x; "
          f"full-run completion {baseline['completion_ms'] / candidate['completion_ms']:.3f}x")


if __name__ == "__main__":
    main()
