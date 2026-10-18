"""Compare StageEqual and VirtualRank on identical single or concurrent streams."""
import argparse
import json
from pathlib import Path
from statistics import median

import torch

from physisserve import PhysisServe, SamplingParams


def measure(engine, requests, warmups, repeats):
    """Worker-side DiT timing. Prompt encoding, loading, CPU copies and VAE are excluded."""
    engine.prepare(requests)
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


def summarize(rank_results, streams, chunks, repeats, skip_chunks):
    """Each stream's final stage has one rank, so its completion gaps share one CUDA clock."""
    if not 0 <= skip_chunks < chunks - 1:
        raise ValueError("skip_chunks must leave at least two measured chunks")
    durations, gaps, max_batch = [], [[] for _ in range(streams)], 0
    for repeat in range(repeats):
        owners, duration = {}, 0.0
        for rank in rank_results:
            row = rank[repeat]
            max_batch = max(max_batch, row["max_batch"])
            for index, values in row["times"].items():
                if index in owners:
                    raise ValueError("a stream's final stage must have exactly one owner")
                owners[index] = values
        if set(owners) != set(range(streams)):
            raise ValueError("missing stream completion measurements")
        for index, values in owners.items():
            values = sorted(values)
            if [chunk for chunk, _ in values] != list(range(chunks)):
                raise ValueError("missing or duplicate chunk completions")
            timestamps = [time for _, time in values]
            intervals = [right - left for left, right in zip(timestamps, timestamps[1:])]
            if timestamps[0] < 0 or any(interval < 0 for interval in intervals):
                raise ValueError("chunk completion times must be nondecreasing")
            duration = max(duration, timestamps[-1])
            # A native batch can finish several chunks together. Average the
            # entire steady window so short within-batch gaps cannot inflate speedup.
            steady = intervals[skip_chunks:]
            gaps[index].append(sum(steady) / len(steady))
        if duration <= 0:
            raise ValueError("completion duration must be positive")
        durations.append(duration)
    completion_ms = median(durations)
    per_stream = [median(values) for values in gaps]
    return {"completion_ms": completion_ms, "chunk_gap_ms": median(per_stream),
            "stream_gap_ms": per_stream, "chunks_per_second": streams * chunks * 1000 / completion_ms,
            "max_batch": max_batch}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Diffusers directory or Hugging Face repository")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--streams", type=int, default=1)
    parser.add_argument("--chunks", type=int, default=40)
    parser.add_argument("--skip-chunks", type=int, default=24, help="discard bootstrap/fill completion gaps")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--batch-costs", type=Path, help="JSON per-block measured costs for batch sizes 1..max-batch-size")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="A corgi running along a beach at sunset.")
    args = parser.parse_args()
    if args.streams < 1 or args.repeats < 1 or args.warmups < 0 or not 0 <= args.skip_chunks < args.chunks - 1:
        parser.error("require positive streams/repeats, nonnegative warmups, and chunks > skip-chunks + 1")
    if not torch.cuda.is_available():
        parser.error("CUDA is required for this benchmark")
    batch_costs = json.loads(args.batch_costs.read_text()) if args.batch_costs else None
    print(f"VirtualRank costs: {args.batch_costs}" if args.batch_costs else
          "VirtualRank costs: unprofiled equal-block batch weights")
    prompts = [args.prompt] * args.streams
    params = [SamplingParams(num_chunks=args.chunks, height=args.height, width=args.width,
                             seed=args.seed + index, mode="dit_only")
              for index in range(args.streams)]
    reports = {}
    for strategy in ("stage_equal", "virtual_rank"):
        service = PhysisServe(args.model, pipeline_parallel_size=args.gpus, strategy=strategy,
                              max_batch_size=1 if strategy == "stage_equal" else args.max_batch_size,
                              batch_costs=batch_costs if strategy == "virtual_rank" else None)
        try:
            requests = service._requests(prompts, params)
            values = service._call(measure, requests, args.warmups, args.repeats)
            reports[strategy] = summarize(values, args.streams, args.chunks, args.repeats, args.skip_chunks)
        finally:
            service.close()
        report = reports[strategy]
        print(f"{strategy:12s}  completion {report['completion_ms']:.2f} ms  "
              f"steady mean gap {report['chunk_gap_ms']:.2f} ms  {report['chunks_per_second']:.2f} chunks/s  "
              f"observed batch {report['max_batch']}")
        if args.streams > 1:
            print("  per-stream steady mean gaps (ms):", ", ".join(f"{value:.2f}" for value in report["stream_gap_ms"]))
    baseline, candidate = reports["stage_equal"], reports["virtual_rank"]
    print(f"VirtualRank speedup: completion {baseline['completion_ms'] / candidate['completion_ms']:.3f}x, "
          f"steady gap {baseline['chunk_gap_ms'] / candidate['chunk_gap_ms']:.3f}x")


if __name__ == "__main__":
    main()
