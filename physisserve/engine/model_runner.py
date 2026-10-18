"""Model runner: drives the DiT through the block schedule on one rank, tick by tick.

Each tick: run this rank's batches using local KV -> release expired pages -> move
(latent, hidden) to the successor cell, and finished latents to the output rank. All transfer lists
are static, so there is no collective metadata exchange. Moves are posted at the end of a tick and
waited for when their consumer cell runs. Transfers can overlap independent work between producer
and consumer; a logical tick does not imply equal block or batch durations. Each stage has a
fixed rank across chunks; historical KV never moves.
Plans may bound admitted chunks with completion credits: chunk i starts only
after chunk i-window completes. Credits use the same ordered asynchronous P2P
path as hidden states, not empty ticks or wall-clock sleeps.
Every chunk runs exactly the sampler's denoising passes. Each pass reads historical KV from that same pass.
"""
from typing import Iterator

import torch
import torch.distributed as dist

from .chunks import Chunks
from .kv_cache import Cache, KVCache
from .sampler import DENOISING_STEPS
from .scheduler import Task
from .transport import Distributed, Local


def generator(device, seed: int, chunk: int, step: int) -> torch.Generator:
    """step=-1 for the initial noise; the sampler's per-pass randomness uses the pass index."""
    return torch.Generator(device=device).manual_seed(seed * 1_000_003 + chunk * 4096 + step + 1)


def _execution_plan(schedule, is_local, latent_spec, hidden_spec):
    """Compile rank-local work once, preserving the global order of matching P2Ps.

    The DAG and ownership do not change. Nonparticipant ranks need not rebuild or
    scan another rank's transfers on every tick, including pipeline fill/drain.
    """
    batches = [tuple(batch for batch in tick if batch and is_local(batch[0].rank))
               for tick in schedule.by_tick_batches]
    moves = [[] for _ in range(schedule.ticks)]
    admission_window = getattr(schedule, "admission_window", None)
    for tick, tasks in enumerate(schedule.by_tick):
        for task in tasks:
            following = schedule.successor(task)
            if following is not None and (is_local(task.rank) or is_local(following.rank)):
                spec = [latent_spec] if following.block == 0 else [latent_spec, hidden_spec]
                moves[tick].append((following, task.rank, following.rank, spec, task))
            if task.step == DENOISING_STEPS - 1 and task.block == schedule.blocks - 1:
                if is_local(task.rank) or is_local(0):
                    moves[tick].append((task.chunk, task.rank, 0, [latent_spec], task))
                future = task.chunk + admission_window if admission_window is not None else None
                if future is not None and future < schedule.chunks:
                    destination = schedule.task(future, 0, 0).rank
                    if is_local(task.rank) or is_local(destination):
                        moves[tick].append((("admit", future), task.rank, destination,
                                            [((1,), torch.uint8)], task))
    return batches, moves


class ModelRunner:
    """One DiT, one sampler and one layout with Same-Step KV; one request at a time.

    In a single process every logical rank of the layout is simulated. Under torch.distributed the
    layout's ranks are processes 0..ranks-1 of the group and each loads only its own blocks. Finished
    latents go to rank zero.
    """

    def __init__(self, dit, sampler, layout, device="cpu", dtype=torch.float32,
                 transport=None):
        """A clean sink runs one extra model-native forward of chunk 0's final latent and pins its pages."""
        self.dit, self.sampler, self.layout = dit, sampler, layout
        self.first_frames, self.sink_sigma = dit.first_frames, dit.sink_sigma
        self.device, self.dtype = torch.device(device), dtype
        if transport is None:
            transport = Distributed(self.device) if dist.is_initialized() else Local()
        self.transport = transport
        blocks = (range(dit.num_blocks) if isinstance(transport, Local)
                  else layout.rank_blocks(transport.rank, dit.num_blocks))
        dit.load(blocks, self.device, dtype)
        self.hidden_dtype = getattr(dit, "hidden_dtype", None) or dtype
        self.page_dtype = getattr(dit, "page_dtype", None) or dtype

    def run(self, condition, shape, chunks: int, *, seed: int = 0, noise=None,
            fps: float | None = None, on_latent=None, on_batch=None) -> Iterator[tuple[int, torch.Tensor]]:
        """Yield (chunk, latent) on the output rank as chunks finish; other processes iterate to the end.

        condition, noise and seed must be identical on every DiT process. ``noise`` contains one tensor per
        chunk, using the model's fixed first-chunk length for chunk 0 and ``shape`` thereafter.
        ``on_latent(chunk, latent)`` is called
        on the rank that finishes a chunk's last denoising pass, before the latent is sent on; the
        latent it receives is an inference-mode tensor, read it or copy it, do not modify it in place.
        Optional ``on_batch(rank, size)`` counts actual block calls after enqueue; it does not time them.
        """
        ticks = self.run_ticks(condition, shape, chunks, seed=seed, noise=noise, fps=fps,
                               on_latent=on_latent, on_batch=on_batch)
        try:
            for outputs in ticks:
                yield from outputs
        finally:
            # Closing the flattening generator must also release the inner request.
            ticks.close()

    def run_ticks(self, condition, shape, chunks: int, *, seed: int = 0,
                  noise=None, fps: float | None = None,
                  on_latent=None, on_batch=None) -> Iterator[list[tuple[int, torch.Tensor]]]:
        """Yield one output list for every logical tick, including empty lists.

        This is the same execution as :meth:`run`, with explicit scheduling boundaries.
        Independent runners can interleave ticks, but must own independent backends:
        ``prepare`` and timestep state are request-local. For distributed interleaving,
        use a separate ``Distributed`` instance per runner on the common process group
        so completion flushes are isolated, and step runners in the same deterministic
        order on every process.
        Exhaust every iterator to flush transfers. Early close or failure releases local
        request state, but is NOT a recoverable distributed cancellation: the caller must
        abort the whole service/process group rather than resume unmatched peer traffic.
        Callback semantics and input requirements are identical to :meth:`run`.
        """
        geometry = Chunks(shape, chunks, self.first_frames)
        schedules = [self.layout.schedule(self.dit.num_blocks, 1)]
        if chunks > 1:
            schedules.append(self.layout.schedule(self.dit.num_blocks, chunks - 1))
        yield from self._run_sink_first(
            condition, geometry, schedules, seed, noise, fps, on_latent, on_batch)

    def _execute_ticks(self, shape, seed, noise, on_latent, on_batch, schedule, kv, states, transfer, *, offset=0):
        """``offset``: the schedule's chunk ``i`` is the request's chunk ``i + offset`` (noise, positions, outputs);
        ``shape`` is the shape of the schedule's chunks. The backend is already prepared for the request."""
        local = self.transport.is_local
        device, dtype, shape = self.device, self.dtype, tuple(shape)
        times = [torch.tensor(t, dtype=torch.float32) for t in self.sampler.timesteps]  # CPU scalars
        self.dit.prepare_timesteps(times)
        latent_spec, hidden_spec = (shape, dtype), (self.dit.hidden_shape(shape), self.hidden_dtype)
        last_block = self.dit.num_blocks - 1
        planned_batches, planned_moves = _execution_plan(
            schedule, local, latent_spec, hidden_spec)
        admission_window = getattr(schedule, "admission_window", None)
        # Immutable payload; P2P posting on the compute stream orders a credit
        # after the completed chunk's kernels, even though its value is reused.
        credit = (torch.ones(1, device=device, dtype=torch.uint8),) if admission_window is not None else None
        ticks = schedule.ticks
        del schedule  # fixed-home KV and the local plans retain only the Tasks this rank needs

        def initial(chunk):
            chunk += offset
            if noise is not None:
                return noise[chunk].to(device=device, dtype=dtype).clone()
            return torch.randn(shape, generator=generator(device, seed, chunk, -1), device=device, dtype=dtype)

        def is_output(task):
            return task.step == DENOISING_STEPS - 1 and task.block == last_block

        for tick in range(ticks):
            outputs = []
            with torch.inference_mode():
                done = {}
                for batch in planned_batches[tick]:
                    groups = {}
                    for task in batch:
                        cache = kv.context(task)
                        # Fill/drain can have different retained context lengths. Split explicitly
                        # instead of padding or pretending several serial calls are one batch.
                        groups.setdefault(len(cache.history), []).append((task, cache))
                    for members in groups.values():
                        samples, hidden_inputs, time_inputs = [], [], []
                        for task, cache in members:
                            if task.step == 0 and task.block == 0:
                                if admission_window is not None and task.chunk >= admission_window:
                                    states.pop(("admit", task.chunk)).wait()
                                sample, hidden = initial(task.chunk), None
                            else:
                                sample, *rest = states.pop(task).wait()
                                hidden = rest[0] if rest else None
                            timestep = times[task.step]
                            if task.block == 0:
                                hidden = self.dit.embed(sample, timestep, task.chunk + offset)
                            samples.append(sample)
                            hidden_inputs.append(hidden)
                            time_inputs.append(timestep)
                        if len(members) == 1:
                            task, cache = members[0]
                            hidden_outputs = [self.dit.block(task.block, hidden_inputs[0], time_inputs[0],
                                                            task.chunk + offset, cache)]
                        else:
                            hidden_outputs = self.dit.block_batch(batch[0].block, hidden_inputs, time_inputs,
                                [task.chunk + offset for task, _ in members], [cache for _, cache in members])
                        if on_batch is not None:
                            on_batch(batch[0].rank, len(members))
                        for (task, cache), sample, hidden, timestep in zip(
                                members, samples, hidden_outputs, time_inputs, strict=True):
                            kv.publish(task, cache)
                            if task.block == last_block:
                                prediction = self.dit.unembed(hidden, timestep, task.chunk + offset)
                                hidden = None
                                sample = self.sampler.advance(prediction, sample, task.step,
                                                              generator(device, seed, task.chunk + offset, task.step))
                            done[task] = (sample,) if hidden is None else (sample, hidden)
                            if is_output(task) and on_latent is not None:
                                on_latent(task.chunk + offset, sample)
                kv.release(tick)
                moves = [(tag, src, dst, spec, credit if isinstance(tag, tuple) else done.get(task))
                         for tag, src, dst, spec, task in planned_moves[tick]]
                for tag, pending in (transfer(moves).items() if moves else ()):
                    if isinstance(tag, (Task, tuple)):
                        states[tag] = pending
                    else:
                        outputs.append((tag, pending))
            # Ordinary tensors, outside inference_mode. Even ranks with no outputs
            # yield once so the outer multi-stream scheduler can advance one tick.
            yield [(chunk + offset, pending.wait()[0].clone()) for chunk, pending in outputs]
        self.transport.flush()

    def _run_sink_first(self, condition, geometry, schedules, seed, noise, fps, on_latent, on_batch):
        """Chunk 0 to completion first, then (clean sink) its sink forward, then every later chunk.

        Chunk 0 runs the layout's own one-chunk schedule in its own shape (``geometry.first_frames`` latent frames).
        The page of one extra forward of its final latent stays for the whole request. That forward runs block by block on
        the ranks that ran chunk 0's last pass; each page then goes to the rank that owns the reading stage, the
        only time KV moves. Later chunks run the layout's schedule of ``count - 1`` chunks, shifted by one chunk.
        Nothing changes the denoising of a chunk
        except the pages it reads: the same noise, ladder, history window and positions.
        """
        local, transfer = self.transport.is_local, self.transport.transfer
        chunks, blocks, last = geometry.count, self.dit.num_blocks, DENOISING_STEPS - 1
        first_shape, shape = geometry.shape_of(0), geometry.shape
        self.dit.prepare(condition, shape, chunks, fps=fps)
        head, *tail = schedules
        rest = tail[0] if tail else None
        final = {}

        def keep_final(chunk, latent):
            final[chunk] = latent
            if on_latent is not None:
                on_latent(chunk, latent)

        kv, states = KVCache(head, local), {}
        try:
            yield from self._execute_ticks(first_shape, seed, noise, keep_final, on_batch, head, kv, states, transfer)
        finally:
            kv.pages.clear()
            states.clear()
        if rest is None:
            return
        sink = {}
        homes = [head.task(0, last, block).rank for block in range(blocks)]
        pages = yield from self._sink_forward(final, homes, first_shape, seed, transfer, on_batch)
        readers = {}
        for task in rest.tasks:
            readers.setdefault((task.step, task.block), task.rank)
        page_spec = [(self.dit.kv_shape(first_shape), self.page_dtype)] * 2
        items = []
        for stage in sorted(readers):
            _, block = stage
            src, page = homes[block], pages.get(block)
            if not (local(src) or local(readers[stage])):
                continue
            items.append((("sink", *stage), src, readers[stage], page_spec, page if local(src) else None))
        for (_, step, block), pending in transfer(items).items():
            sink[0, step, block] = tuple(pending.wait())
        final.clear()
        # Make the page redistribution visible to multi-stream tick interleaving.
        yield []
        kv, states = KVCache(rest, local, offset=1, sink=sink), {}
        try:
            execution = self._execute_ticks(shape, seed, noise, on_latent, on_batch,
                                            rest, kv, states, transfer, offset=1)
            del rest
            yield from execution
        finally:
            states.clear()
            kv.pages.clear()
            sink.clear()

    def _sink_forward(self, final, homes, shape, seed, transfer, on_batch):
        """The clean sink pages: one forward of chunk 0's final latent through every block, block ``b`` on
        ``homes[b]``, without history. Each block is one explicit empty-output tick."""
        local, device, dtype = self.transport.is_local, self.device, self.dtype
        timestep = torch.tensor(self.dit.timestep_for_sigma(self.sink_sigma), dtype=torch.float32)
        latent_spec, hidden_spec = [(shape, dtype)], [(self.dit.hidden_shape(shape), self.hidden_dtype)]
        holder = homes[-1]  # where chunk 0's last pass ended (the rank of its last block)
        received = transfer([("sink-latent", holder, homes[0], latent_spec,
                              (final[0],) if local(holder) else None)]) if local(holder) or local(homes[0]) else {}
        pages, hidden, pending = {}, None, None
        for block, rank in enumerate(homes):
            with torch.inference_mode():
                if local(rank):
                    if block == 0:
                        sample = received["sink-latent"].wait()[0]
                        if self.sink_sigma:
                            noise = torch.randn(shape, generator=generator(device, seed, 0, DENOISING_STEPS),
                                                device=device, dtype=torch.float32)
                            sample = ((1.0 - self.sink_sigma) * sample.float() + self.sink_sigma * noise).to(dtype)
                        hidden = self.dit.embed(sample, timestep, 0)
                    elif hidden is None:
                        hidden = pending.wait()[0]
                    cache = Cache([])
                    hidden = self.dit.block(block, hidden, timestep, 0, cache)
                    pages[block] = cache.page
                    if on_batch is not None:
                        on_batch(rank, 1)
                if block + 1 < len(homes) and homes[block + 1] != rank and (local(rank) or local(homes[block + 1])):
                    moved = transfer([("sink-hidden", rank, homes[block + 1], hidden_spec,
                                       (hidden,) if local(rank) else None)])
                    pending = moved.get("sink-hidden")
                    hidden = None
            # Do not suspend inside inference_mode: it is thread-local state shared by interleaved runners.
            yield []
        self.transport.flush()
        return pages
