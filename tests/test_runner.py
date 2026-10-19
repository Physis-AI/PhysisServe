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

from datetime import timedelta
from itertools import zip_longest
import multiprocessing
import os
from pathlib import Path
import socket
import time

import pytest
import torch
import torch.distributed as dist

from tests import reference
from physisserve.engine.model_runner import ModelRunner, _execution_plan
from physisserve.engine.sampler import Sampler
from physisserve.engine.strategies import StageEqual, VirtualRank
from physisserve.engine.strategies.placement import MappedLayout
from physisserve.engine.transport import Distributed, Local


SHAPE = (1, 2, 1, 2, 2)
SAMPLER = Sampler((1.0, 0.75, 0.5, 0.25, 0.0), (1.0, 2.0, 3.0, 4.0))


class TinyDiT:
    """Batched tensor arithmetic with history-sensitive outputs and native KV pages."""

    num_blocks = 3

    def __init__(self, first_frames=3, sink_sigma=0.0):
        self.first_frames, self.sink_sigma = first_frames, sink_sigma

    def load(self, blocks, device, dtype):
        self.loaded = tuple(blocks)

    def prepare(self, condition, shape, chunks, fps=None):
        self.condition = 0.0 if condition is None else condition
        self.batch_sizes = []

    def prepare_timesteps(self, timesteps):
        pass

    def timestep_for_sigma(self, sigma):
        return 9.0 + 1000 * sigma

    def hidden_shape(self, shape):
        return shape

    def kv_shape(self, shape):
        return shape[0], 1, shape[2], 1

    def embed(self, latent, timestep, chunk):
        return latent + self.condition

    def block(self, index, hidden, timestep, chunk, cache):
        return self.block_batch(index, [hidden], [timestep], [chunk], [cache])[0]

    def block_batch(self, index, hidden, timesteps, chunks, caches):
        assert index in self.loaded
        self.batch_sizes.append(len(hidden))
        stacked = torch.cat(hidden)
        pages = stacked.mean(dim=(1, 3, 4))[:, None, :, None]
        contexts = torch.cat([cache(pages[i:i + 1], pages[i:i + 1])[1]
                              for i, cache in enumerate(caches)])
        history = contexts.mean(dim=(1, 2, 3)).reshape(-1, 1, 1, 1, 1)
        times = torch.stack(timesteps).reshape(-1, 1, 1, 1, 1)
        outputs = 0.7 * stacked + 0.2 * history + 0.001 * index + times * 1e-5
        return list(outputs.split(1))

    def unembed(self, hidden, timestep, chunk):
        return hidden


LAYOUTS = [StageEqual(1), StageEqual(2), StageEqual(), VirtualRank(1), VirtualRank(2),
           VirtualRank(2, max_batch=1), MappedLayout(VirtualRank(3), (0, 1, 0), 2)]


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("first_frames,sink_sigma", [(3, 0.0), (1, 0.001)])
def test_layouts_match_serial_clean_sink_and_bounded_history(layout, first_frames, sink_sigma):
    expected = reference.run(TinyDiT(first_frames, sink_sigma), SAMPLER, 0.3, SHAPE, 26, seed=17)
    runner = ModelRunner(TinyDiT(first_frames, sink_sigma), SAMPLER, layout)
    for _ in range(2):
        outputs = list(runner.run(0.3, SHAPE, 26, seed=17))
        assert [chunk for chunk, _ in outputs] == list(range(26))
        assert outputs[0][1].shape[2] == first_frames
        assert all(latent.shape[2] == 1 for _, latent in outputs[1:])
        actual = torch.cat([latent for _, latent in outputs], dim=2)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        if getattr(layout, "max_batch", 1) > 1:
            assert max(runner.dit.batch_sizes) > 1


@pytest.mark.parametrize("layout", [StageEqual(2), VirtualRank(2)])
def test_interleaved_streams_match_independent_runs(layout):
    runners = [ModelRunner(TinyDiT(), SAMPLER, layout) for _ in range(2)]
    expected = [list(runner.run(condition, SHAPE, count, seed=seed))
                for runner, condition, count, seed in zip(runners, [0.2, 0.4], [24, 27], [3, 8])]
    iterators = [runner.run_ticks(condition, SHAPE, count, seed=seed)
                 for runner, condition, count, seed in zip(runners, [0.2, 0.4], [24, 27], [3, 8])]
    actual = [[], []]
    for pair in zip_longest(*iterators):
        for index, tick in enumerate(pair):
            if tick is not None:
                actual[index].extend(tick)
        assert not torch.is_inference_mode_enabled()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not torch.equal(actual[0][0][1], actual[1][0][1])


@pytest.mark.parametrize("chunks", [1, 7])
def test_tick_counts_include_clean_sink_bootstrap(chunks):
    layout = VirtualRank(2)
    runner = ModelRunner(TinyDiT(), SAMPLER, layout)
    head = layout.schedule(3, 1)
    ticks = list(runner.run_ticks(None, SHAPE, chunks))
    expected = head.ticks + (4 + layout.schedule(3, chunks - 1).ticks if chunks > 1 else 0)
    assert len(ticks) == expected
    assert all(not torch.is_inference(latent) for tick in ticks for _, latent in tick)
    if chunks > 1:
        assert ticks[head.ticks:head.ticks + 4] == [[]] * 4


@pytest.mark.parametrize("family", ["wan21_1p3b", "wan21_14b", "wan22_5b", "ltx2", "h3"])
def test_native_clean_sink_pages_match_transfer_dtype(tmp_path, family):
    from tests.test_models import tiny_transformer

    torch.set_num_threads(1)
    model, adapter, channels, text_dim = tiny_transformer(family)
    model.save_pretrained(tmp_path)
    dit = adapter(tmp_path, compute_dtype=torch.bfloat16) if family == "ltx2" else adapter(tmp_path)
    state_dtype = torch.float32 if family == "ltx2" else torch.bfloat16
    stages = set()

    class SinkTransport(Local):
        def transfer(self, items):
            for tag, _, _, spec, tensors in items:
                if isinstance(tag, tuple) and tag[0] == "sink":
                    stages.add(tag[1:])
                    assert len(tensors) == len(spec) == 2
                    for tensor, (shape, dtype) in zip(tensors, spec):
                        assert tensor.shape == shape
                        assert tensor.dtype == dtype == torch.bfloat16
            return super().transfer(items)

    runner = ModelRunner(dit, SAMPLER, StageEqual(2), dtype=state_dtype, transport=SinkTransport())
    outputs = list(runner.run({"text": torch.randn(1, 5, text_dim)},
                              (1, channels, 1, 4, 4), 2, fps=24))
    assert stages == {(step, block) for step in range(4) for block in range(dit.num_blocks)}
    assert len(outputs) == 2
    assert all(latent.dtype == state_dtype and torch.isfinite(latent).all() for _, latent in outputs)


def test_closed_and_failed_requests_release_state_before_reuse():
    runner = ModelRunner(TinyDiT(), SAMPLER, StageEqual(2))
    expected = list(runner.run(None, SHAPE, 4))
    active = runner.run_ticks(None, SHAPE, 4)
    next(active)
    active.close()
    torch.testing.assert_close(list(runner.run(None, SHAPE, 4)), expected, rtol=0, atol=0)

    def fail_after_batch(rank, size):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        list(runner.run(None, SHAPE, 4, on_batch=fail_after_batch))
    torch.testing.assert_close(list(runner.run(None, SHAPE, 4)), expected, rtol=0, atol=0)


def test_rank_local_plans_preserve_transfer_order_and_credit_edges():
    layout = VirtualRank(3)
    schedule = layout.schedule(3, 25)
    spec = (SHAPE, torch.float32)
    global_batches, global_moves = _execution_plan(schedule, lambda rank: True, spec, spec)
    for rank in range(3):
        batches, moves = _execution_plan(schedule, lambda other: other == rank, spec, spec)
        for tick in range(schedule.ticks):
            assert batches[tick] == tuple(batch for batch in global_batches[tick] if batch[0].rank == rank)
            assert moves[tick] == [move for move in global_moves[tick] if rank in move[1:3]]
    credits = [move for tick in global_moves for move in tick if isinstance(move[0], tuple)]
    assert len(credits) == 25 - schedule.admission_window


def _distributed_worker(rank, directory, strategy):
    torch.set_num_threads(1)
    interfaces = {name for _, name in socket.if_nameindex()}
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if "lo0" in interfaces else "lo"
    dist.init_process_group("gloo", rank=rank, world_size=2,
        init_method=(Path(directory) / "rendezvous").as_uri(), timeout=timedelta(seconds=30))
    try:
        layout = StageEqual(2) if strategy == "stage_equal" else VirtualRank(2)
        runners = [ModelRunner(TinyDiT(), SAMPLER, layout, transport=Distributed("cpu")) for _ in range(2)]
        streams = [runner.run_ticks(condition, SHAPE, chunks, seed=seed)
                   for runner, condition, chunks, seed in zip(runners, [0.2, 0.4], [24, 27], [3, 8])]
        outputs = [[], []]
        for pair in zip_longest(*streams):
            for index, tick in enumerate(pair):
                if tick is not None:
                    outputs[index].extend(tick)
        if rank == 0:
            torch.save(outputs, Path(directory) / "outputs.pt")
        else:
            assert outputs == [[], []]
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
@pytest.mark.parametrize("strategy", ["stage_equal", "virtual_rank"])
def test_two_rank_interleaving_preserves_single_stream_outputs(tmp_path, strategy):
    context = multiprocessing.get_context("spawn")
    processes = [context.Process(target=_distributed_worker, args=(rank, str(tmp_path), strategy))
                 for rank in range(2)]
    deadline = time.monotonic() + 60
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0, deadline - time.monotonic()))
        assert not any(process.is_alive() for process in processes), "distributed test exceeded 60 seconds"
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
            if process.is_alive():
                process.kill()
                process.join(5)
            if process.pid is not None:
                process.close()
    actual = torch.load(tmp_path / "outputs.pt", weights_only=True)
    for outputs, condition, chunks, seed in zip(actual, [0.2, 0.4], [24, 27], [3, 8]):
        expected = reference.run(TinyDiT(), SAMPLER, condition, SHAPE, chunks, seed=seed)
        assert [chunk for chunk, _ in outputs] == list(range(chunks))
        torch.testing.assert_close(torch.cat([latent for _, latent in outputs], dim=2), expected, rtol=0, atol=0)
