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
from types import SimpleNamespace
import weakref

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from physisserve.engine.engine import Engine
from physisserve.engine.kv_cache import BatchCache, Cache
from physisserve.engine.sampler import Sampler
from physisserve.engine.strategies import StageEqual, VirtualRank
from physisserve.engine.strategies.placement import Candidate, DeviceBudget, RankDemand, StreamCandidates
from physisserve.sampling_params import SamplingParams


@pytest.mark.parametrize("history_lengths", [(), (3, 1, 1)])
def test_batch_cache_matches_serial_with_independent_pages(history_lengths):
    batch, members = 2, 3
    histories = [[tuple(torch.randn(batch, 2, tokens, 4) for _ in range(2))
                  for tokens in history_lengths] for _ in range(members)]
    originals = [[tuple(value.clone() for value in page) for page in history] for history in histories]
    caches = [Cache(history) for history in histories]
    key, value = (torch.randn(members * batch, 2, 1, 4) for _ in range(2))
    with torch.inference_mode():
        serial = [Cache(history)(k, v) for history, k, v in zip(histories, key.split(batch), value.split(batch))]
        expected = tuple(torch.cat([page[i] for page in serial], dim=0) for i in (0, 1))
        actual = BatchCache(caches, batch)(key, value)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for component, source in enumerate((key, value)):
        pages = [cache.page[component] for cache in caches]
        pointers = [page.untyped_storage().data_ptr() for page in pages]
        assert len(set(pointers)) == members
        assert source.untyped_storage().data_ptr() not in pointers
        assert actual[component].untyped_storage().data_ptr() not in pointers
        assert all(page.untyped_storage().nbytes() == page.numel() * page.element_size() for page in pages)
        torch.testing.assert_close(torch.cat(pages), source, rtol=0, atol=0)
    torch.testing.assert_close(histories, originals, rtol=0, atol=0)


class FakeDiT:
    num_blocks = 2
    sink_sigma = 0.0

    def __init__(self, config):
        self.config, self.first_frames = config, config.first_frames

    def load(self, blocks, device, dtype):
        self.loaded = tuple(blocks)
        self.config.loaded.append(self)

    def prepare(self, condition, shape, chunks, fps=None):
        self.condition = condition["text"].item()
        assert fps is None or fps == self.config.fps

    def prepare_timesteps(self, timesteps):
        pass

    def timestep_for_sigma(self, sigma):
        return sigma * 1000

    def hidden_shape(self, shape):
        return shape

    def kv_shape(self, shape):
        return (1, 1, 1, 1)

    def embed(self, latent, timestep, chunk):
        return latent + self.condition / 10000

    def block(self, index, hidden, timestep, chunk, cache):
        assert index in self.loaded
        page = hidden.mean().reshape(1, 1, 1, 1)
        _, values = cache(page, page)
        return hidden * 0.95 + values.mean() * 0.01 + index * 0.0001

    def block_batch(self, index, hidden, timesteps, chunks, caches):
        return [self.block(index, value, timestep, chunk, cache)
                for value, timestep, chunk, cache in zip(hidden, timesteps, chunks, caches)]

    def unembed(self, hidden, timestep, chunk):
        return hidden * 0.1


class FakeEncoder:
    def __init__(self, config):
        self.config = config

    def load(self, device, dtype):
        self.config.encoder_loads += 1

    def encode(self, prompt):
        self.config.encoded.append(prompt)
        return {"text": torch.tensor([sum(map(ord, prompt))], dtype=torch.float32)}


class FakeDecoder:
    def __init__(self, config):
        self.config = config

    def load(self, device, dtype):
        self.config.decoder_loads += 1

    def reset(self):
        self.next_chunk = 0
        self.config.resets += 1

    def decode(self, latent, chunk):
        assert chunk == self.next_chunk
        self.next_chunk += 1
        pixels = latent[:, :1].repeat(1, 3, 2, 1, 1)
        return pixels[:, :, 1:] if chunk == 0 else pixels


class SequenceDecoder(FakeDecoder):
    def decode_sequence(self, latent):
        self.config.sequences.append(tuple(latent.shape))
        return latent[:, :1].repeat(1, 3, 2, 1, 1)[:, :, 1:]

    def decode(self, latent, chunk):
        raise AssertionError("a whole-sequence VAE must not decode individual chunks")


class FakeConfig:
    device, dtype, compute_dtype, encoder_device, model = "cpu", "float32", None, "cpu", "fake"
    channels, height, width, spatial_scale, patch_size = 2, 2, 2, 1, (1, 1, 1)
    chunks, fps, backend, supports_online_decode, min_decode_chunks = 5, 16, "fake", True, 1

    def __init__(self, strategy="virtual_rank", first_frames=3, sequence=False, ranks=1):
        self.strategy, self.first_frames, self.sequence, self.ranks = strategy, first_frames, sequence, ranks
        self.loaded, self.encoded, self.sequences = [], [], []
        self.encoder_loads = self.decoder_loads = self.resets = 0

    def make_backend(self):
        decoder = SequenceDecoder if self.sequence else FakeDecoder
        return SimpleNamespace(dit=FakeDiT(self), encoder=FakeEncoder(self), decoder=decoder(self))

    def make_layout(self):
        return StageEqual(self.ranks) if self.strategy == "stage_equal" else VirtualRank(self.ranks)

    def make_sampler(self):
        return Sampler((1.0, 0.75, 0.5, 0.25, 0.0), (1000.0, 750.0, 500.0, 250.0))


def inputs():
    return ["cat", "dog"], [SamplingParams(num_chunks=chunks, height=2, width=2, seed=seed, mode="dit_only")
                           for chunks, seed in ((5, 3), (4, 8))]


@pytest.mark.parametrize("strategy", ["stage_equal", "virtual_rank"])
@pytest.mark.parametrize("first_frames", [1, 3])
def test_interleaved_streams_equal_independent_requests_and_reuse_weights(strategy, first_frames):
    prompts, params = inputs()
    expected = []
    for prompt, param in zip(prompts, params):
        single = Engine(FakeConfig(strategy, first_frames))
        single.prepare([prompt], [param])
        expected.append(single.run()[0]["latents"])
    config = FakeConfig(strategy, first_frames)
    engine = Engine(config)
    engine.prepare(prompts, params)
    batches, completions, latent_shapes = [], [], {}

    def completed(index, chunk, latent):
        completions.append((index, chunk))
        latent_shapes[index, chunk] = tuple(latent.shape)

    results = engine.run(on_batch=lambda index, rank, size: batches.append(index),
                         on_latent=completed)
    assert batches[:2] == [0, 1]
    assert completions.index((1, 0)) < completions.index((0, params[0].num_chunks - 1))
    assert len(config.loaded) == 2 and config.encoder_loads == 1 and config.decoder_loads == 0
    assert [output["prompt"] for output in results] == ["cat", "dog"]
    for index, (output, reference) in enumerate(zip(results, expected)):
        param = params[index]
        assert [chunk for stream, chunk in completions if stream == index] == list(range(param.num_chunks))
        for chunk in range(param.num_chunks):
            frames = first_frames if chunk == 0 else 1
            assert latent_shapes[index, chunk] == (1, config.channels, frames, param.height, param.width)
        assert output["chunks"] == param.num_chunks and output["first_frames"] == first_frames
        assert output["fps"] == config.fps
        torch.testing.assert_close(output["latents"], reference, rtol=0, atol=0)
        assert output["latents"].device.type == "cpu" and output["latents"].is_contiguous()
    again = engine.run()
    engine.prepare(list(reversed(prompts)), list(reversed(params)))
    swapped = engine.run()
    assert len(config.loaded) == 2
    for index in range(2):
        assert torch.equal(results[index]["latents"], again[index]["latents"])
        assert torch.equal(results[index]["latents"], swapped[1 - index]["latents"])


@pytest.mark.parametrize("strategy", ["stage_equal", "virtual_rank"])
@pytest.mark.parametrize("first_frames", [1, 3])
def test_timing_run_delivers_without_retaining_outputs_and_preserves_next_run(strategy, first_frames, monkeypatch):
    config = FakeConfig(strategy, first_frames)
    engine = Engine(config)
    prompts, params = inputs()
    engine.prepare(prompts, params)
    expected = engine.run()
    produced, delivered, references = [], [], {}

    def output(index, chunk, latent):
        assert (index, chunk) in produced
        delivered.append((index, chunk))
        start = 0 if chunk == 0 else first_frames + chunk - 1
        stop = first_frames if chunk == 0 else start + 1
        torch.testing.assert_close(latent, expected[index]["latents"][:, :, start:stop], rtol=0, atol=0)
        if chunk > 1:
            assert references[index, 0]() is None
        references[index, chunk] = weakref.ref(latent)

    concatenate = torch.cat

    def no_video_concat(tensors, *args, **kwargs):
        assert tensors[0].ndim != 5, "timing runs must not concatenate completed latents"
        return concatenate(tensors, *args, **kwargs)

    def no_cpu_copy(*args, **kwargs):
        pytest.fail("timing runs must not materialize CPU results")

    with monkeypatch.context() as patch:
        patch.setattr(torch, "cat", no_video_concat)
        patch.setattr(torch.Tensor, "cpu", no_cpu_copy)
        result = engine.run(on_latent=lambda index, chunk, latent: produced.append((index, chunk)),
                            on_output=output, return_outputs=False)
    assert result is None
    assert produced == delivered
    assert all(reference() is None for reference in references.values())
    for index, param in enumerate(params):
        assert [chunk for stream, chunk in delivered if stream == index] == list(range(param.num_chunks))
    assert delivered.index((1, 0)) < delivered.index((0, params[0].num_chunks - 1))
    again = engine.run()
    assert len(config.loaded) == 2 and config.encoder_loads == 1 and config.decoder_loads == 0
    for output, reference in zip(again, expected):
        torch.testing.assert_close(output["latents"], reference["latents"], rtol=0, atol=0)


@pytest.mark.parametrize("sequence", [False, True])
def test_offline_decode_resets_each_video_and_accounts_for_first_chunk(sequence, monkeypatch):
    config = FakeConfig(sequence=sequence)
    engine = Engine(config)
    engine.prepare(*inputs())
    outputs = engine.run()
    assert config.decoder_loads == 0
    load = FakeDecoder.load

    def load_after_dit_release(decoder, device, dtype):
        assert engine.sessions == () and engine.runners == {}
        load(decoder, device, dtype)

    monkeypatch.setattr(FakeDecoder, "load", load_after_dit_release)
    decoded = engine.decode(outputs)
    assert config.decoder_loads == 1 and config.resets == 2
    for raw, output in zip(outputs, decoded):
        assert "video" not in raw
        assert output["video"].dtype == torch.uint8
        assert not torch.is_inference(output["video"])
        assert output["video"].shape == (raw["latents"].shape[2] * 2 - 1, 2, 2, 3)
    if sequence:
        assert config.sequences == [tuple(output["latents"].shape) for output in outputs]
    engine.prepare(*inputs())
    assert len(engine.run()) == 2


def test_placement_preserves_input_order_and_reports_rejections_without_loading():
    config = FakeConfig()
    engine = Engine(config)
    candidate = Candidate("one", StageEqual(1), (RankDemand(1, 0.1),), 1.0)
    too_large = Candidate("two", StageEqual(1), (RankDemand(2, 0.1),), 1.0)
    candidates = [StreamCandidates("0", 2.0, (too_large,)), StreamCandidates("1", 2.0, (candidate,))]
    engine.prepare(*inputs(), candidates, [DeviceBudget(1)])
    results = engine.run()
    assert results[0] == {"prompt": "cat", "fps": 16, "status": "rejected",
                          "reason": "no placement found by this bounded policy; infeasibility not proved"}
    assert results[1]["prompt"] == "dog" and results[1]["latents"].shape[2] == 6
    assert len(config.loaded) == 1 and config.encoded == ["dog"]
    assert engine.decode(results)[0] == results[0]
    engine.prepare([], [])
    assert engine.run() == [] and not engine.runners


@pytest.mark.parametrize("layout", [StageEqual(2), VirtualRank(2)])
def test_internal_pool_colocates_stream_stages_without_changing_single_stream_latents(layout):
    prompts, params = inputs()
    single = Engine(FakeConfig("stage_equal"))
    expected = []
    for prompt, param in zip(prompts, params):
        single.prepare([prompt], [param])
        expected.append(single.run()[0]["latents"])

    candidate = Candidate("two ranks", layout, (RankDemand(1, 0.1),) * 2, 1.0)
    candidates = [StreamCandidates(str(index), 2.0, (candidate,)) for index in range(2)]
    engine = Engine(FakeConfig())
    engine.prepare(prompts, params, candidates, [DeviceBudget(4)])
    assert all(runner.layout.rank_map == (0, 0) for _, runner in engine.runners.values())
    outputs = engine.run()
    assert [output["prompt"] for output in outputs] == prompts
    for output, latent in zip(outputs, expected):
        torch.testing.assert_close(output["latents"], latent, rtol=0, atol=0)


def test_failed_callback_closes_stream_iterators_before_reuse(monkeypatch):
    engine = Engine(FakeConfig())
    engine.prepare(*inputs())
    closed = []
    for _, runner in engine.runners.values():
        run_ticks = runner.run_ticks

        def tracked(*args, run_ticks=run_ticks, runner=runner, **kwargs):
            try:
                yield from run_ticks(*args, **kwargs)
            finally:
                closed.append(runner)

        monkeypatch.setattr(runner, "run_ticks", tracked)

    def callback(*args):
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        engine.run(on_latent=callback)
    assert set(closed) == {runner for _, runner in engine.runners.values()}
    assert len(engine.run()) == 2


def distributed_worker(rank, rendezvous, optimized):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        config = FakeConfig(ranks=2)
        engine = Engine(config)
        # Rank zero receives every output even when it owns no model blocks.
        candidate = Candidate("one", StageEqual(1), (RankDemand(1, 0.1),), 1.0)
        candidates = [StreamCandidates(str(index), 2.0, (candidate,)) for index in range(2)]
        prompts, params = inputs()
        if optimized:
            small = Candidate("one", StageEqual(1), (RankDemand(1, 0.1),), 2.0, 2.0)
            fast = Candidate("two", StageEqual(2), (RankDemand(1, 0.1),) * 2, 1.0, 1.0)
            if optimized == "upgrade":
                prompts, params = prompts[:1], params[:1]
            candidates = [StreamCandidates(str(index), 2.0, (small, fast)) for index in range(len(prompts))]
        engine.prepare(prompts, params, candidates, [DeviceBudget(10), DeviceBudget(2)])
        if optimized:
            assert sorted(rank for _, runner in engine.runners.values() for rank in runner.layout.rank_map) == [0, 1]
            assert [len(runner.layout.rank_map) for _, runner in engine.runners.values()] == (
                [2] if optimized == "upgrade" else [1, 1])
        else:
            assert all(runner.layout.rank_map == (1,) for _, runner in engine.runners.values())
        # All remapping precedes inference; neither run may replan live KV homes.
        import physisserve.engine.engine as engine_module
        engine_module.plan_pool = lambda *args: pytest.fail("inference must not replan")
        results = engine.run()
        if rank == 0:
            assert len(results) == len(prompts)
            for prompt, param, result in zip(prompts, params, results):
                reference = FakeDiT(FakeConfig())
                from tests.reference import run
                condition = {"text": torch.tensor([sum(map(ord, prompt))], dtype=torch.float32)}
                shape = (1, config.channels, 1, param.height, param.width)
                expected = run(reference, config.make_sampler(), condition, shape, param.num_chunks, seed=param.seed)
                torch.testing.assert_close(result["latents"], expected, rtol=0, atol=0)
            if not optimized:
                assert all(not dit.loaded for dit in config.loaded)
        else:
            assert results is None
        produced, delivered = [], []
        assert engine.run(on_latent=lambda index, chunk, latent: produced.append((index, chunk)),
                          on_output=lambda index, chunk, latent: delivered.append((index, chunk)),
                          return_outputs=False) is None
        if rank == 0 or not optimized:
            observed = delivered if rank == 0 else produced
            for index, param in enumerate(params):
                assert [chunk for stream, chunk in observed if stream == index] == list(range(param.num_chunks))
        if rank != 0 or not optimized:
            assert not (produced if rank == 0 else delivered)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("optimized", [None, "spread", "upgrade"])
def test_distributed_placement_matches_single_stream_without_runtime_remapping(tmp_path, optimized):
    context = mp.spawn(distributed_worker, args=(str(tmp_path / "rendezvous"), optimized), nprocs=2, join=False)
    try:
        # spawn.join can return after a single child exits; wait once per child.
        for _ in range(3):
            if context.join(timeout=40):
                break
        else:
            pytest.fail("distributed workers exceeded their deadline")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
