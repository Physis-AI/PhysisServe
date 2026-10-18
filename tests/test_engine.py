from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from physisserve.engine.engine import Engine
from physisserve.engine.sampler import Sampler
from physisserve.engine.strategies import StageEqual, VirtualRank
from physisserve.engine.strategies.placement import Candidate, PoolPlan, RankDemand, StreamPlacement
from physisserve.sampling_params import Request


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

    def frames(self, chunk, shape):
        return shape[2] * 2 - (chunk == 0)

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
    device, dtype, compute_dtype, encoder_device, model = "cpu", "float32", None, None, "fake"

    def __init__(self, strategy="virtual_rank", first_frames=3, sequence=False, ranks=1):
        self.strategy, self.first_frames, self.sequence, self.ranks = strategy, first_frames, sequence, ranks
        self.loaded, self.encoded, self.sequences = [], [], []
        self.encoder_loads = self.decoder_loads = self.resets = 0

    def make_backend(self, root):
        decoder = SequenceDecoder if self.sequence else FakeDecoder
        return SimpleNamespace(dit=FakeDiT(self), encoder=FakeEncoder(self), decoder=decoder(self))

    def make_layout(self):
        return StageEqual(self.ranks) if self.strategy == "stage_equal" else VirtualRank(self.ranks)

    def make_sampler(self):
        return Sampler.grid()


def requests():
    return [Request("cat", (1, 2, 1, 2, 2), 5, 3, 16),
            Request("dog", (1, 2, 1, 2, 2), 4, 8, 24)]


@pytest.mark.parametrize("strategy", ["stage_equal", "virtual_rank"])
@pytest.mark.parametrize("first_frames", [1, 3])
def test_interleaved_streams_equal_independent_requests_and_reuse_weights(strategy, first_frames):
    expected = []
    for request in requests():
        single = Engine(FakeConfig(strategy, first_frames))
        single.prepare([request])
        expected.append(single.run()[0]["latents"])
    config = FakeConfig(strategy, first_frames)
    engine = Engine(config)
    engine.prepare(requests())
    batches, completions = [], []
    results = engine.run(on_batch=lambda index, rank, size: batches.append(index),
                         on_latent=lambda index, chunk, value: completions.append((index, chunk)))
    assert batches[:2] == [0, 1]
    assert completions.index((1, 0)) < completions.index((0, requests()[0].chunks - 1))
    assert len(config.loaded) == 2 and config.encoder_loads == 1 and config.decoder_loads == 0
    assert [output["prompt"] for output in results] == ["cat", "dog"]
    for output, reference in zip(results, expected):
        torch.testing.assert_close(output["latents"], reference, rtol=0, atol=0)
        assert output["latents"].device.type == "cpu" and output["latents"].is_contiguous()
    again = engine.run()
    engine.prepare(list(reversed(requests())))
    swapped = engine.run()
    assert len(config.loaded) == 2
    for index in range(2):
        assert torch.equal(results[index]["latents"], again[index]["latents"])
        assert torch.equal(results[index]["latents"], swapped[1 - index]["latents"])


@pytest.mark.parametrize("sequence", [False, True])
def test_deferred_decode_resets_each_video_and_accounts_for_first_chunk(sequence, monkeypatch):
    config = FakeConfig(sequence=sequence)
    engine = Engine(config)
    engine.prepare(requests())
    outputs = engine.run()
    assert config.decoder_loads == 0
    load = FakeDecoder.load

    def load_after_dit_release(decoder, device, dtype):
        assert engine.sessions == () and engine.runners == {} and not engine.prepared
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
    engine.prepare(requests())
    assert len(engine.run()) == 2


def test_placement_preserves_input_order_and_reports_rejections_without_loading():
    config = FakeConfig()
    engine = Engine(config)
    candidate = Candidate("one", StageEqual(1), (RankDemand(1, 0.1),), 1.0)
    plan = PoolPlan((StreamPlacement("1", candidate, (0,)),), {"0": "pool full"}, (RankDemand(1, 0.1),))
    engine.prepare(requests(), plan)
    results = engine.run()
    assert results[0] == {"prompt": "cat", "fps": 16, "status": "rejected", "reason": "pool full"}
    assert results[1]["prompt"] == "dog" and results[1]["latents"].shape[2] == 6
    assert len(config.loaded) == 1 and config.encoded == ["dog"]
    assert engine.decode(results)[0] == results[0]
    engine.prepare([])
    assert engine.run() == [] and not engine.runners


def test_bad_placement_and_decoder_shapes_fail_explicitly(monkeypatch):
    engine = Engine(FakeConfig())
    with pytest.raises(ValueError, match="cover every input"):
        engine.prepare(requests(), PoolPlan((), {}, (RankDemand(0, 0.0),)))
    with pytest.raises(RuntimeError, match="prepare"):
        engine.run()
    engine.prepare(requests())
    outputs = engine.run()
    monkeypatch.setattr(FakeDecoder, "decode", lambda self, latent, chunk: torch.zeros(1, 3, 1, 2, 2))
    with pytest.raises(ValueError, match="decoder must return"):
        engine.decode(outputs)
    with pytest.raises(ValueError, match="metadata"):
        engine.decode([{**outputs[0], "chunks": 1}])


def test_failed_callback_releases_local_runner_and_reentrancy_is_rejected():
    engine = Engine(FakeConfig())
    engine.prepare(requests())

    def callback(*args):
        with pytest.raises(RuntimeError, match="already running"):
            engine.run()
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        engine.run(on_latent=callback)
    assert not engine.busy and not engine.failed
    assert all(not runner.busy for _, runner in engine.runners.values())
    assert len(engine.run()) == 2


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "shape"])
def test_incomplete_or_wrong_shape_runner_output_is_never_returned(monkeypatch, corruption):
    engine = Engine(FakeConfig())
    engine.prepare(requests()[:1])
    runner = engine.runners[0][1]
    run_ticks = runner.run_ticks

    def bad_outputs(*args, **kwargs):
        for outputs in run_ticks(*args, **kwargs):
            if outputs and outputs[0][0] == 0:
                if corruption == "missing":
                    outputs = []
                elif corruption == "duplicate":
                    outputs = outputs + outputs
                else:
                    outputs = [(0, outputs[0][1][:, :, :1])]
            yield outputs

    monkeypatch.setattr(runner, "run_ticks", bad_outputs)
    with pytest.raises(RuntimeError, match="missing or duplicate|incorrect latent shape"):
        engine.run()
    assert not engine.busy


def distributed_worker(rank, rendezvous):
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2,
                            timeout=timedelta(seconds=30))
    try:
        config = FakeConfig(ranks=2)
        engine = Engine(config)
        # Rank zero receives every output even when it owns no model blocks.
        candidate = Candidate("one", StageEqual(1), (RankDemand(1, 0.1),), 1.0)
        plan = PoolPlan(tuple(StreamPlacement(str(index), candidate, (1,)) for index in range(2)),
                        {}, (RankDemand(0, 0.0), RankDemand(2, 0.2)))
        engine.prepare(requests(), plan)
        results = engine.run()
        if rank == 0:
            assert len(results) == 2
            for request, result in zip(requests(), results):
                reference = FakeDiT(FakeConfig())
                from tests.reference import run
                condition = {"text": torch.tensor([sum(map(ord, request.prompt))], dtype=torch.float32)}
                expected = run(reference, Sampler.grid(), condition, request.shape, request.chunks, seed=request.seed)
                torch.testing.assert_close(result["latents"], expected, rtol=0, atol=0)
            assert all(not dit.loaded for dit in config.loaded)
        else:
            assert results is None
        engine.run()
    finally:
        dist.destroy_process_group()


def test_distributed_output_rank_without_compute_blocks(tmp_path):
    context = mp.spawn(distributed_worker, args=(str(tmp_path / "rendezvous"),), nprocs=2, join=False)
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
