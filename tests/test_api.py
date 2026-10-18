import json

import pytest
import torch

from physisserve import PhysisServe, SamplingParams
from physisserve.config import Config
from physisserve.sampling_params import Request
from tests.test_engine import FakeConfig


@pytest.mark.parametrize("name,channels,blocks,backend,shape,chunks,fps", [
    ("WanTransformer3DModel", 16, 30, "wan", (1, 16, 1, 60, 104), 19, 16),
    ("WanTransformer3DModel", 16, 40, "wan", (1, 16, 1, 60, 104), 19, 16),
    ("WanTransformer3DModel", 48, 30, "wan", (1, 48, 1, 44, 80), 29, 24),
    ("LTX2VideoTransformer3DModel", 128, 48, "ltx2", (1, 128, 1, 17, 30), 14, 24),
    ("MiniMaxH3Transformer3DModel", 24, 50, "h3", (1, 24, 1, 48, 84), 48, 24),
])
def test_standard_diffusers_profiles(tmp_path, name, channels, blocks, backend, shape, chunks, fps):
    for component in ("transformer", "vae", "text_encoder", "tokenizer"):
        (tmp_path / component).mkdir()
    (tmp_path / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", name]}))
    (tmp_path / "transformer" / "config.json").write_text(json.dumps({
        "_class_name": name, "in_channels": channels, "num_layers": blocks,
        "patch_size": 1 if backend == "ltx2" else [1, 2, 2],
    }))
    config = Config(str(tmp_path))
    request = SamplingParams().request("cat", config)
    assert config.backend == backend and config.num_blocks == blocks
    assert request.shape == shape and request.chunks == chunks and request.fps == fps
    assert request.mode == "deferred_decode" and request.latent_dir is None
    assert len(config.make_sampler().sigmas) == 5 and config.make_sampler().sigmas[-1] == 0
    if backend == "ltx2":
        assert config.dtype == "float32" and config.compute_dtype == "bfloat16"
        assert config.sigmas[1:4] == pytest.approx((0.8920687437, 0.6825016737, 0.1000000238))
    elif backend == "h3":
        assert config.timesteps == pytest.approx(tuple(1 - value for value in config.sigmas[:-1]))
    elif channels == 16:
        assert config.timesteps == ((1000, 750, 500, 250) if blocks == 30 else (700, 500, 400, 200))
    with pytest.raises(ValueError, match="spatial patch"):
        SamplingParams(width=831).request("cat", config)
    assert Config(str(tmp_path), strategy="stage_equal").max_batch == 1


def fake_config(model, *, ranks=1, strategy="virtual_rank", **kwargs):
    config = FakeConfig(ranks=ranks, strategy=strategy)
    config.backend, config.channels, config.height, config.width = "wan", 2, 2, 2
    config.spatial_scale, config.patch_size, config.chunks, config.fps = 1, (1, 1, 1), 5, 16
    return config


@pytest.mark.parametrize("ranks", [1, 2])
def test_public_generate_unifies_single_multiple_and_deferred_video(monkeypatch, ranks):
    monkeypatch.setattr("physisserve.serve.Config", fake_config)
    with PhysisServe("fake", device="cpu", pipeline_parallel_size=ranks) as model:
        params = SamplingParams(num_chunks=5, seed=17, mode="dit_only")
        single = model.generate("cat", params)
        assert len(single) == 1 and "video" not in single[0]
        multiple = model.generate(["cat", "dog"], params)
        assert [output["prompt"] for output in multiple] == ["cat", "dog"]
        torch.testing.assert_close(single[0]["latents"], multiple[0]["latents"], rtol=0, atol=0)
        decoded = model.decode(multiple)
        assert [output["video"].shape for output in decoded] == [(13, 2, 2, 3)] * 2
        assert all(output["video"].dtype == torch.uint8 for output in decoded)
        again = model.generate("dog", SamplingParams(num_chunks=5, seed=17))
        torch.testing.assert_close(again[0]["video"], decoded[1]["video"], rtol=0, atol=0)
        assert model.generate([], params) == []
        with pytest.raises(ValueError, match="one per prompt"):
            model.generate(["cat", "dog"], [params])
    assert not model._processes and model._closed
    with pytest.raises(RuntimeError, match="closed"):
        model.generate("cat", params)


def fail_worker(engine):
    if engine.rank == 1:
        raise ValueError("test worker failure")
    return None


def test_worker_failure_is_reported_and_processes_are_reaped(monkeypatch):
    monkeypatch.setattr("physisserve.serve.Config", fake_config)
    with PhysisServe("fake", device="cpu", pipeline_parallel_size=2) as model:
        processes = list(model._processes)
        with pytest.raises(RuntimeError, match="test worker failure"):
            model._call(fail_worker)
        assert model._closed and not any(process.is_alive() for process in processes)


@pytest.mark.parametrize("kwargs", [
    {"num_chunks": 0}, {"height": -1}, {"seed": -1}, {"fps": True},
    {"mode": "video"}, {"mode": None}, {"mode": []},
    {"latent_dir": ""}, {"latent_dir": "  "}, {"latent_dir": 3},
])
def test_invalid_sampling_inputs(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


@pytest.mark.parametrize("mode", ["dit_only", "deferred_decode", "stream_decode"])
def test_sampling_mode_and_latent_directory_reach_request(tmp_path, mode):
    target = tmp_path / "latents"
    params = SamplingParams(mode=mode, latent_dir=str(target))
    request = params.request("cat", fake_config("fake"))
    assert request.mode == mode and request.latent_dir == target
    assert not target.exists()
    assert Request("cat", (1, 2, 1, 2, 2), 5, 0, 16).mode == "dit_only"


def test_latent_directory_rejects_file(tmp_path):
    path = tmp_path / "file"
    path.write_text("data")
    with pytest.raises(ValueError, match="directory"):
        SamplingParams(latent_dir=path)


def test_mixed_modes_decode_after_all_streams_and_preserve_order_and_rejections():
    model = PhysisServe.__new__(PhysisServe)
    model.config = fake_config("fake")
    events = []
    outputs = [{"prompt": prompt, "latents": index}
               for index, prompt in enumerate(("raw", "deferred", "stream", "rejected", "deferred2"))]
    outputs[2]["video"] = "streamed video"
    outputs[3] = {"prompt": "rejected", "status": "rejected", "reason": "capacity"}

    def call(operation, *args, **kwargs):
        events.append(operation)
        if operation == "prepare":
            assert [request.mode for request in args[0]] == [
                "dit_only", "deferred_decode", "stream_decode", "deferred_decode", "deferred_decode"]
            return [None]
        if operation == "run":
            return [outputs]
        assert operation == "decode" and events == ["prepare", "run", "decode"]
        assert [output["prompt"] for output in args[0]] == ["deferred", "deferred2"]
        return [[{**output, "video": "decoded video"} for output in args[0]]]

    model._call = call
    result = model.generate([output["prompt"] for output in outputs], [
        SamplingParams(mode=mode) for mode in
        ("dit_only", "deferred_decode", "stream_decode", "deferred_decode", "deferred_decode")])
    assert events == ["prepare", "run", "decode"]
    assert [output["prompt"] for output in result] == ["raw", "deferred", "stream", "rejected", "deferred2"]
    assert "video" not in result[0] and "video" not in result[3]
    assert result[1]["video"] == result[4]["video"] == "decoded video"
    assert result[2]["video"] == "streamed video"
    assert result[3] == {"prompt": "rejected", "status": "rejected", "reason": "capacity"}


@pytest.mark.parametrize("backend", ["h3", "ltx2"])
def test_whole_sequence_vae_rejects_stream_mode_before_worker_call(backend):
    model = PhysisServe.__new__(PhysisServe)
    model.config = fake_config("fake")
    model.config.backend = backend
    model._call = lambda *args, **kwargs: pytest.fail("unsupported mode must fail before worker calls")
    with pytest.raises(ValueError, match="whole-sequence VAE.*stream_decode"):
        model.generate(["cat", "dog"], [SamplingParams(), SamplingParams(mode="stream_decode")])


def test_h3_short_chunks_allowed_only_for_dit_mode():
    model = PhysisServe.__new__(PhysisServe)
    model.config = fake_config("fake")
    model.config.backend = "h3"
    calls = []

    def call(operation, *args, **kwargs):
        calls.append(operation)
        return [[{"prompt": "cat", "latents": "raw"}]] if operation == "run" else [None]

    model._call = call
    assert model.generate("cat", SamplingParams(num_chunks=1, mode="dit_only"))[0]["latents"] == "raw"
    assert calls == ["prepare", "run"]
    calls.clear()
    with pytest.raises(ValueError, match="at least 3 chunks"):
        model.generate("cat", SamplingParams(num_chunks=1))
    assert calls == []


def test_request_rejects_batched_or_multi_latent_input():
    with pytest.raises(ValueError, match="shape"):
        Request("cat", (1, 16, 3, 60, 104), 5, 0, 16)
    with pytest.raises(ValueError, match="shape"):
        Request("cat", (2, 16, 1, 60, 104), 5, 0, 16)
    with pytest.raises(ValueError, match="mode"):
        Request("cat", (1, 16, 1, 60, 104), 5, 0, 16, mode="video")


def test_complete_diffusers_model_generates_decodable_video(tmp_path):
    from diffusers import AutoencoderKLWan, WanTransformer3DModel
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast, UMT5Config, UMT5EncoderModel
    import imageio.v3 as iio

    torch.set_num_threads(1)
    torch.manual_seed(11)
    WanTransformer3DModel(
        num_attention_heads=2, attention_head_dim=8, in_channels=16, out_channels=16,
        text_dim=8, freq_dim=8, ffn_dim=32, num_layers=30,
    ).save_pretrained(tmp_path / "transformer")
    AutoencoderKLWan(
        base_dim=8, z_dim=16, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
        latents_mean=[0.0] * 16, latents_std=[1.0] * 16,
    ).save_pretrained(tmp_path / "vae")
    UMT5EncoderModel(UMT5Config(vocab_size=8, d_model=8, d_kv=4, d_ff=16,
                              num_layers=1, num_heads=2)).save_pretrained(tmp_path / "text_encoder")
    tokenizer = Tokenizer(models.WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "cat": 3}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="<pad>", eos_token="</s>",
                           unk_token="<unk>").save_pretrained(tmp_path / "tokenizer")
    (tmp_path / "model_index.json").write_text(json.dumps({
        "_class_name": "WanPipeline", "transformer": ["diffusers", "WanTransformer3DModel"],
        "vae": ["diffusers", "AutoencoderKLWan"],
        "text_encoder": ["transformers", "UMT5EncoderModel"],
        "tokenizer": ["transformers", "PreTrainedTokenizerFast"],
    }))
    with PhysisServe(tmp_path, device="cpu") as model:
        output = model.generate("cat", SamplingParams(num_chunks=3, height=16, width=16))[0]
    assert output["video"].shape == (17, 16, 16, 3)
    assert output["latents"].shape == (1, 16, 5, 2, 2)
    path = tmp_path / "video.mp4"
    iio.imwrite(path, output["video"].numpy(), fps=output["fps"], codec="libx264", pixelformat="yuv420p",
                macro_block_size=None, output_params=["-movflags", "+faststart"])
    assert iio.imread(path).shape == (17, 16, 16, 3)
