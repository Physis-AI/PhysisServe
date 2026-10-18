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
        params = SamplingParams(num_chunks=5, seed=17)
        single = model.generate("cat", params, output_type="latent")
        assert len(single) == 1 and "video" not in single[0]
        multiple = model.generate(["cat", "dog"], params, output_type="latent")
        assert [output["prompt"] for output in multiple] == ["cat", "dog"]
        torch.testing.assert_close(single[0]["latents"], multiple[0]["latents"], rtol=0, atol=0)
        decoded = model.decode(multiple)
        assert [output["video"].shape for output in decoded] == [(13, 2, 2, 3)] * 2
        assert all(output["video"].dtype == torch.uint8 for output in decoded)
        again = model.generate("dog", params)
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


@pytest.mark.parametrize("kwargs", [{"num_chunks": 0}, {"height": -1}, {"seed": -1}, {"fps": True}])
def test_invalid_sampling_inputs(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


def test_request_rejects_batched_or_multi_latent_input():
    with pytest.raises(ValueError, match="shape"):
        Request("cat", (1, 16, 3, 60, 104), 5, 0, 16)
    with pytest.raises(ValueError, match="shape"):
        Request("cat", (2, 16, 1, 60, 104), 5, 0, 16)


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
