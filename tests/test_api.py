import importlib
import json
import pickle

import pytest
import torch

from physisserve import PhysisServe, SamplingParams
from physisserve.config import Config
from physisserve.sampling_params import Request
from tests.test_engine import FakeConfig


@pytest.mark.parametrize("name,channels,blocks,backend,shape,dimensions,chunks,fps,dtype,compute_dtype,sigmas,timesteps", [
    ("WanTransformer3DModel", 16, 30, "wan", (1, 16, 1, 60, 104), (480, 832, 8), 19, 16,
     "bfloat16", None,
     (1.0, 0.750257670879364, 0.499718636274338, 0.250000029802322, 0.0),
     (1000.0, 750.0, 500.0, 250.0)),
    ("WanTransformer3DModel", 16, 40, "wan", (1, 16, 1, 60, 104), (480, 832, 8), 19, 16,
     "bfloat16", None,
     (0.700232386589050, 0.499718636274338, 0.400259941816330, 0.198347106575966, 0.0),
     (700.0, 500.0, 400.0, 200.0)),
    ("WanTransformer3DModel", 48, 30, "wan", (1, 48, 1, 44, 80), (704, 1280, 16), 29, 24,
     "bfloat16", None,
     (1.0, 0.9097071290016174, 0.7173174619674683, 0.0244140625, 0.0),
     (1000.0, 909.7071290016174, 717.3174619674683, 24.4140625)),
    ("LTX2VideoTransformer3DModel", 128, 48, "ltx2", (1, 128, 1, 17, 30), (544, 960, 32), 14, 24,
     "float32", "bfloat16",
     (1.0, 0.8920687437057495, 0.6825016736984253, 0.10000002384185791, 0.0),
     (1000.0, 892.0687437057495, 682.5016736984253, 100.00002384185791)),
    ("MiniMaxH3Transformer3DModel", 24, 50, "h3", (1, 24, 1, 48, 84), (768, 1344, 16), 48, 24,
     "bfloat16", None,
     (1.0, 0.9097071290016174, 0.7173174619674683, 0.0244140625, 0.0),
     (0.0, 0.09029287099838257, 0.28268253803253174, 0.9755859375)),
])
def test_standard_diffusers_profiles(tmp_path, name, channels, blocks, backend, shape, dimensions,
                                    chunks, fps, dtype, compute_dtype, sigmas, timesteps):
    for component in ("transformer", "vae", "text_encoder", "tokenizer"):
        (tmp_path / component).mkdir()
    (tmp_path / "model_index.json").write_text(json.dumps({"transformer": ["diffusers", name]}))
    (tmp_path / "transformer" / "config.json").write_text(json.dumps({
        "_class_name": name, "in_channels": channels, "num_layers": blocks,
        "patch_size": 1 if backend == "ltx2" else [1, 2, 2],
    }))
    config = Config(str(tmp_path))
    assert vars(pickle.loads(pickle.dumps(config))) == vars(config)
    request = SamplingParams().request("cat", config)
    assert (config.backend, config.num_blocks, config.channels) == (backend, blocks, channels)
    assert (config.height, config.width, config.spatial_scale) == dimensions
    assert (config.chunks, config.fps) == (chunks, fps)
    assert (config.dtype, config.compute_dtype) == (dtype, compute_dtype)
    assert config.supports_online_decode == (backend == "wan")
    assert config.min_decode_chunks == (3 if backend == "h3" else 1)
    assert config.patch_size == ((1, 1, 1) if backend == "ltx2" else (1, 2, 2))
    assert config.sigmas == sigmas and config.timesteps == timesteps
    sampler = config.make_sampler()
    assert sampler.sigmas == sigmas and sampler.timesteps == timesteps
    assert request.shape == shape and request.chunks == chunks and request.fps == fps
    assert request.mode == "offline_decode" and request.latent_dir is None
    with pytest.raises(ValueError, match="spatial patch"):
        SamplingParams(width=831).request("cat", config)
    assert Config(str(tmp_path), strategy="stage_equal").max_batch == 1


@pytest.mark.parametrize("family,dit_class,options,text_length", [
    ("wan", "WanDiT", {"cache_text_kv": True}, 512),
    ("ltx2", "LTX2DiT", {"compute_dtype": torch.bfloat16}, 1024),
    ("h3", "H3DiT", {}, None),
])
def test_model_factories_own_component_defaults(monkeypatch, tmp_path, family, dit_class, options, text_length):
    (tmp_path / "vae").mkdir()
    (tmp_path / "vae" / "config.json").write_text("{}")
    module = importlib.import_module(f"physisserve.models.{family}")
    monkeypatch.setattr(module, dit_class, lambda source, **kwargs: (source, kwargs))
    backend = module.backend(tmp_path)
    assert backend.dit == (tmp_path / "transformer", options)
    assert backend.encoder.max_length == text_length


def fake_config(model, *, ranks=1, strategy="virtual_rank", **kwargs):
    config = FakeConfig(ranks=ranks, strategy=strategy)
    config.backend, config.channels, config.height, config.width = "wan", 2, 2, 2
    config.spatial_scale, config.patch_size, config.chunks, config.fps = 1, (1, 1, 1), 5, 16
    config.supports_online_decode, config.min_decode_chunks = True, 1
    return config


def test_example_direct_api_writes_videos(monkeypatch, tmp_path):
    import example
    import imageio.v3 as iio

    monkeypatch.setattr("physisserve.serve.Config", fake_config)
    services, modes = [], []

    def make_service(model, **kwargs):
        service = PhysisServe(model, device="cpu", **kwargs)
        services.append(service)
        requests = service._requests

        def record_requests(prompts, params):
            result = requests(prompts, params)
            modes.extend(request.mode for request in result)
            return result

        monkeypatch.setattr(service, "_requests", record_requests)
        return service

    monkeypatch.setattr(example, "PhysisServe", make_service)
    try:
        example.main()
        assert len(services) == 1 and not services[0]._closed
        assert modes == ["offline_decode", "offline_decode"]
        for index in range(2):
            assert iio.imread(tmp_path / f"video-{index}.mp4").shape == (13, 2, 2, 3)
        assert len(list((tmp_path / "latents").glob("generate-*/*.safetensors"))) == 2
    finally:
        for service in services:
            service.close()


@pytest.mark.parametrize("ranks", [1, 2])
def test_public_generate_unifies_single_multiple_and_offline_video(monkeypatch, ranks):
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
    {"mode": "video"}, {"mode": "stream_decode"}, {"mode": "deferred_decode"},
    {"mode": None}, {"mode": []},
    {"latent_dir": ""}, {"latent_dir": "  "}, {"latent_dir": 3},
])
def test_invalid_sampling_inputs(kwargs):
    with pytest.raises(ValueError):
        SamplingParams(**kwargs)


@pytest.mark.parametrize("mode", ["dit_only", "offline_decode", "online_decode"])
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
               for index, prompt in enumerate(("raw", "offline", "online", "rejected", "offline2"))]
    outputs[2]["video"] = "online video"
    outputs[3] = {"prompt": "rejected", "status": "rejected", "reason": "capacity"}

    def call(operation, *args, **kwargs):
        events.append(operation)
        if operation == "prepare":
            assert [request.mode for request in args[0]] == [
                "dit_only", "offline_decode", "online_decode", "offline_decode", "offline_decode"]
            return [None]
        if operation == "run":
            return [outputs]
        assert operation == "decode" and events == ["prepare", "run", "decode"]
        assert [output["prompt"] for output in args[0]] == ["offline", "offline2"]
        return [[{**output, "video": "decoded video"} for output in args[0]]]

    model._call = call
    result = model.generate([output["prompt"] for output in outputs], [
        SamplingParams(mode=mode) for mode in
        ("dit_only", "offline_decode", "online_decode", "offline_decode", "offline_decode")])
    assert events == ["prepare", "run", "decode"]
    assert [output["prompt"] for output in result] == ["raw", "offline", "online", "rejected", "offline2"]
    assert "video" not in result[0] and "video" not in result[3]
    assert result[1]["video"] == result[4]["video"] == "decoded video"
    assert result[2]["video"] == "online video"
    assert result[3] == {"prompt": "rejected", "status": "rejected", "reason": "capacity"}


@pytest.mark.parametrize("backend", ["h3", "ltx2"])
def test_whole_sequence_vae_rejects_online_mode_before_worker_call(backend):
    model = PhysisServe.__new__(PhysisServe)
    model.config = fake_config("fake")
    model.config.backend = backend
    model.config.supports_online_decode = False
    model._call = lambda *args, **kwargs: pytest.fail("unsupported mode must fail before worker calls")
    with pytest.raises(ValueError, match="whole-sequence VAE.*online_decode"):
        model.generate(["cat", "dog"], [SamplingParams(), SamplingParams(mode="online_decode")])


def test_h3_short_chunks_allowed_only_for_dit_mode():
    model = PhysisServe.__new__(PhysisServe)
    model.config = fake_config("fake")
    model.config.backend = "h3"
    model.config.supports_online_decode, model.config.min_decode_chunks = False, 3
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


@pytest.mark.parametrize("prompt", [None, 42, ["nested"]])
def test_invalid_prompt_rejected_before_worker_call(prompt):
    model = PhysisServe.__new__(PhysisServe)
    model.config = fake_config("fake")
    model._call = lambda *args, **kwargs: pytest.fail("invalid input must fail before worker calls")
    with pytest.raises(TypeError, match="prompt.*string"):
        model.generate([prompt])


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
