from dataclasses import replace
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from physisserve import PhysisServe, SamplingParams
from physisserve.engine.engine import Engine
from physisserve.models.wan import WanDecoder
from tests.test_api import fake_config
from tests.test_engine import FakeConfig, FakeDecoder, FakeDiT


@pytest.mark.parametrize("mode", ["dit_only", "offline_decode", "online_decode"])
def test_decode_happens_at_the_selected_time_and_preserves_latents(monkeypatch, tmp_path, mode):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("physisserve.serve.Config", fake_config)
    events = []
    unembed, decode = FakeDiT.unembed, FakeDecoder.decode

    def record_dit(self, hidden, timestep, chunk):
        events.append(("dit", chunk))
        return unembed(self, hidden, timestep, chunk)

    def record_decode(self, latent, chunk):
        events.append(("vae", chunk))
        return decode(self, latent, chunk)

    monkeypatch.setattr(FakeDiT, "unembed", record_dit)
    monkeypatch.setattr(FakeDecoder, "decode", record_decode)
    with PhysisServe("fake", device="cpu") as model:
        params = SamplingParams(num_chunks=8, seed=17, mode="dit_only")
        raw = model.generate("cat", params)[0]
        assert "latent_path" not in raw and not (tmp_path / "latents").exists()
        events.clear()
        output = model.generate("cat", replace(params, mode=mode))[0]
        torch.testing.assert_close(output["latents"], raw["latents"], rtol=0, atol=0)
        assert ("latent_path" in output) == (mode == "offline_decode")
        assert (tmp_path / "latents").exists() == (mode == "offline_decode")
        if mode == "dit_only":
            assert all(kind == "dit" for kind, _ in events)
            assert "video" not in output and model.config.decoder_loads == 0
        else:
            first_decode = next(index for index, event in enumerate(events) if event[0] == "vae")
            last_dit = max(index for index, event in enumerate(events) if event[0] == "dit")
            assert (first_decode > last_dit) == (mode == "offline_decode")
            assert [chunk for kind, chunk in events if kind == "vae"] == list(range(8))
            expected = model.decode([raw])[0]["video"]
            assert torch.equal(output["video"], expected)
            assert not torch.is_inference(output["video"])


@pytest.mark.parametrize("explicit_directory", [False, True], ids=["default", "explicit"])
def test_latents_are_saved_before_offline_decode_without_overwriting(monkeypatch, tmp_path, explicit_directory):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("physisserve.serve.Config", fake_config)
    root = tmp_path / ("saved" if explicit_directory else "latents")
    load = FakeDecoder.load
    saved = []

    def check_saved(self, device, dtype):
        paths = sorted(root.glob("generate-*/stream-*.safetensors"))
        assert len(paths) == len(saved) + 2  # both streams persisted before the VAE loads
        saved[:] = paths
        load(self, device, dtype)

    monkeypatch.setattr(FakeDecoder, "load", check_saved)
    with PhysisServe("fake", device="cpu") as model:
        params = SamplingParams(num_chunks=4, mode="offline_decode", latent_dir=root if explicit_directory else None)
        first = model.generate(["cat", "dog"], params)
        again = model.generate(["cat", "dog"], params)
    assert len(saved) == 4
    first_directory = {Path(output["latent_path"]).parent for output in first}
    again_directory = {Path(output["latent_path"]).parent for output in again}
    assert len(first_directory) == len(again_directory) == 1
    assert first_directory.isdisjoint(again_directory)
    assert len(list(root.iterdir())) == 2
    for directory in first_directory | again_directory:
        assert directory.name.startswith("generate-")
        assert {path.name for path in directory.iterdir()} == {"stream-0000.safetensors", "stream-0001.safetensors"}
    assert {output["latent_path"] for output in first}.isdisjoint(output["latent_path"] for output in again)
    for output in first + again:
        path = Path(output["latent_path"])
        assert path.is_absolute() and path.parent.parent == root
        torch.testing.assert_close(load_file(path)["latents"], output["latents"], rtol=0, atol=0)
        with safe_open(path, framework="pt") as file:
            assert file.metadata()["prompt"] == output["prompt"]
            assert file.metadata()["first_frames"] == "3"


def test_explicit_latent_directories_save_all_modes_and_group_by_resolved_root(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("physisserve.serve.Config", fake_config)
    first_root, second_root = tmp_path / "first", tmp_path / "second"
    params = [SamplingParams(num_chunks=4, mode="dit_only", latent_dir=first_root),
              SamplingParams(num_chunks=4, mode="online_decode", latent_dir=Path("first")),
              SamplingParams(num_chunks=4, mode="offline_decode", latent_dir=second_root)]
    with PhysisServe("fake", device="cpu") as model:
        outputs = model.generate(["cat", "dog", "beach"], params)
    paths = [Path(output["latent_path"]) for output in outputs]
    assert paths[0].parent == paths[1].parent
    assert paths[0].parent.parent == first_root
    assert paths[2].parent.parent == second_root
    assert paths[0].parent != paths[2].parent
    assert not (tmp_path / "latents").exists()
    for index, (path, output) in enumerate(zip(paths, outputs)):
        assert path.name == f"stream-{index:04d}.safetensors"
        torch.testing.assert_close(load_file(path)["latents"], output["latents"], rtol=0, atol=0)


@pytest.mark.parametrize("strategy", ["stage_equal", "virtual_rank"])
def test_spawned_mixed_modes_preserve_per_stream_decoder_state(monkeypatch, strategy):
    monkeypatch.setattr("physisserve.serve.Config", fake_config)
    prompts = ["cat", "dog", "beach"]
    params = [SamplingParams(num_chunks=6, seed=11, mode="online_decode"),
              SamplingParams(num_chunks=4, seed=12, mode="offline_decode"),
              SamplingParams(num_chunks=5, seed=13, mode="dit_only")]
    with PhysisServe("fake", device="cpu", pipeline_parallel_size=2, strategy=strategy) as model:
        outputs = model.generate(prompts, params)
        assert ["video" in output for output in outputs] == [True, True, False]
        more = model.generate(prompts[:2], [replace(param, mode="online_decode") for param in params[:2]])
    with PhysisServe("fake", device="cpu", strategy=strategy) as model:
        expected = [model.generate(prompt, replace(param, mode="offline_decode"))[0]
                    for prompt, param in zip(prompts, params)]
    for index, (output, reference) in enumerate(zip(outputs, expected)):
        torch.testing.assert_close(output["latents"], reference["latents"], rtol=0, atol=0)
        if index < 2:
            assert torch.equal(output["video"], reference["video"])
            assert torch.equal(more[index]["video"], reference["video"])


@pytest.mark.parametrize("residual", [False, True], ids=["wan21", "wan22"])
def test_online_matches_offline_with_real_wan_vae(monkeypatch, tmp_path, residual):
    from diffusers import AutoencoderKLWan

    torch.set_num_threads(1)
    torch.manual_seed(19)
    options = dict(is_residual=True, in_channels=12, out_channels=12, patch_size=2) if residual else {}
    AutoencoderKLWan(base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
                     latents_mean=[0.1, -0.2, 0.3, 0.0], latents_std=[1.5, 0.8, 1.2, 1.0],
                     **options).save_pretrained(tmp_path / "vae")
    config = FakeConfig()
    config.channels = 4
    backend = config.make_backend

    def with_real_vae():
        result = backend()
        result.decoder = WanDecoder(tmp_path)
        return result

    monkeypatch.setattr(config, "make_backend", with_real_vae)
    engine = Engine(config)
    prompts = ["cat", "dog"]
    params = [SamplingParams(num_chunks=4, height=2, width=2, seed=seed, mode="online_decode")
              for seed in range(len(prompts))]
    engine.prepare(prompts, params)
    online = engine.run()
    engine.prepare(prompts, [replace(param, mode="offline_decode", latent_dir=tmp_path) for param in params])
    latents = engine.run()
    assert all("video" not in output for output in latents)
    offline = engine.decode(latents)
    for actual, expected in zip(online, offline):
        torch.testing.assert_close(actual["latents"], expected["latents"], rtol=0, atol=0)
        torch.testing.assert_close(actual["video"], expected["video"], rtol=0, atol=0)
