import json

import pytest
import torch

from physisserve.engine.kv_cache import Cache
from physisserve.models.base import load_selected_weights
from physisserve.models.h3 import H3DiT, phase_aligned_frames
from physisserve.models.ltx2 import LTX2Decoder, LTX2DiT
from physisserve.models.wan import WanDecoder, WanDiT


def test_h3_window_preserves_sink_and_five_frame_phase():
    for chunk in range(128):
        pages = min(chunk, 18) + (chunk > 18)
        frames = phase_aligned_frames(pages, 1, chunk, sink_frames=1)
        assert len(frames) == pages + 1
        assert frames[0] == 0
        if chunk > 18:
            original = tuple(range(chunk - pages + 1, chunk + 1))
            assert all((actual - wanted) % 5 == 0 for actual, wanted in zip(frames[1:], original))
            assert all(right == left + 1 for left, right in zip(frames[1:], frames[2:]))
        else:
            assert frames == tuple(range(chunk + 1))


def test_selective_shards_load_only_owned_tensors(tmp_path):
    from safetensors.torch import save_file

    save_file({"owned": torch.tensor([3]), "other": torch.tensor([7])}, tmp_path / "part.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"owned": "part.safetensors", "other": "unopened.safetensors"},
    }))
    state = load_selected_weights(tmp_path, ("owned",))
    assert set(state) == {"owned"}
    assert state["owned"].item() == 3
    with pytest.raises(ValueError, match="missing tensors"):
        load_selected_weights(tmp_path, ("missing",))


@pytest.mark.parametrize("filename", ["../outside.safetensors", "/outside.safetensors"])
def test_selected_shard_paths_must_be_relative(tmp_path, filename):
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"owned": filename}}))
    with pytest.raises(ValueError, match="shard path must be relative"):
        load_selected_weights(tmp_path, ("owned",))


def test_unindexed_shards_reject_missing_or_duplicate_selected_tensors(tmp_path):
    from safetensors.torch import save_file

    save_file({"owned": torch.tensor([3])}, tmp_path / "part-1.safetensors")
    save_file({"other": torch.tensor([7])}, tmp_path / "part-2.safetensors")
    state = load_selected_weights(tmp_path, ("owned",))
    assert set(state) == {"owned"}
    assert state["owned"].item() == 3
    with pytest.raises(ValueError, match="missing tensors"):
        load_selected_weights(tmp_path, ("missing",))
    save_file({"owned": torch.tensor([4])}, tmp_path / "part-3.safetensors")
    with pytest.raises(ValueError, match="duplicate tensor"):
        load_selected_weights(tmp_path, ("owned",))


def test_indexed_shard_cannot_leave_selected_tensor_unloaded(tmp_path):
    from safetensors.torch import save_file

    save_file({"other": torch.tensor([7])}, tmp_path / "part.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"owned": "part.safetensors"},
    }))
    with pytest.raises(ValueError, match="missing tensors"):
        load_selected_weights(tmp_path, ("owned",))


def test_ltx_decoder_rejects_broadcastable_latent_channels():
    from types import SimpleNamespace

    decoder = LTX2Decoder.__new__(LTX2Decoder)
    decoder.mean = torch.zeros(1, 128, 1, 1, 1)
    decoder.std = torch.ones_like(decoder.mean)
    decoder.device, decoder.dtype = torch.device("cpu"), torch.float32
    decoder.vae = SimpleNamespace(
        config=SimpleNamespace(scaling_factor=1, timestep_conditioning=False),
        decode=lambda *args, **kwargs: pytest.fail("invalid channels must not reach the VAE"))
    with pytest.raises(ValueError, match="channels"):
        decoder.decode_sequence(torch.zeros(1, 1, 3, 2, 2))


def tiny_transformer(family):
    from diffusers import LTX2VideoTransformer3DModel, MiniMaxH3Transformer3DModel, WanTransformer3DModel

    if family.startswith("wan"):
        channels = 48 if family == "wan22_5b" else 16
        layers = 3 if family == "wan21_14b" else 2
        return WanTransformer3DModel(num_attention_heads=2, attention_head_dim=8, in_channels=channels,
                                     out_channels=channels, text_dim=8, freq_dim=8,
                                     ffn_dim=32, num_layers=layers), WanDiT, channels, 8
    if family == "ltx2":
        return LTX2VideoTransformer3DModel(
            in_channels=4, out_channels=4, num_attention_heads=2, attention_head_dim=8,
            cross_attention_dim=16, audio_in_channels=4, audio_out_channels=4, audio_num_attention_heads=2,
            audio_attention_head_dim=8, audio_cross_attention_dim=16, num_layers=2, caption_channels=8,
            vae_scale_factors=(8, 32, 32), rope_type="split", gated_attn=True, audio_gated_attn=True,
            cross_attn_mod=True, audio_cross_attn_mod=True, use_prompt_adaln_single=True,
            use_prompt_embeddings=False, perturbed_attn=True, ff_bias=False,
            use_keyframes_abs_pos_embedding=True), LTX2DiT, 4, 16
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=32, hidden_size=64, num_layers=2, num_refiner_layers=1,
        ffn_dim=128, in_channels=24, audio_in_channels=8, patch_size=(1, 2, 2), text_dim=32,
        freq_dim=16, time_embed_hidden_dim=64, time_embed_dim=32, rope_freq_dim=4), H3DiT, 24, 32


@pytest.mark.parametrize("family", ["wan21_14b", "ltx2", "h3"])
def test_selective_blocks_materialize_one_shot_ownership(tmp_path, family):
    model, adapter, _, _ = tiny_transformer(family)
    model.save_pretrained(tmp_path)
    dit = adapter(tmp_path)
    owned = tuple(range(0, dit.num_blocks, 2))
    dit.load((block for block in owned), "cpu", torch.float32)
    blocks = dit.model.blocks if family.startswith("wan") else dit.model.transformer_blocks
    assert tuple(blocks) == tuple(map(str, owned))
    original = model.state_dict()
    for name, tensor in dit.model.state_dict().items():
        torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


@pytest.mark.parametrize("family", ["wan21_1p3b", "ltx2", "h3"])
def test_unsupported_patch_is_rejected_when_opening_checkpoint(tmp_path, family):
    model, adapter, _, _ = tiny_transformer(family)
    model.register_to_config(patch_size=2 if family == "ltx2" else (2, 2, 2))
    model.save_config(tmp_path)
    with pytest.raises(ValueError, match="patch size 1"):
        adapter(tmp_path)


@pytest.mark.parametrize("family", ["wan21_1p3b", "wan21_14b", "wan22_5b", "ltx2", "h3"])
def test_native_batch_preserves_independent_history_and_conditioning(tmp_path, family):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model, adapter, channels, text_dim = tiny_transformer(family)
    model.save_pretrained(tmp_path)
    dit = adapter(tmp_path)
    dit.load(range(dit.num_blocks), "cpu", torch.float32)
    shape = (1, channels, 1, 4, 4)
    dit.prepare({"text": torch.randn(1, 5, text_dim)}, shape, 32, fps=24)
    chunks = [22, 27]
    times = [torch.tensor(value) for value in ((0.25, 0.75) if family == "h3" else (750.0, 250.0))]
    histories = []
    for chunk in chunks:
        pages = []
        for past in (0, chunk - 1):
            past_shape = (1, channels, dit.first_frames if past == 0 else 1, 4, 4)
            pages.append(tuple(torch.randn(dit.kv_shape(past_shape)) for _ in range(2)))
        histories.append(pages)
    hidden = [torch.randn(dit.hidden_shape(shape)) for _ in chunks]
    serial_caches = [Cache(history) for history in histories]
    batch_caches = [Cache(history) for history in histories]
    history_copies = [[tuple(value.clone() for value in page) for page in history] for history in histories]
    with torch.inference_mode():
        expected = [dit.block(1, value, timestep, chunk, cache)
                    for value, timestep, chunk, cache in zip(hidden, times, chunks, serial_caches)]
        actual = dit.block_batch(1, hidden, times, chunks, batch_caches)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    for batch, serial in zip(batch_caches, serial_caches):
        torch.testing.assert_close(batch.page, serial.page, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(histories, history_copies, rtol=0, atol=0)


@pytest.mark.parametrize("family", ["wan21_1p3b", "wan22_5b", "ltx2", "h3"])
def test_output_only_rank_can_prepare_metadata_without_transformer_blocks(tmp_path, family):
    model, adapter, channels, text_dim = tiny_transformer(family)
    model.save_pretrained(tmp_path)
    dit = adapter(tmp_path)
    dit.load((), "cpu", torch.float32)
    shape = (1, channels, 1, 4, 4)
    dit.prepare({"text": torch.randn(1, 5, text_dim)}, shape, 4, fps=24)
    dit.prepare_timesteps([torch.tensor(0.5 if family == "h3" else 500.0)])
    assert all(size > 0 for size in dit.hidden_shape(shape))
    assert all(size > 0 for size in dit.kv_shape(shape))
    assert not any(parameter.is_meta for parameter in dit.model.parameters())


@pytest.mark.parametrize("residual", [False, True], ids=["wan21", "wan22"])
def test_wan_vae_three_latent_head_and_one_latent_tail_match_full_decode(tmp_path, residual):
    from diffusers import AutoencoderKLWan

    torch.set_num_threads(1)
    options = dict(is_residual=True, in_channels=12, out_channels=12, patch_size=2) if residual else {}
    vae = AutoencoderKLWan(
        base_dim=8, decoder_base_dim=8, z_dim=4, dim_mult=[1, 1, 1, 1], num_res_blocks=1,
        latents_mean=[0.1, -0.2, 0.3, 0.0], latents_std=[1.5, 0.8, 1.2, 1.0], **options).eval()
    vae.save_pretrained(tmp_path / "vae")
    latent = torch.randn(1, 4, 5, 2, 2)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    decoder = WanDecoder(tmp_path)
    decoder.load("cpu", torch.float32)
    decoder.reset()
    with torch.inference_mode():
        expected = vae.decode(latent * std + mean).sample.clamp(-1, 1)
        chunks = [decoder.decode(value, index)
                  for index, value in enumerate((latent[:, :, :3], latent[:, :, 3:4], latent[:, :, 4:5]))]
    assert [value.shape[2] for value in chunks] == [9, 4, 4]
    torch.testing.assert_close(torch.cat(chunks, dim=2), expected, rtol=0, atol=0)
