"""MiniMax-H3 video-only chunk-AR backend.

The packaged checkpoint uses H3's video stream only. Text is carried in
the hidden packet so that the existing DiT contract needs no separate prefix-KV
API.  Text and video queries both see the complete text, cached-history and
current-chunk document.  Only video K/V is published to the engine cache.

H3 consumes timesteps in its native convention (``t = 1 - sigma`` in [0, 1])
and predicts a data-ward velocity.  ``unembed`` flips that prediction into the
generic engine convention, ``v = eps - x0``.
"""
from collections import OrderedDict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .base import Backend, block_indices, drop_attributes, load_selected_weights
from .batching import BatchCache, validate_batch
from .rope import select_rope, validate_rope_backend, validate_rope_device


def backend(root, text_length=None, rope_backend="native") -> Backend:
    """Build the video-only H3 components from an official Diffusers layout."""
    root = Path(root)
    return Backend(
        H3DiT(root / "transformer", rope_backend=rope_backend),
        H3Encoder(root, max_length=text_length),
        H3LatentDecoder(root),
    )


def _patchify(latent, patch_size):
    """[B,C,T,H,W] -> frame-major [B,patches,C*pt*ph*pw]."""
    pt, ph, pw = patch_size
    b, c, t, h, w = latent.shape
    rows = latent.reshape(b, c, t // pt, pt, h // ph, ph, w // pw, pw)
    return rows.permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(b, -1, c * pt * ph * pw)


def _unpatchify(rows, shape, patch_size):
    """Inverse of :func:`_patchify`."""
    b, c, t, h, w = shape
    pt, ph, pw = patch_size
    rows = rows.reshape(b, t // pt, h // ph, w // pw, c, pt, ph, pw)
    return rows.permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(b, c, t, h, w)


def phase_aligned_frames(read_frames, query_frames, window_start, sink_frames):
    """Effective frames of ``[sink | history | query]`` in the checkpoint's phase-aligned layout.

    While the read set is contiguous from frame 0 it keeps its absolute frames. Once the history detaches from
    the sink, the sink keeps ``0 .. S-1`` and the history and query move down by the largest multiple of five
    that keeps them after the sink, so every latent keeps its place in H3's five-latent VAE clock.
    """
    if read_frames >= window_start:
        return tuple(range(read_frames + query_frames))
    if not 0 <= sink_frames <= read_frames:
        raise ValueError(f"a detached read set must start with the {sink_frames}-frame sink, got {read_frames}")
    first = window_start - (read_frames - sink_frames)
    shift = 5 * ((first - sink_frames) // 5)
    return tuple(range(sink_frames)) + tuple(frame - shift for frame in range(first, window_start + query_frames))


def _video_positions(frames, height, width, patch_size, text_tokens, device):
    """Official H3 fp64 (time, height, width) coordinates for selected frames."""
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        _frame_position_grid,
        _temporal_position_grid,
    )

    frames = torch.as_tensor(frames, dtype=torch.long)
    if not frames.numel() or int(frames.min()) < 0:
        raise ValueError("H3 frame positions must be nonempty and nonnegative")
    _, ph, pw = patch_size
    spatial, _ = _frame_position_grid(height, width, ph, pw)
    clock = _temporal_position_grid(int(frames.max()) + 1, float(text_tokens)).index_select(0, frames)
    positions = torch.empty(frames.numel(), spatial.shape[0], 3, dtype=torch.float64)
    positions[..., 0] = clock[:, None]
    positions[..., 1:] = spatial[None]
    return positions.reshape(-1, 3).to(device)


class _Attention:
    """H3 attention with a permanent text prefix and pre-RoPE video cache.

    History keys are cached before RoPE. The sink stays at frame 0, while detached
    history is shifted by whole five-frame periods to preserve H3's clock phase.
    """

    def __init__(self, rope_backend="native"):
        self.rope_backend = validate_rope_backend(rope_backend)
        self.rope = select_rope(rope_backend, "h3")
        self.cache = self.key_rotary = None
        self.text_tokens = 0
        self.batch_shape = None

    def __call__(self, attn, hidden_states, rotary_emb=None, attention_mask=None):
        if attention_mask is not None:
            raise ValueError("H3 AR attention builds its prefix/window mask internally")
        if self.cache is None or self.key_rotary is None or rotary_emb is None:
            raise RuntimeError("H3 attention processor was called outside H3DiT.block")
        from diffusers.models.transformers.transformer_minimax_h3 import _apply_rotary_emb
        rope = self.rope or _apply_rotary_emb

        original_shape = hidden_states.shape
        if self.batch_shape is not None:
            # AdaLN addresses packed token rows, while attention must keep each
            # member as a separate document. Restore that batch before Q/K/V.
            hidden_states = hidden_states.reshape(*self.batch_shape, original_shape[-1])

        def rotate(tensor, tables):
            if self.batch_shape is None:
                return rope(tensor, *tables)
            # H3's native and Triton RoPE accept shared [tokens, rotary] tables.
            # Packing only this elementwise operation supports different tables
            # per member without mixing their attention documents.
            batch, tokens, heads, width = tensor.shape
            packed = tensor.reshape(1, batch * tokens, heads, width)
            return rope(packed, *tables).reshape(batch, tokens, heads, width)

        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))

        text = self.text_tokens
        query = rotate(query, rotary_emb).transpose(1, 2)
        text_key = rotate(key[:, :text], self.key_rotary[0]).transpose(1, 2)
        text_value = value[:, :text].transpose(1, 2)

        # The cache page is deliberately the unrotated current video K/V.
        window_key, window_value = self.cache(
            key[:, text:].transpose(1, 2), value[:, text:].transpose(1, 2)
        )
        window_key = rotate(window_key.transpose(1, 2), self.key_rotary[1]).transpose(1, 2)
        key = torch.cat((text_key, window_key), dim=2)
        value = torch.cat((text_value, window_value), dim=2)

        # Official H3 visibility: both text and video queries attend to the
        # complete [text | history | current] document.
        hidden = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        hidden = hidden.transpose(1, 2).reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        return attn.to_out[1](attn.to_out[0](hidden)).reshape(original_shape)


class H3DiT:
    """Selective-block MiniMax-H3 adapter for the packaged video-only checkpoint."""

    first_frames = 1
    sink_sigma = 0.001

    def __init__(self, source, rope_backend="native"):
        from diffusers import MiniMaxH3Transformer3DModel

        self.source = Path(source)
        self.rope_backend = validate_rope_backend(rope_backend)
        self.config = MiniMaxH3Transformer3DModel.load_config(self.source)
        self.config = {key: value for key, value in self.config.items() if not key.startswith("_")}
        if rope_backend == "triton" and self.config["attention_head_dim"] != 128:
            raise ValueError("Triton H3 RoPE requires attention_head_dim=128")
        if tuple(self.config["patch_size"])[0] != 1:
            raise ValueError("H3 chunk inference requires temporal patch size 1")
        self.num_blocks = self.config["num_layers"]
        self.model = None

    def load(self, blocks, device, dtype):
        from diffusers import MiniMaxH3Transformer3DModel
        from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed

        validate_rope_device(self.rope_backend, device, dtype)
        blocks = block_indices(blocks, self.num_blocks)
        with torch.device("meta"):
            model = MiniMaxH3Transformer3DModel.from_config(self.config)
        model.transformer_blocks = nn.ModuleDict({str(i): model.transformer_blocks[i] for i in blocks})
        drop_attributes(model, ("audio_proj_in", "audio_proj_out"))
        if 0 not in blocks:
            drop_attributes(model, ("proj_in", "context_embedder", "token_refiner"))
        if self.num_blocks - 1 not in blocks:
            drop_attributes(model, ("norm_out", "proj_out"))

        keep_fp32 = set(model._keep_in_fp32_modules)
        state = {}
        for name, tensor in load_selected_weights(self.source, model.state_dict()).items():
            target = torch.float32 if any(part in keep_fp32 for part in name.split(".")) else dtype
            state[name] = tensor.to(device=device, dtype=target)
        model.load_state_dict(state, assign=True)
        # ``rope.inv_freq`` is non-persistent and was constructed on meta.
        model.rope = MiniMaxH3RotaryPosEmbed(
            self.config["rope_freq_dim"], self.config["rope_theta"]
        ).to(device)
        for block in model.transformer_blocks.values():
            block.attn.set_processor(_Attention(self.rope_backend))
        self.model = model.eval().requires_grad_(False)
        self.device, self.dtype = torch.device(device), dtype

    def timestep_for_sigma(self, sigma):
        return 1.0 - float(sigma)

    def prepare(self, condition, shape, chunks, fps=None):
        self._batch_conditioning_cache = self._batch_rotary_cache = None
        del fps
        shape = tuple(shape)
        cfg, b, c, t, h, w = self.config, *shape
        pt, ph, pw = cfg["patch_size"]
        if c != cfg["in_channels"] or t % pt or h % ph or w % pw:
            raise ValueError("H3 requires checkpoint channels and patch-aligned T/H/W")
        text = condition.get("text")
        if text is None or text.ndim != 3 or text.shape[0] != b or text.shape[2] != cfg["text_dim"]:
            raise ValueError("condition['text'] must be [B, L, text_dim]")
        if text.shape[1] < 1:
            raise ValueError("H3 requires at least one text token")
        self.shape = tuple(shape)
        self.chunks = chunks
        self.text = text.to(self.device)
        self.text_tokens = text.shape[1]
        self.embeddings, self.rotary = {}, OrderedDict()
        # This prefix is independent of chunk and diffusion time. Only block-0
        # owners retain the projection/refiner modules after selective loading.
        self.refined_text = None
        if hasattr(self.model, "context_embedder"):
            self.refined_text = self.model.token_refiner(self.model.context_embedder(
                self.text.to(self.model.context_embedder.weight.dtype)))
        video_tokens = self._video_tokens(self.shape)
        self.timestep_indices = torch.zeros(
            self.text_tokens + video_tokens, device=self.device, dtype=torch.long
        )
        self.adaln_indices = torch.cat((
            torch.ones(self.text_tokens, device=self.device, dtype=torch.long),
            torch.zeros(video_tokens, device=self.device, dtype=torch.long),
        ))
        self.text_positions = torch.zeros(self.text_tokens, 3, dtype=torch.float64, device=self.device)
        self.text_positions[:, 0] = torch.arange(self.text_tokens, dtype=torch.float64, device=self.device)
        self.prefix_rotary = self.model.rope(self.text_positions)

    def prepare_timesteps(self, timesteps):
        for timestep in timesteps:
            self._embedding(timestep)

    def _video_tokens(self, shape):
        _, _, t, h, w = shape
        pt, ph, pw = self.config["patch_size"]
        return (t // pt) * (h // ph) * (w // pw)

    def hidden_shape(self, shape):
        return (shape[0], self.text_tokens + self._video_tokens(shape), self.config["hidden_size"])

    def kv_shape(self, shape):
        return (shape[0], self.config["num_attention_heads"], self._video_tokens(shape),
                self.config["attention_head_dim"])

    def _embedding(self, timestep):
        """H3-native timestep: t=0 is full noise and t=1 is clean."""
        key = float(timestep)
        if not 0.0 <= key <= 1.0:
            raise ValueError("H3 timestep must be t=1-sigma in [0, 1]")
        if key not in self.embeddings:
            # Official H3 row timesteps put the prompt at the current window's
            # (noisiest) timestep.  A q=1 Same-Step forward has one video
            # timestep, hence one shared timestep table row for text and video.
            values = torch.tensor((key,), device=self.device, dtype=torch.float32)
            temb = self.model.time_embedder(self.model.time_proj(values).to(
                self.model.time_embedder.linear_1.weight.dtype))
            self.embeddings[key] = temb, self.adaln_indices, self.timestep_indices
        return self.embeddings[key]

    def _rotary(self, chunk, pages):
        if not 0 <= pages <= chunk:
            raise ValueError("H3 cache has more history pages than preceding chunks")
        _, _, t, h, w = self.shape
        frames = phase_aligned_frames(pages * t, t, chunk * t, sink_frames=t)
        key = frames
        if key not in self.rotary:
            own_positions = _video_positions(
                torch.tensor(frames[-t:]), h, w, self.config["patch_size"], self.text_tokens, self.device
            )
            query = self.model.rope(torch.cat((self.text_positions, own_positions)))
            window = self.model.rope(_video_positions(
                torch.tensor(frames), h, w, self.config["patch_size"], self.text_tokens, self.device
            ))
            self.rotary[key] = (query, self.prefix_rotary, window)
            if len(self.rotary) > 5:
                self.rotary.popitem(last=False)
        self.rotary.move_to_end(key)
        return self.rotary[key]

    def _batch_conditioning(self, timesteps, batch_size):
        """One ordered, read-only preparation entry for the current request."""
        from diffusers.models.transformers.transformer_minimax_h3 import MINIMAX_H3_MODALITY_NUM

        key = tuple(float(timestep) for timestep in timesteps), batch_size
        enabled = not torch.is_grad_enabled()
        if enabled and self._batch_conditioning_cache is not None and self._batch_conditioning_cache[0] == key:
            return self._batch_conditioning_cache[1]
        self._batch_conditioning_cache = None
        temb = torch.cat([self._embedding(timestep)[0] for timestep in timesteps], dim=0)
        indices = torch.cat([(self.adaln_indices + member * MINIMAX_H3_MODALITY_NUM).repeat(batch_size)
                             for member in range(len(timesteps))])
        if enabled:
            self._batch_conditioning_cache = key, (temb, indices)
        return temb, indices

    def _batch_rotary(self, chunks, pages, batch_size):
        """Reuse packed tables without retaining hidden, KV or block-specific work."""
        key = tuple(chunks), pages, batch_size
        enabled = not torch.is_grad_enabled()
        if enabled and self._batch_rotary_cache is not None and self._batch_rotary_cache[0] == key:
            return self._batch_rotary_cache[1]
        self._batch_rotary_cache = None  # release the old entry before native preparation
        tables = [self._rotary(chunk, pages) for chunk in chunks]
        rotary = tuple(tuple(torch.cat([table[part][trig].repeat(batch_size, 1) for table in tables])
                             for trig in (0, 1)) for part in (0, 1, 2))
        if enabled:
            self._batch_rotary_cache = key, rotary
        return rotary

    def embed(self, latent, timestep, chunk):
        del timestep, chunk
        video = self.model.proj_in(_patchify(latent, self.config["patch_size"]).to(
            self.model.proj_in.weight.dtype))
        text = self.refined_text
        return torch.cat((text, video.to(text.dtype)), dim=1)

    def block(self, index, hidden, timestep, chunk, cache):
        temb, adaln_indices, _ = self._embedding(timestep)
        query, prefix, window = self._rotary(chunk, len(cache.history))
        block = self.model.transformer_blocks[str(index)]
        processor = block.attn.processor
        processor.cache, processor.key_rotary = cache, (prefix, window)
        processor.text_tokens = self.text_tokens
        try:
            return block(hidden, temb, adaln_indices, query)
        finally:
            processor.cache = processor.key_rotary = None
            processor.text_tokens = 0

    def block_batch(self, index, hidden_list, timestep_list, chunk_list, cache_list):
        """Run one native block over independently modulated attention documents.

        H3's block selects AdaLN rows along the token dimension, so members are
        packed there with distinct timestep offsets. The attention processor
        restores their batch dimension before attention and publishes separate
        pre-RoPE video pages. Text remains in each member's hidden packet.
        """
        pages = validate_batch(self, index, hidden_list, timestep_list, chunk_list, cache_list,
                               self.model.transformer_blocks if self.model is not None else None)
        members = len(hidden_list)
        expected = self.hidden_shape(self.shape)
        if any(not 0.0 <= float(timestep) <= 1.0 for timestep in timestep_list):
            raise ValueError("H3 timestep must be t=1-sigma in [0, 1]")

        batch_size, tokens, width = expected
        temb, indices = self._batch_conditioning(timestep_list, batch_size)
        rotary = self._batch_rotary(chunk_list, pages, batch_size)
        block = self.model.transformer_blocks[str(index)]
        processor = block.attn.processor
        processor.cache = BatchCache(cache_list, batch_size)
        processor.key_rotary = rotary[1:]
        processor.text_tokens = self.text_tokens
        processor.batch_shape = (members * batch_size, tokens)
        try:
            hidden = torch.cat(hidden_list, dim=0).reshape(1, members * batch_size * tokens, width)
            hidden = block(hidden, temb, indices, rotary[0])
            return list(hidden.reshape(members * batch_size, tokens, width).split(batch_size))
        finally:
            processor.cache = processor.key_rotary = processor.batch_shape = None
            processor.text_tokens = 0

    def unembed(self, hidden, timestep, chunk):
        del chunk
        temb, _, timestep_indices = self._embedding(timestep)
        hidden = self.model.norm_out(hidden, temb, timestep_indices).to(self.model.proj_out.weight.dtype)
        rows = self.model.proj_out(hidden[:, self.text_tokens:])
        # H3 predicts x0 - eps; the engine sampler consumes eps - x0.
        return -_unpatchify(rows, self.shape, self.config["patch_size"])


class H3Encoder:
    """Qwen3-VL prompt encoder producing H3's hidden_states[50] condition."""

    def __init__(self, source, max_length=None):
        self.source, self.max_length = Path(source), max_length

    def load(self, device, dtype):
        self.device, self.dtype = torch.device(device), dtype

    @torch.inference_mode()
    def encode(self, prompt):
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("H3 requires a nonempty prompt")
        from diffusers.modular_pipelines.minimax_h3.encoders import get_qwen3vl_prompt_embeds

        from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

        source = str(self.source)
        tokenizer = AutoTokenizer.from_pretrained(source, subfolder="tokenizer")
        processor = AutoProcessor.from_pretrained(source, subfolder="processor")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            source, subfolder="text_encoder", dtype=self.dtype
        ).to(self.device).eval()
        try:
            options = {"add_special_tokens": False}
            if self.max_length is not None:
                options.update(max_length=self.max_length, truncation=True)
            ids = tokenizer(prompt, **options)["input_ids"]
            text = get_qwen3vl_prompt_embeds(
                model, processor, ids, {}, text_encoder_layer=50,
                device=self.device, dtype=self.dtype,
            )
            return {"text": text.contiguous()}
        finally:
            del model, processor, tokenizer
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


class H3LatentDecoder:
    """Whole-sequence MiniMax-H3 VAE decoder.

    H3's VAE is not independently decodable one latent at a time.  A caller
    must concatenate all q=1 chunks along temporal dimension 2 and call
    :meth:`decode_sequence` once.

    ``frames`` retains the engine's pixel-frame accounting contract for DiT-only
    benchmarks. H3's temporal VAE assigns repeating ``(1, 4, 4, 4, 4)`` pixel
    contributions to latent frames and accepts a final partial group.
    """

    pixel_mean = (0.485, 0.456, 0.406)
    pixel_std = (0.229, 0.224, 0.225)

    def __init__(self, source):
        self.source = Path(source)
        self.vae = None

    def load(self, device, dtype):
        del dtype  # The reference H3 decoder keeps weights in fp32 and autocasts CUDA execution to fp16.
        self.device = torch.device(device)
        from diffusers import AutoencoderKLMiniMaxH3

        self.vae = AutoencoderKLMiniMaxH3.from_pretrained(
            self.source / "vae", torch_dtype=torch.float32
        ).to(self.device)
        self.vae.eval().requires_grad_(False)
        cfg = self.vae.config
        self.mean = torch.tensor(cfg.latents_mean, device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(cfg.latents_std, device=self.device, dtype=torch.float32).view(1, -1, 1, 1, 1)
        self.image_mean = torch.tensor(
            self.pixel_mean, device=self.device, dtype=torch.float32
        ).view(1, -1, 1, 1, 1)
        self.image_std = torch.tensor(
            self.pixel_std, device=self.device, dtype=torch.float32
        ).view(1, -1, 1, 1, 1)

    def frames(self, chunk, shape):
        temporal = shape[2]
        first = chunk * temporal
        contributions = (1, 4, 4, 4, 4)
        return sum(contributions[index % len(contributions)] for index in range(first, first + temporal))

    def decode_sequence(self, latents):
        """Decode ``[B,C,T,H,W]`` concatenated latents to pixels in ``[-1,1]``."""
        if self.vae is None:
            raise RuntimeError("load() must be called before decode_sequence()")
        if latents.ndim != 5:
            raise ValueError("H3 whole-sequence latents must have shape [B,C,T,H,W]")
        if latents.shape[2] < 3:
            raise ValueError("H3 whole-sequence decode requires at least 3 latent frames")
        z = latents.to(device=self.device, dtype=torch.float32) * self.std + self.mean
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                video = self.vae.decode(z, return_dict=False)[0]
        else:
            video = self.vae.decode(z, return_dict=False)[0]
        pixels = (video.float() * self.image_std + self.image_mean).clamp(0, 1)
        return pixels.mul(2).sub(1)
