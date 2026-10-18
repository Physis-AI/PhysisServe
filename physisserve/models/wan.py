"""Wan 2.1/2.2 prompt-only backend: DiT, text encoder and VAE decoder as independent components.

``backend(root)`` reads a diffusers-layout model directory (``transformer/``, ``text_encoder/``,
``tokenizer/``, ``vae/``). Each component loads only its own weights, on the rank
that runs it. The DiT loads only the blocks of its rank. History keys are cached before RoPE and
re-embedded at read time on positions contiguous with the current chunk, with the sink first; see
``_Attention``. The Wan 2.2 5B full-noise T2V path is supported;
image-prefix TI2V is not (it needs image input and per-token clean/noisy timesteps).
"""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..engine.chunks import Chunks
from ..engine.kv_cache import BatchCache
from .base import Backend, block_indices, load_selected_weights


def backend(root, text_length=None, cache_text_kv=False) -> Backend:
    """Build a backend from a complete Diffusers package root.

    ``text_length`` defaults to Diffusers' WanPipeline maximum of 512.
    """
    root = Path(root)
    return Backend(WanDiT(root / "transformer", cache_text_kv=cache_text_kv),
                   WanEncoder(root, **({} if text_length is None else {"max_length": text_length})), WanDecoder(root))


def _rope(x, cosine, sine):
    even, odd = x.unflatten(-1, (-1, 2)).unbind(-1)
    result = torch.empty_like(x)
    result[..., 0::2] = even * cosine[..., 0::2] - odd * sine[..., 1::2]
    result[..., 1::2] = even * sine[..., 0::2] + odd * cosine[..., 1::2]
    return result


def _tensor_revision(value):
    # Inference tensors have no version counter. Prepared conditioning is owned
    # by the request and must stay immutable, as in the existing embedding cache.
    return None if value.is_inference() else value._version


class _CachedTextProjection(nn.Module):
    """Bounded memo of the original projection, without changing its GEMM shape.

    The strong source reference prevents pointer reuse from becoming a cache hit.
    Normal tensors also check their mutation counter. Only inference/no-grad calls
    may reuse results; prepared inference tensors and model weights are immutable.
    """

    def __init__(self, module, capacity=8):
        super().__init__()
        self.module, self.capacity, self.entries = module, capacity, []
        self.train(module.training)

    def forward(self, value):
        if self.training or torch.is_grad_enabled():
            return self.module(value)
        revision = _tensor_revision(value)
        for index, (source, version, result) in enumerate(self.entries):
            if source is value:
                self.entries.pop(index)
                if version == revision:
                    self.entries.append((source, version, result))
                    return result
                break
        result = self.module(value)
        self.entries.append((value, revision, result))
        if len(self.entries) > self.capacity:
            self.entries.pop(0)
        return result


class _Attention:
    """Replacement for diffusers' self-attention processor: K/V pass through the engine cache, then SDPA.

    Keys are cached before RoPE and re-embedded at read time on the positions of the window: the
    history pages (in the cache's order: sink chunks first, then the recent chunks) are laid out as the
    frames just before the current chunk, so a sink chunk keeps a constant distance to the chunk that
    reads it however far the window has moved. The cache occupies the ``sink + temporal`` slots right
    before the window; without a sink, or with the whole history in the window, it is the plain absolute
    position.
    ``rotary_emb`` is ``(own, window)``: the (cos, sin) tables of the current chunk and of the window
    [history pages | current chunk]. One instance per block, installed at load time; ``cache`` is set for
    the duration of one block call only, so no KV page outlives the plan through the processor.
    """

    def __init__(self):
        self.cache = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, rotary_emb=None):
        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("this processor serves unmasked chunk self-attention only")
        own, window = rotary_emb
        q = attn.norm_q(attn.to_q(hidden_states)).unflatten(2, (attn.heads, -1))
        k = attn.norm_k(attn.to_k(hidden_states)).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        v = attn.to_v(hidden_states).unflatten(2, (attn.heads, -1)).transpose(1, 2)
        k, v = self.cache(k, v)  # pages are [B, heads, tokens, head_dim], before RoPE
        q = _rope(q, *own).type_as(q).transpose(1, 2)
        k = _rope(k.transpose(1, 2), *window).type_as(k).transpose(1, 2)
        hidden = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).flatten(2)
        return attn.to_out[1](attn.to_out[0](hidden))


class WanDiT:
    """A Wan transformer loaded from a Diffusers ``transformer/`` directory."""

    first_frames = 3
    sink_sigma = 0.0

    def __init__(self, source, cache_text_kv=False):
        from diffusers import WanTransformer3DModel

        self.source = Path(source)
        if type(cache_text_kv) is not bool:
            raise ValueError("cache_text_kv must be a boolean")
        self.cache_text_kv = cache_text_kv
        self.config = WanTransformer3DModel.load_config(self.source)
        if self.config["in_channels"] != self.config["out_channels"] or self.config.get("image_dim") is not None:
            raise ValueError("WanDiT supports prompt-only checkpoints with matching input/output channels; "
                             "image-prefix TI2V is not implemented")
        self.num_blocks = self.config["num_layers"]
        self.model = None

    def load(self, blocks, device, dtype):
        from diffusers import WanTransformer3DModel
        from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

        blocks = block_indices(blocks, self.num_blocks)
        cfg = self.config
        with torch.device("meta"):
            model = WanTransformer3DModel.from_config(cfg)
        model.blocks = nn.ModuleDict({str(i): model.blocks[i] for i in blocks})
        if 0 not in blocks:
            del model.patch_embedding
        if self.num_blocks - 1 not in blocks:
            del model.norm_out, model.proj_out, model.scale_shift_table
        keep_fp32 = model._keep_in_fp32_modules
        state = {name: tensor.to(device=device, dtype=torch.float32 if any(
            part in keep_fp32 for part in name.split(".")) else dtype)
            for name, tensor in load_selected_weights(self.source, model.state_dict()).items()}
        model.load_state_dict(state, assign=True)
        model.rope = WanRotaryPosEmbed(cfg["attention_head_dim"], cfg["patch_size"],
                                      cfg.get("rope_max_seq_len", 1024)).to(device)
        for block in model.blocks.values():
            block.attn1.set_processor(_Attention())
        self.model = model.eval().requires_grad_(False)
        self.device, self.dtype = torch.device(device), dtype

    @property
    def page_dtype(self):
        """dtype of the K/V pages this DiT writes."""
        return self.dtype

    def timestep_for_sigma(self, sigma):
        return 1000.0 * float(sigma)

    def prepare(self, condition, shape, chunks, fps=None):
        shape = tuple(shape)
        cfg, b, c, _, h, w = self.config, *shape
        pt, ph, pw = cfg["patch_size"]
        if pt != 1 or h % ph or w % pw or c != cfg["in_channels"]:
            raise ValueError("requires temporal patch 1, patch-aligned spatial size and C == in_channels")
        text = condition.get("text")
        if text is None or text.ndim != 3 or text.shape[0] != b or text.shape[2] != cfg["text_dim"]:
            raise ValueError("condition['text'] must be [B, L, text_dim]")
        self.geometry = Chunks(shape, chunks, self.first_frames)
        required = max(self.geometry.total, h // ph, w // pw)
        rope = self.model.rope
        if required > rope.freqs_cos.shape[0]:
            from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

            # Grow the table, not the position scale: current chunks retain their
            # absolute positions, and old positions keep their exact stored bits.
            grown = WanRotaryPosEmbed(cfg["attention_head_dim"], cfg["patch_size"], required).to(
                device=rope.freqs_cos.device, dtype=rope.freqs_cos.dtype).train(rope.training)
            for name in ("freqs_cos", "freqs_sin"):
                previous = getattr(rope, name)
                getattr(grown, name)[:previous.shape[0]].copy_(previous)
            self.model.rope = grown
        self.shape = tuple(shape)
        self.chunks = chunks
        self.text = text.to(self.device, self.dtype)
        self.embeddings, self.rotary = {}, {}
        self._frame_cache = None
        if self.cache_text_kv:
            self._reset_text_kv_cache()

    def _reset_text_kv_cache(self):
        """Install request-local caches only around known prompt-only processors.

        Keep Diffusers' processor, attention backend, query path and rounding.
        Added image K/V, fused projections and custom processors use their
        original uncached modules. They are not silently reinterpreted as T2V.
        """
        from diffusers.models.transformers.transformer_wan import WanAttnProcessor

        self._text_batches = []
        for block in self.model.blocks.values():
            attn = block.attn2
            for name in ("to_k", "to_v", "norm_k"):
                module = getattr(attn, name)
                if isinstance(module, _CachedTextProjection):
                    setattr(attn, name, module.module)
            if (type(attn.processor) is not WanAttnProcessor
                    or not attn.is_cross_attention or attn.fused_projections
                    or attn.add_k_proj is not None or attn.add_v_proj is not None):
                continue
            for name in ("to_k", "to_v", "norm_k"):
                setattr(attn, name, _CachedTextProjection(getattr(attn, name)))

    def _batch_text(self, texts):
        """Reuse identical ordered conditioning packs, not single-member K/V.

        Each first use still executes the original full-batch projection. The
        cache distinguishes member order and size, and retains at most 8 packs.
        """
        if not self.cache_text_kv or torch.is_grad_enabled():
            return torch.cat(texts, dim=0)
        versions = tuple(_tensor_revision(text) for text in texts)
        for index, (sources, revisions, result) in enumerate(self._text_batches):
            if len(sources) == len(texts) and all(a is b for a, b in zip(sources, texts)):
                self._text_batches.pop(index)
                if revisions == versions:
                    self._text_batches.append((sources, revisions, result))
                    return result
                break
        result = torch.cat(texts, dim=0)
        self._text_batches.append((tuple(texts), versions, result))
        if len(self._text_batches) > 8:
            self._text_batches.pop(0)
        return result

    def prepare_timesteps(self, timesteps):
        for timestep in timesteps:
            self._embedding(timestep)

    def hidden_shape(self, shape):
        b, _, t, h, w = shape
        _, ph, pw = self.config["patch_size"]
        return (b, t * (h // ph) * (w // pw), self.config["num_attention_heads"] * self.config["attention_head_dim"])

    def kv_shape(self, shape):
        b, tokens, _ = self.hidden_shape(shape)
        return (b, self.config["num_attention_heads"], tokens, self.config["attention_head_dim"])

    def _embedding(self, timestep):
        key = float(timestep)  # timesteps are CPU scalars, so this is free (no device sync)
        if key not in self.embeddings:
            timestep = timestep.to(self.device)
            temb, modulation, text, _ = self.model.condition_embedder(timestep.expand(self.shape[0]), self.text)
            self.embeddings[key] = temb, modulation.unflatten(1, (6, -1)), text
        return self.embeddings[key]

    def _frames_table(self, start, stop):
        """(cos, sin) of latent frames [start, stop) of a clip that starts at frame 0, as [1, frames*h*w, 1, dim]."""
        _, _, _, h, w = self.shape
        _, ph, pw = self.config["patch_size"]
        h, w, rope, frames = h // ph, w // pw, self.model.rope, stop - start
        parts = []
        for frequencies in (rope.freqs_cos, rope.freqs_sin):
            ft, fh, fw = frequencies.split((rope.t_dim, rope.h_dim, rope.w_dim), dim=1)
            parts.append(torch.cat((ft[start:stop, None, None].expand(frames, h, w, -1),
                                    fh[None, :h, None].expand(frames, h, w, -1),
                                    fw[None, None, :w].expand(frames, h, w, -1)), dim=-1).reshape(1, frames * h * w, 1, -1))
        return tuple(x.float() for x in parts)

    def _rotary(self, chunk, pages):
        """(own, window): the tables of the current chunk, and of [pages history chunks | current chunk] laid out
        as the contiguous frames ending with the chunk, so the history is always read as the frames just before it
        (a sink chunk is relocated; a contiguous recent history keeps its absolute positions)."""
        if (chunk, pages) not in self.rotary:
            if not 0 <= pages <= chunk:
                raise ValueError("Wan cache has more history pages than preceding chunks")
            # Positions count latent frames, so a first chunk of its own length lays out the same way.
            geometry = self.geometry
            start = geometry.start(chunk)
            first, stop = start - geometry.history_frames(pages), start + geometry.frames_of(chunk)
            # Reuse the overlapping absolute-position window, with at most
            # eight lookahead frames. No positions or fp32 RoPE math change.
            cached = self._frame_cache
            if (cached is None or first < cached[0] or stop > cached[1]
                    or cached[1] - cached[0] > stop - first + 8):
                end = min(stop + 8, geometry.total)
                cached = self._frame_cache = (first, end, self._frames_table(first, end))
            tokens = self.hidden_shape(self.shape)[1] // self.shape[2]
            def take(left_frame):
                left, right = (left_frame - cached[0]) * tokens, (stop - cached[0]) * tokens
                return tuple(table[:, left:right] for table in cached[2])
            own = take(start)
            window = own if pages == 0 else take(first)
            self.rotary = {(chunk, pages): (own, window)}  # keep only the current tables
        return self.rotary[chunk, pages]

    def embed(self, latent, timestep, chunk):
        return self.model.patch_embedding(latent).flatten(2).transpose(1, 2)

    def block(self, index, hidden, timestep, chunk, cache):
        _, modulation, text = self._embedding(timestep)
        block = self.model.blocks[str(index)]
        block.attn1.processor.cache = cache
        try:
            return block(hidden, text, modulation, self._rotary(chunk, len(cache.history)))
        finally:
            block.attn1.processor.cache = None

    def block_batch(self, index, hidden_list, timestep_list, chunk_list, cache_list):
        """Execute one native block batch for compatible members of the prepared stream.

        Each member retains its own timestep, chunk position and engine cache. History lengths must
        match; the scheduler groups that compatibility class before calling this method. Both own
        and window RoPE tables are stacked, so noncontiguous sink history keeps the serial path's
        effective positions and published pages remain pre-RoPE. Each returned tensor has the
        prepared hidden shape (including its original batch dimension).
        """
        history = len(cache_list[0].history)
        embeddings = [self._embedding(timestep) for timestep in timestep_list]
        modulation = torch.cat([embedding[1] for embedding in embeddings], dim=0)
        text = self._batch_text([embedding[2] for embedding in embeddings])
        batch_size = self.shape[0]
        tables = [self._rotary(chunk, history) for chunk in chunk_list]
        rotary = tuple(tuple(torch.cat([table[part][trig].expand(batch_size, -1, -1, -1)
                                        for table in tables], dim=0) for trig in (0, 1)) for part in (0, 1))
        block = self.model.blocks[str(index)]
        block.attn1.processor.cache = BatchCache(cache_list, batch_size)
        try:
            hidden = block(torch.cat(hidden_list, dim=0), text, modulation, rotary)
            return list(hidden.split(batch_size, dim=0))
        finally:
            block.attn1.processor.cache = None

    def unembed(self, hidden, timestep, chunk):
        temb, _, _ = self._embedding(timestep)
        shift, scale = (self.model.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden = (self.model.norm_out(hidden.float()) * (1 + scale) + shift).type_as(hidden)
        hidden = self.model.proj_out(hidden)
        b, _, _, h, w = self.shape
        t = self.geometry.frames_of(chunk)
        pt, ph, pw = self.config["patch_size"]
        hidden = hidden.reshape(b, t // pt, h // ph, w // pw, pt, ph, pw, -1)
        return hidden.permute(0, 7, 1, 4, 2, 5, 3, 6).flatten(6, 7).flatten(4, 5).flatten(2, 3)


class WanEncoder:
    """UMT5 text encoder loaded from a complete Diffusers model root.

    Produces the same [1, max_length, text_dim] embedding as diffusers' WanPipeline: the prompt cleaned
    the same way, the encoder output up to the prompt's length, zero-padded to max_length.
    """

    def __init__(self, source, max_length: int = 512):
        self.root, self.max_length = Path(source), max_length
        self.tokenizer = self.model = None

    def load(self, device, dtype):
        from transformers import AutoTokenizer, UMT5EncoderModel

        self.tokenizer = AutoTokenizer.from_pretrained(self.root / "tokenizer")
        self.model = UMT5EncoderModel.from_pretrained(self.root / "text_encoder", torch_dtype=dtype).to(device)
        self.model.eval().requires_grad_(False)
        self.device = torch.device(device)

    def encode(self, prompt):
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean

        tokens = self.tokenizer([prompt_clean(prompt)], padding="max_length", max_length=self.max_length,
                                truncation=True, add_special_tokens=True, return_attention_mask=True,
                                return_tensors="pt")
        ids, mask = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)
        hidden = self.model(ids, mask).last_hidden_state
        length = int(mask.gt(0).sum())
        return {"text": torch.cat((hidden[:, :length], hidden.new_zeros(1, self.max_length - length, hidden.shape[2])), dim=1)}


class WanDecoder:
    """Causal Wan VAE decoded chunk by chunk with its temporal cache carried across chunks.

    The VAE is loaded from a complete Diffusers model root. The output is identical to decoding the whole
    latent video at once, for the Wan 2.1 VAE and the residual/patched Wan 2.2 VAE.
    """

    def __init__(self, source):
        self.root = Path(source)
        self.vae = None
        cfg = json.loads((self.root / "vae" / "config.json").read_text())
        self.temporal = cfg.get("scale_factor_temporal") or 4  # known before loading: frame counts need no VAE

    def load(self, device, dtype):
        from diffusers import AutoencoderKLWan

        self.vae = AutoencoderKLWan.from_pretrained(self.root / "vae", torch_dtype=dtype).to(device)
        self.vae.eval().requires_grad_(False)
        cfg = self.vae.config
        self.mean = torch.tensor(cfg.latents_mean, device=device, dtype=dtype).view(1, -1, 1, 1, 1)
        self.std = torch.tensor(cfg.latents_std, device=device, dtype=dtype).view(1, -1, 1, 1, 1)

    def frames(self, chunk, shape):
        return shape[2] * self.temporal - (self.temporal - 1 if chunk == 0 else 0)

    def reset(self):
        self.vae.clear_cache()

    def decode(self, latent, chunk):
        hidden = self.vae.post_quant_conv(latent * self.std + self.mean)
        frames = []
        for i in range(hidden.shape[2]):
            self.vae._conv_idx = [0]
            frames.append(self.vae.decoder(hidden[:, :, i:i + 1], feat_cache=self.vae._feat_map,
                                           feat_idx=self.vae._conv_idx, first_chunk=chunk == 0 and i == 0))
        pixels = torch.cat(frames, dim=2)
        if self.vae.config.patch_size is not None:
            from diffusers.models.autoencoders.autoencoder_kl_wan import unpatchify

            pixels = unpatchify(pixels, patch_size=self.vae.config.patch_size)
        return pixels.clamp(-1, 1)
