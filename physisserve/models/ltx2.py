"""LTX-2.5 backend: the video stream of the audio-video DiT, the Gemma text encoder with the LTX text
connectors, and the video VAE, as three independent components.

``backend(root)`` reads a diffusers-layout model directory (``transformer/``, ``text_encoder/``,
``tokenizer/``, ``connectors/``, ``vae/``, ``model_index.json``). History keys are cached before RoPE and
re-embedded at read time on positions contiguous with the current chunk, so the sink and recent pages form
the checkpoint window. The video path runs with modalities isolated, so the transformer's audio stream
is not loaded. Needs diffusers >= 0.40 and transformers >= 5.14.
"""
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from ..engine.chunks import Chunks
from ..engine.kv_cache import BatchCache
from .base import Backend, block_indices, drop_attributes, load_selected_weights



def backend(root, text_length=None, compute_dtype=None) -> Backend:
    """Build a backend from a complete Diffusers package root.

    ``text_length`` defaults to Diffusers' LTX2Pipeline maximum of 1024. ``compute_dtype``: see ``LTX2DiT``.
    """
    root = Path(root)
    return Backend(LTX2DiT(root / "transformer", compute_dtype=compute_dtype),
                   LTX2Encoder(root, **({} if text_length is None else {"max_length": text_length})), LTX2Decoder(root))


def _autocast(device, dtype):
    """The checkpoint is run under autocast in its reference implementation; match its numerics."""
    return torch.autocast(device.type, dtype=dtype) if dtype != torch.float32 else nullcontext()


class _Attention:
    """Replacement for the video self-attention processor: history K/V come through the engine cache.

    Keys are cached before RoPE; ``rotary`` holds the (cos, sin) tables of the current chunk and of the
    whole window (history pages first, in the cache's order, then the current chunk), set for the
    duration of one block call only.
    """

    def __init__(self):
        self.cache = self.rotary = None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, query_rotary_emb=None,
                 key_rotary_emb=None, perturbation_mask=None, all_perturbed=None):
        from diffusers.models.transformers.transformer_ltx2 import apply_interleaved_rotary_emb, apply_split_rotary_emb

        if encoder_hidden_states is not None or attention_mask is not None:
            raise ValueError("this processor serves unmasked chunk self-attention only")
        rope = apply_split_rotary_emb if attn.rope_type == "split" else apply_interleaved_rotary_emb
        own, window = self.rotary
        gate_logits = attn.to_gate_logits(hidden_states) if attn.to_gate_logits is not None else None
        query = attn.norm_q(attn.to_q(hidden_states))
        key = attn.norm_k(attn.to_k(hidden_states))
        value = attn.to_v(hidden_states)
        key, value = self.cache(key.unsqueeze(1), value.unsqueeze(1))  # pages are [B, 1, tokens, inner]
        query = rope(query, own)
        key = rope(key.squeeze(1), window)
        value = value.squeeze(1)
        batch, length, _ = query.shape

        def split_heads(x):
            return x.view(batch, x.shape[1], attn.heads, attn.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(split_heads(query), split_heads(key), split_heads(value))
        out = out.transpose(1, 2).reshape(batch, length, -1).to(query.dtype)
        if gate_logits is not None:
            gate = 2.0 * torch.sigmoid(gate_logits)
            out = (out.view(batch, length, attn.heads, attn.head_dim) * gate.unsqueeze(-1)).reshape(batch, length, -1)
        return attn.to_out[1](attn.to_out[0](out))


AUDIO_BLOCK_PARTS = ("audio_norm1", "audio_attn1", "audio_norm2", "audio_attn2", "audio_to_video_norm",
                     "audio_to_video_attn", "video_to_audio_norm", "video_to_audio_attn", "audio_norm3", "audio_ff",
                     "audio_scale_shift_table", "audio_prompt_scale_shift_table", "video_a2v_cross_attn_scale_shift_table",
                     "audio_a2v_cross_attn_scale_shift_table")
AUDIO_MODEL_PARTS = ("audio_proj_in", "audio_caption_projection", "audio_time_embed", "audio_prompt_adaln",
                     "av_cross_attn_video_scale_shift", "av_cross_attn_audio_scale_shift", "av_cross_attn_video_a2v_gate",
                     "av_cross_attn_audio_v2a_gate", "audio_rope", "cross_attn_rope", "cross_attn_audio_rope",
                     "audio_norm_out", "audio_proj_out", "audio_scale_shift_table")


class LTX2DiT:
    """A Diffusers LTX2VideoTransformer3DModel transformer directory."""

    first_frames = 3
    sink_sigma = 0.0

    def __init__(self, source, compute_dtype=None):
        """``compute_dtype``: the weights' dtype and the autocast dtype when it differs from the engine's state
        dtype (``load``'s dtype). The packaged LTX-2.5 checkpoint keeps latents, noise and prompt context in
        float32 and runs bf16 weights under bf16 autocast: engine dtype float32 with
        compute_dtype bfloat16 is that regime. None: one dtype for both."""
        from diffusers import LTX2VideoTransformer3DModel

        self.compute_dtype = compute_dtype
        self.source = Path(source)
        self.config = LTX2VideoTransformer3DModel.load_config(self.source)
        if self.config.get("use_prompt_embeddings", True):
            raise ValueError("LTX2DiT expects text conditions from the LTX text connectors (use_prompt_embeddings=False)")
        if self.config["patch_size"] != 1 or self.config["patch_size_t"] != 1:
            raise ValueError("LTX2DiT supports patch size 1 only")
        self.num_blocks = self.config["num_layers"]
        self.model = None

    def load(self, blocks, device, dtype):
        from diffusers import LTX2VideoTransformer3DModel

        state_dtype, dtype = dtype, self.compute_dtype or dtype  # dtype: weights and autocast from here on
        blocks = block_indices(blocks, self.num_blocks)
        with torch.device("meta"):
            model = LTX2VideoTransformer3DModel.from_config(self.config)
        model.transformer_blocks = nn.ModuleDict({str(i): model.transformer_blocks[i] for i in blocks})
        for block in model.transformer_blocks.values():
            drop_attributes(block, AUDIO_BLOCK_PARTS)
        drop_attributes(model, AUDIO_MODEL_PARTS)
        if 0 not in blocks:
            drop_attributes(model, ("proj_in", "keyframes_abs_pos_embedding"))
        if self.num_blocks - 1 not in blocks:
            drop_attributes(model, ("norm_out", "proj_out", "scale_shift_table"))
        state = {name: tensor.to(device=device, dtype=dtype)
                 for name, tensor in load_selected_weights(self.source, model.state_dict()).items()}
        model.load_state_dict(state, assign=True)
        for block in model.transformer_blocks.values():
            block.attn1.set_processor(_Attention())
        self.model = model.eval().requires_grad_(False)
        self.device, self.dtype, self.state_dtype = torch.device(device), dtype, state_dtype

    def _state_dtype(self):
        """dtype of the engine state (latents, noise, prompt context): load's dtype, else the weights'."""
        return getattr(self, "state_dtype", None) or self.dtype

    @property
    def hidden_dtype(self):
        """dtype of the hidden states between blocks: the compute dtype (autocast), not the engine state's."""
        return self.dtype

    @property
    def page_dtype(self):
        """dtype of the K/V pages this DiT writes (projections under autocast)."""
        return self.dtype

    def timestep_for_sigma(self, sigma):
        return 1000.0 * float(sigma)

    def prepare(self, condition, shape, chunks, fps=None):
        """Prepare the fixed 3-latent head followed by one-latent steady chunks."""
        shape = tuple(shape)
        cfg = self.config
        b, c = shape[:2]
        if c != cfg["in_channels"]:
            raise ValueError("requires C == in_channels")
        if fps is None:
            raise ValueError("ltx2 needs the request's fps: its temporal positions depend on the frame rate")
        self.fps = float(fps)
        text = condition.get("text")
        if text is None or text.ndim != 3 or text.shape[0] != b or text.shape[2] != cfg["cross_attention_dim"]:
            raise ValueError("condition['text'] must be [B, L, cross_attention_dim] (the connectors' video output)")
        rope = self.model.rope
        if (rope.patch_size_t, rope.patch_size) != (1, 1):
            raise ValueError("LTX2DiT coordinates support patch size 1 only")
        self.shape = shape
        self.geometry = Chunks(shape, chunks, self.first_frames)
        self.text = text.to(self.device, self._state_dtype())
        # what the transformer's forward makes of an all-ones 2D text mask: an additive zero mask, kept so that
        # cross-attention takes the same kernel path as the reference implementation
        self.text_mask = torch.zeros(b, 1, text.shape[1], device=self.device, dtype=self.dtype)
        self.embeddings, self.rotary = {}, {}
        self.chunks = chunks
        self._frame_cache = None
        self._batch_rotary_cache = None
        # The native coordinate formula uses these same FP32 axes and constants.
        # Prepare their device storage once: constructing a CUDA tensor from a
        # Python sequence otherwise performs a blocking host-to-device copy.
        self._coord_height = torch.arange(shape[3], dtype=torch.float32, device=self.device)
        self._coord_width = torch.arange(shape[4], dtype=torch.float32, device=self.device)
        self._coord_patch = torch.tensor((1, 1, 1), dtype=torch.float32, device=self.device).view(3, 1, 1, 1)
        self._coord_scale = torch.tensor(rope.scale_factors, device=self.device).view(1, 3, 1, 1)

    def prepare_timesteps(self, timesteps):
        for timestep in timesteps:
            self._embedding(timestep)

    def hidden_shape(self, shape):
        b, _, t, h, w = shape
        return (b, t * h * w, self.config["num_attention_heads"] * self.config["attention_head_dim"])

    def kv_shape(self, shape):
        b, tokens, inner = self.hidden_shape(shape)
        return (b, 1, tokens, inner)

    def _embedding(self, timestep, chunk=None):
        """Per-token modulation of one timestep (the reference feeds one timestep per token) for the tokens of
        ``chunk`` (default: a chunk of the request's shape), cached per (value, token count)."""
        shape = self.shape if chunk is None else self.geometry.shape_of(chunk)
        b, tokens, _ = self.hidden_shape(shape)
        key = float(timestep), tokens
        if key not in self.embeddings:
            with _autocast(self.device, self.dtype):
                per_token = torch.full((b * tokens,), key[0], device=self.device)
                temb, embedded = self.model.time_embed(per_token, batch_size=b, hidden_dtype=self._state_dtype())
                temb, embedded = temb.view(b, -1, temb.size(-1)), embedded.view(b, -1, embedded.size(-1))
                prompt = None
                if self.model.prompt_modulation and self.config.get("use_prompt_adaln_single", True):
                    prompt, _ = self.model.prompt_adaln(torch.full((b,), key[0], device=self.device), batch_size=b,
                                                        hidden_dtype=self._state_dtype())
                    prompt = prompt.view(b, -1, prompt.size(-1))
            self.embeddings[key] = temb, embedded, prompt
        return self.embeddings[key]

    def _coords(self, positions):
        """RoPE coordinates of the given latent frames of a clip that starts at frame 0 (frame, height, width)."""
        positions = tuple(positions)
        if not positions or any(type(frame) is not int or frame < 0 for frame in positions):
            raise ValueError("LTX frame positions must be nonempty nonnegative integers")
        if all(right == left + 1 for left, right in zip(positions, positions[1:])):
            frames = torch.arange(positions[0], positions[-1] + 1, device=self.device,
                                  dtype=torch.int64).to(torch.float32)
        else:
            # Noncontiguous coordinates are useful to independent references.
            # Keep this path bounded too, without transferring a Python index
            # list to CUDA or generating the entire absolute-position prefix.
            frames = torch.stack([torch.full((), frame, dtype=torch.float32, device=self.device)
                                  for frame in positions])
        grid = torch.stack(torch.meshgrid(frames, self._coord_height, self._coord_width, indexing="ij"))
        latent_coords = torch.stack((grid, grid + self._coord_patch), dim=-1)
        latent_coords = latent_coords.flatten(1, 3).unsqueeze(0).repeat(self.shape[0], 1, 1, 1)
        coords = latent_coords * self._coord_scale
        # Preserve native FP32 operation order, especially the first-frame
        # causal clamp. Shifting already-divided relative coordinates would not
        # have the same rounding or first-frame semantics.
        rope = self.model.rope
        coords[:, 0, ...] = (coords[:, 0, ...] + rope.causal_offset - rope.scale_factors[0]).clamp(min=0)
        coords[:, 0, ...] = coords[:, 0, ...] / self.fps
        return coords

    def _rotary(self, chunk, pages):
        """(cos, sin) of the current chunk, and of [history pages | current] laid out contiguously before it."""
        key = (chunk, pages)
        if key not in self.rotary:
            if not 0 <= pages <= chunk:
                raise ValueError("LTX cache has more history pages than preceding chunks")
            # Positions count latent frames, so a first chunk of its own length lays out the same way.
            geometry = self.geometry
            start = geometry.start(chunk)
            first, stop = start - geometry.history_frames(pages), start + geometry.frames_of(chunk)
            cached = self._frame_cache
            if (cached is None or first < cached[0] or stop > cached[1]
                    or cached[1] - cached[0] > stop - first + 8):
                end = min(stop + 8, geometry.total)
                tables = self.model.rope(self._coords(range(first, end)), device=self.device)
                cached = self._frame_cache = (first, end, tables)
            tokens = self.shape[3] * self.shape[4]
            def take(left_frame):
                left, right = (left_frame - cached[0]) * tokens, (stop - cached[0]) * tokens
                # Split tables are [B, heads, tokens, r]; interleaved tables
                # are [B, tokens, dim]. Both retain exact absolute coordinates.
                return tuple(table.narrow(2 if table.ndim == 4 else 1, left, right - left)
                             for table in cached[2])
            own = take(start)
            window = own if pages == 0 else take(first)
            self.rotary = {key: (own, window)}  # keep only the current tables
        return self.rotary[key]

    def _batch_rotary(self, chunks, history):
        """Reuse read-only members of one packed entry across local blocks.

        A contiguous subset views the retained superset. Otherwise replace the
        entry, reusing matching members and constructing only missing ones.
        Partial replacement temporarily retains the previous allocation.
        Request preparation clears this cache; native singleton tables stay separate.
        """
        chunks = tuple(chunks)
        if (not chunks or type(history) is not int or history < 0
                or any(type(chunk) is not int or not history <= chunk < self.chunks for chunk in chunks)):
            raise ValueError("chunks must be in the prepared request and have the stated history")
        batch = self.shape[0]
        key = chunks, history, batch
        old = self._batch_rotary_cache
        if old is not None and old[0] == key:
            return old[1]

        def take(tables, offset, members):
            return tuple(tuple(value.narrow(0, offset * batch, members * batch) for value in pair)
                         for pair in tables)

        old_chunks = ()
        if old is not None and old[0][1:] == key[1:]:
            old_chunks = old[0][0]
            for offset in range(len(old_chunks) - len(chunks) + 1):
                if old_chunks[offset:offset + len(chunks)] == chunks:
                    # Keep the allocation's superset key, not the view's key.
                    return take(old[1], offset, len(chunks))
        self._batch_rotary_cache = None
        positions = {chunk: index for index, chunk in enumerate(old_chunks)}
        if not any(chunk in positions for chunk in chunks):
            old = None  # release disjoint/incompatible storage before native construction
        tables = [take(old[1], positions[chunk], 1) if old is not None and chunk in positions
                  else self._rotary(chunk, history) for chunk in chunks]
        # Preserve native FP32 coordinates and table layout for each member.
        rotary = tuple(tuple(torch.cat([table[part][trig] for table in tables], dim=0)
                             for trig in (0, 1)) for part in (0, 1))
        self._batch_rotary_cache = key, rotary
        return rotary

    def embed(self, latent, timestep, chunk):
        b, c, t, h, w = latent.shape
        rows = latent.permute(0, 2, 3, 4, 1).reshape(b, t * h * w, c)  # tokens in (frame, height, width) order
        with _autocast(self.device, self.dtype):
            return self.model.proj_in(rows)

    def block(self, index, hidden, timestep, chunk, cache):
        temb, _, prompt = self._embedding(timestep, chunk)
        return self._block(index, hidden, temb, prompt, self.text, self.text_mask,
                           self._rotary(chunk, len(cache.history)), cache)

    def _block(self, index, hidden, temb, prompt, text, text_mask, rotary, cache):
        """The video block's native arithmetic, shared by singleton and packed calls."""
        blk = self.model.transformer_blocks[str(index)]
        processor = blk.attn1.processor
        processor.cache, processor.rotary = cache, rotary
        try:
            with _autocast(self.device, self.dtype):
                batch = hidden.shape[0]
                params = blk.get_mod_params(blk.scale_shift_table, temb, batch)
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params[:6]
                if blk.video_cross_attn_adaln:
                    shift_text_q, scale_text_q, gate_text_q = params[6:9]
                norm = blk.norm1(hidden) * (1 + scale_msa) + shift_msa
                hidden = hidden + blk.attn1(norm, encoder_hidden_states=None, query_rotary_emb=processor.rotary[0],
                                            attention_mask=None) * gate_msa
                if blk.cross_attn_adaln:
                    if prompt is not None:
                        shift_text_kv, scale_text_kv = blk.get_mod_params(blk.prompt_scale_shift_table, prompt, batch)
                    else:
                        shift_text_kv, scale_text_kv = (blk.prompt_scale_shift_table[None, None]
                                                        .to(device=hidden.device, dtype=hidden.dtype).unbind(dim=2))
                    text = text * (1 + scale_text_kv) + shift_text_kv
                norm = blk.norm2(hidden)
                if blk.video_cross_attn_adaln:
                    norm = norm * (1 + scale_text_q) + shift_text_q
                attn = blk.attn2(norm, encoder_hidden_states=text, query_rotary_emb=None, attention_mask=text_mask)
                if blk.video_cross_attn_adaln:
                    attn = attn * gate_text_q
                hidden = hidden + attn
                norm = blk.norm3(hidden) * (1 + scale_mlp) + shift_mlp
                return hidden + blk.ff(norm) * gate_mlp
        finally:
            processor.cache = processor.rotary = None

    def block_batch(self, index, hidden_list, timestep_list, chunk_list, cache_list):
        """Run one video block over packed compatible members with independent pre-RoPE KV.

        The request's original batch dimension is retained for each member. Timesteps, prompt
        AdaLN and both RoPE windows follow that member, while the block projections, attention
        and feed-forward run once over the combined batch. Unequal history lengths are grouped
        by the engine before entry; this method never substitutes serial block calls.
        """
        history_length = len(cache_list[0].history)
        members = len(hidden_list)
        embeddings = [self._embedding(timestep, chunk) for timestep, chunk in zip(timestep_list, chunk_list)]
        temb = torch.cat([embedding[0] for embedding in embeddings], dim=0)
        prompt = (torch.cat([embedding[2] for embedding in embeddings], dim=0)
                  if embeddings[0][2] is not None else None)
        rotary = self._batch_rotary(chunk_list, history_length)
        batch_size = self.shape[0]
        hidden = self._block(index, torch.cat(hidden_list, dim=0), temb, prompt,
                             torch.cat([self.text] * members, dim=0),
                             torch.cat([self.text_mask] * members, dim=0), rotary,
                             BatchCache(cache_list, batch_size))
        return list(hidden.split(batch_size, dim=0))

    def unembed(self, hidden, timestep, chunk):
        _, embedded, _ = self._embedding(timestep, chunk)
        b, c, _, h, w = self.shape
        t = self.geometry.frames_of(chunk)
        with _autocast(self.device, self.dtype):
            values = self.model.scale_shift_table[None, None] + embedded[:, :, None]
            shift, scale = values[:, :, 0], values[:, :, 1]
            out = self.model.proj_out(self.model.norm_out(hidden) * (1 + scale) + shift)
        return out.reshape(b, t, h, w, c).permute(0, 4, 1, 2, 3)


class LTX2Encoder:
    """Gemma text encoder + LTX text connectors, the video branch: {"text": [1, L, cross_attention_dim]}.

    source: a complete model root (model_index.json names the tokenizer and text encoder classes; text_encoder/,
    tokenizer/, connectors/).
    """

    def __init__(self, source, max_length: int = 1024):
        self.source, self.max_length = Path(source), max_length
        self.tokenizer = self.model = self.connectors = None

    def load(self, device, dtype):
        import transformers
        from diffusers.pipelines.ltx2.connectors import LTX2TextConnectors

        index = json.loads((self.source / "model_index.json").read_text())
        self.tokenizer = getattr(transformers, index["tokenizer"][1]).from_pretrained(self.source / "tokenizer")
        self.model = getattr(transformers, index["text_encoder"][1]).from_pretrained(
            self.source / "text_encoder", dtype=dtype).to(device)
        self.connectors = LTX2TextConnectors.from_pretrained(
            self.source, subfolder="connectors", dtype=dtype).to(device)
        self.model.eval().requires_grad_(False)
        self.connectors.eval().requires_grad_(False)
        self.device, self.dtype = torch.device(device), dtype

    def encode(self, prompt):
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        tokens = self.tokenizer([prompt.strip()], padding="max_length", max_length=self.max_length, truncation=True,
                                add_special_tokens=True, return_tensors="pt").to(self.device)
        hidden = self.model(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask,
                            output_hidden_states=True).hidden_states  # the embedding layer and every transformer layer
        stacked = torch.stack(hidden, dim=-1).flatten(2, 3).to(self.dtype)
        video, _, mask = self.connectors(stacked, tokens.attention_mask, padding_side="left")
        return {"text": video[:, mask[0].bool()]}


class LTX2Decoder:
    """The non-causal LTX-2.5 video VAE, decoded as one latent sequence.

    source: a complete model root containing vae/.
    """

    def __init__(self, source):
        from diffusers import AutoencoderKLLTX2Video

        self.source = Path(source)
        cfg = AutoencoderKLLTX2Video.load_config(self.source / "vae")
        self.temporal = cfg.get("temporal_compression_ratio") or 8
        self.vae = None

    def load(self, device, dtype):
        from diffusers import AutoencoderKLLTX2Video

        self.vae = AutoencoderKLLTX2Video.from_pretrained(self.source / "vae", dtype=dtype).to(device)
        self.vae.eval().requires_grad_(False)
        self.mean = self.vae.latents_mean.float().view(1, -1, 1, 1, 1)
        self.std = self.vae.latents_std.float().view(1, -1, 1, 1, 1)
        self.device, self.dtype = torch.device(device), dtype

    def frames(self, chunk, shape):
        return shape[2] * self.temporal - (self.temporal - 1 if chunk == 0 else 0)

    def decode_sequence(self, latent):
        """Decode the complete latent video, matching the non-causal pipeline VAE semantics."""
        if self.vae is None:
            raise RuntimeError("load() must be called before decode_sequence()")
        if latent.ndim != 5 or latent.shape[1] != self.mean.shape[1]:
            raise ValueError(f"LTX-2.5 sequence latents must have shape [B,{self.mean.shape[1]},T,H,W]")
        z = latent.float() * self.std / self.vae.config.scaling_factor + self.mean
        temb = torch.zeros(1, device=self.device, dtype=self.dtype) if self.vae.config.timestep_conditioning else None
        return self.vae.decode(z.to(self.dtype), temb, return_dict=False)[0].clamp(-1, 1)
