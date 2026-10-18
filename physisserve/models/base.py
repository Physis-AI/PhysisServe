"""Model contracts. A backend supplies three independent components that never need to share a rank:

- DiT: the denoising transformer, driven block by block by the model runner; KV goes through Cache.
- Encoder: turns a prompt into the condition tensors the DiT consumes (runs once per request).
- Decoder: turns finished latent chunks into pixel frames, keeping any temporal state across chunks.

``physisserve.models.wan``, ``ltx2`` and ``h3`` implement the family-specific components.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import torch


def load_selected_weights(source, names) -> dict[str, torch.Tensor]:
    """Load only ``names`` from a Diffusers safetensors directory.

    Read the indexed shards or the single unsharded safetensors file.
    """
    names = tuple(names)
    from safetensors import safe_open

    source = Path(source)
    indexes = list(source.glob("*.safetensors.index.json"))
    if indexes:
        index, = indexes
        mapping = json.loads(index.read_text())["weight_map"]
        files = {}
        for name in names:
            relative = Path(mapping[name])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"shard path must be relative to {source}: {str(relative)!r}")
            files.setdefault(source / relative, []).append(name)
    else:
        path, = source.glob("*.safetensors")
        files = {path: names}

    state = {}
    for path, wanted in files.items():
        with safe_open(path, framework="pt", device="cpu") as file:
            for name in wanted:
                state[name] = file.get_tensor(name)
    return state


def drop_attributes(module, names) -> None:
    for name in names:
        if hasattr(module, name):
            delattr(module, name)


@dataclass(frozen=True)
class Backend:
    """The transformer, prompt encoder and video decoder of one model family."""

    dit: object
    encoder: object
    decoder: object
