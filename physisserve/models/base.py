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

    The directory may contain either one safetensors index or ordinary shards.
    Multiple indexes are ambiguous and rejected explicitly.
    """
    names = tuple(names)
    from safetensors import safe_open

    source = Path(source)
    indexes = sorted(source.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"multiple safetensors indexes found in {source}: {[path.name for path in indexes]}")
    if indexes:
        mapping = json.loads(indexes[0].read_text())["weight_map"]
        files = {}
        for name in names:
            if name in mapping:
                filename = mapping[name]
                relative = Path(filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"shard path must be relative to {source}: {filename!r}")
                files.setdefault(source / relative, []).append(name)
    else:
        files = {path: names for path in sorted(source.glob("*.safetensors"))}

    state = {}
    for path, wanted in files.items():
        with safe_open(path, framework="pt", device="cpu") as file:
            available = set(file.keys())
            for name in wanted:
                if name in available:
                    if name in state:
                        raise ValueError(f"duplicate tensor {name!r} across safetensors shards")
                    state[name] = file.get_tensor(name)
    if missing := set(names) - state.keys():
        raise ValueError(f"checkpoint is missing tensors: {sorted(missing)[:5]}")
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
