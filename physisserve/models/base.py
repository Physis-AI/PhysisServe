"""Model contracts. A backend supplies three independent components that never need to share a rank:

- DiT: the denoising transformer, driven block by block by the model runner; KV goes through Cache.
- Encoder: turns a prompt into the condition tensors the DiT consumes (runs once per request).
- Decoder: turns finished latent chunks into pixel frames, keeping any temporal state across chunks.

``physisserve.models.wan``, ``ltx2`` and ``h3`` implement the family-specific components.
"""
import json
from collections.abc import Iterable
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
    if not source.is_dir():
        raise ValueError(f"transformer checkpoint must be a safetensors directory: {source}")
    indexes = sorted(source.glob("*.safetensors.index.json"))
    if len(indexes) > 1:
        raise ValueError(f"multiple safetensors indexes found in {source}: {[path.name for path in indexes]}")
    if indexes:
        try:
            mapping = json.loads(indexes[0].read_text())["weight_map"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid safetensors index {indexes[0]}") from error
        if not isinstance(mapping, dict):
            raise ValueError(f"invalid safetensors index {indexes[0]}: weight_map must be an object")
        files = {}
        for name in names:
            if name in mapping:
                filename = mapping[name]
                if not isinstance(filename, str) or not filename:
                    raise ValueError(f"invalid shard path for tensor {name!r} in {indexes[0]}")
                relative = Path(filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"shard path must be relative to {source}: {filename!r}")
                files.setdefault(source / relative, []).append(name)
    else:
        shards = sorted(source.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"no safetensors weights found in {source}")
        files = {path: names for path in shards}

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


def block_indices(blocks: Iterable[int], num_blocks: int) -> tuple[int, ...]:
    """Materialize the exact owned block set, including a noncontiguous one-shot iterable."""
    blocks = tuple(blocks)
    if any(type(index) is not int or not 0 <= index < num_blocks for index in blocks):
        raise ValueError("owned block indices must be integers in [0, num_blocks)")
    if len(set(blocks)) != len(blocks):
        raise ValueError("owned block indices must be distinct")
    return blocks


@dataclass(frozen=True)
class Backend:
    """The transformer, prompt encoder and video decoder of one model family."""

    dit: object
    encoder: object
    decoder: object
