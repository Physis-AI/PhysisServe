"""Explicit, per-model RoPE selection shared by every scheduling strategy.

The native path remains the default. Opting into Triton never changes dtype,
layout, attention history or scheduling, and never silently falls back.
"""
import importlib
import importlib.util

import torch


def validate_rope_backend(backend):
    if not isinstance(backend, str) or backend not in ("native", "triton"):
        raise ValueError("rope_backend must be native or triton")
    return backend


def select_rope(backend, model):
    """Return the fused callable or None, without importing Triton on CPU."""
    validate_rope_backend(backend)
    if backend == "native":
        return None
    return importlib.import_module(f"physisserve.kernels.{model}_rope").fused_rope


def validate_rope_device(backend, device, dtype):
    """Reject an unsupported opt-in before loading model weights."""
    validate_rope_backend(backend)
    if backend == "native":
        return
    if torch.device(device).type != "cuda" or torch.version.hip is not None:
        raise ValueError("Triton RoPE requires NVIDIA CUDA; use rope_backend=native on CPU/ROCm")
    if dtype != torch.bfloat16:
        raise ValueError("Triton RoPE requires bfloat16; no precision conversion is performed")
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(device)[0] < 8:
        raise ValueError("Triton RoPE requires a CUDA GPU with compute capability 8.0 or newer")
    if importlib.util.find_spec("triton") is None:
        raise ImportError("rope_backend=triton requires the Triton version compatible with your CUDA PyTorch build")
