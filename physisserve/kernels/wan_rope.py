"""Opt-in exact Wan RoPE for dense BNHD and BHND-transposed BF16.

Four FP32 products and the following add/sub use explicit PTX rounding, so
the compiler cannot contract them into FMA. Head width is fixed at 128.
Triton is imported only when a supported NVIDIA CUDA invocation reaches the
kernel. CPU contract tests do not replace target-GPU exact-parity validation.

API reference: https://triton-lang.org/main/python-api/generated/
triton.language.inline_asm_elementwise.html
"""

_KERNEL = None


def _validate(x, cosine, sine):
    import torch

    if not all(isinstance(t, torch.Tensor) for t in (x, cosine, sine)):
        raise TypeError("RoPE inputs must be tensors")
    if any(t.layout != torch.strided for t in (x, cosine, sine)):
        raise ValueError("Wan RoPE requires strided tensors")
    if x.ndim != 4 or any(size <= 0 for size in x.shape) or x.shape[-1] != 128:
        raise ValueError("Wan RoPE requires nonempty [B,N,H,128] input")
    if x.dtype != torch.bfloat16:
        raise ValueError("Wan RoPE input must be BF16")
    if not (x.is_contiguous() or x.transpose(1, 2).is_contiguous()):
        raise ValueError("Wan RoPE supports dense BNHD or BHND-transposed input only")
    if x.numel() > 2**31 - 1:
        raise ValueError("Wan RoPE input exceeds the kernel indexing bound")
    for table in (cosine, sine):
        if table.ndim != 4 or table.shape[-1] != 128 or any(
            actual not in (1, expected) for actual, expected in zip(table.shape[:3], x.shape[:3])
        ):
            raise ValueError("RoPE tables must broadcast as [B,N,H,128]")
        if table.dtype != torch.float32:
            raise ValueError("RoPE tables must remain FP32")
        if table.device != x.device:
            raise ValueError("RoPE input and tables must share one CUDA device")
        if any(stride < 0 for stride in table.stride()):
            raise ValueError("RoPE table strides must be nonnegative")
        span = sum((size - 1) * stride for size, stride in zip(table.shape, table.stride()))
        if table.numel() > 2**31 - 1 or span > 2**31 - 1:
            raise ValueError("Wan RoPE tables exceed the kernel indexing bound")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (x, cosine, sine)):
        raise ValueError("fused Wan RoPE is inference-only")
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("fused Wan RoPE requires NVIDIA CUDA; no eager fallback")
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        raise ValueError("fused Wan RoPE requires native BF16 conversion on sm_80 or newer")


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def rotary_kernel(
        X, COS, SIN, OUT,
        N: tl.constexpr, H: tl.constexpr, PAIRS: tl.constexpr,
        KEY_LAYOUT: tl.constexpr,
        CB: tl.constexpr, CN: tl.constexpr, CH: tl.constexpr,
        CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr, CS3: tl.constexpr,
        SB: tl.constexpr, SN: tl.constexpr, SH: tl.constexpr,
        SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = pair < PAIRS
        # Walk physical adjacent BF16 pairs in either dense layout, preserving
        # coalesced input/output access without a full-history contiguous copy.
        row = pair // 64
        channel = (pair % 64) * 2
        batch = row // (N * H)
        if KEY_LAYOUT:
            token = row % N
            head = (row // N) % H
        else:
            token = (row // H) % N
            head = row % H
        cos_offset = ((batch if CB > 1 else 0) * CS0
                      + (token if CN > 1 else 0) * CS1
                      + (head if CH > 1 else 0) * CS2 + channel * CS3)
        sin_offset = ((batch if SB > 1 else 0) * SS0
                      + (token if SN > 1 else 0) * SS1
                      + (head if SH > 1 else 0) * SS2 + channel * SS3)
        even = tl.load(X + pair * 2, valid, 0).to(tl.float32)
        odd = tl.load(X + pair * 2 + 1, valid, 0).to(tl.float32)
        ce = tl.load(COS + cos_offset, valid, 0)
        co = tl.load(COS + cos_offset + CS3, valid, 0)
        se = tl.load(SIN + sin_offset, valid, 0)
        so = tl.load(SIN + sin_offset + SS3, valid, 0)
        real, imaginary = tl.inline_asm_elementwise(
            asm="""{
                .reg .f32 p0, p1, p2, p3;
                mul.rn.f32 p0, $2, $4;
                mul.rn.f32 p1, $3, $7;
                mul.rn.f32 p2, $2, $6;
                mul.rn.f32 p3, $3, $5;
                sub.rn.f32 $0, p0, p1;
                add.rn.f32 $1, p2, p3;
            }""",
            constraints="=f,=f,f,f,f,f,f,f",
            args=[even, odd, ce, co, se, so],
            dtype=(tl.float32, tl.float32), is_pure=True, pack=1,
        )
        tl.store(OUT + pair * 2, real.to(tl.bfloat16), valid)
        tl.store(OUT + pair * 2 + 1, imaginary.to(tl.bfloat16), valid)

    _KERNEL = rotary_kernel
    return _KERNEL


def fused_rope(x, cosine, sine):
    """Return exact-order RoPE with the input's dense layout; never fall back."""
    import torch

    _validate(x, cosine, sine)
    kernel = _get_kernel()
    out = torch.empty_like(x, memory_format=torch.preserve_format)
    if out.stride() != x.stride():
        raise ValueError("output allocation did not preserve the native input strides")
    pairs, block = x.numel() // 2, 256
    with torch.cuda.device(x.device):
        kernel[((pairs + block - 1) // block,)](
            x, cosine, sine, out,
            N=x.shape[1], H=x.shape[2], PAIRS=pairs,
            KEY_LAYOUT=not x.is_contiguous(),
            CB=cosine.shape[0], CN=cosine.shape[1], CH=cosine.shape[2],
            CS0=cosine.stride(0), CS1=cosine.stride(1),
            CS2=cosine.stride(2), CS3=cosine.stride(3),
            SB=sine.shape[0], SN=sine.shape[1], SH=sine.shape[2],
            SS0=sine.stride(0), SS1=sine.stride(1),
            SS2=sine.stride(2), SS3=sine.stride(3),
            BLOCK=block, num_warps=4, enable_fp_fusion=False,
        )
    return out
