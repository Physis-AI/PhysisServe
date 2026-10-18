"""Opt-in LTX split RoPE for contiguous BF16 [B,N,H*128] packets.

The native function separately computes ``split_x.float() * cos`` before
FP32 addcmul_. Keep those FP32 operation boundaries with explicit rounded PTX
products and sums; do not replace the expression with a single FMA. PyTorch
2.8 CUDA addcmul is ``a + alpha * (b * c)`` with runtime alpha=1:
https://github.com/pytorch/pytorch/blob/v2.8.0/aten/src/ATen/native/cuda/PointwiseOpsKernel.cu
The target GPU's actual native function remains the mandatory exact oracle.

Only the split 3D path is admitted. Tables remain FP32, including sliced
noncontiguous tables. Output is contiguous BF16 as in the native 3D path.
No CPU, interleaved, 4D, reduced-precision or numerical fallback exists.
"""

_KERNEL = None


def _validate(x, freqs):
    import torch

    if not isinstance(freqs, (tuple, list)) or len(freqs) != 2:
        raise ValueError("split RoPE freqs must be a (cosine, sine) pair")
    cosine, sine = freqs
    if not all(isinstance(value, torch.Tensor) for value in (x, cosine, sine)):
        raise TypeError("split RoPE inputs must be tensors")
    if any(value.layout != torch.strided for value in (x, cosine, sine)):
        raise ValueError("LTX split RoPE requires strided tensors")
    if x.ndim != 3 or any(size <= 0 for size in x.shape) or x.shape[-1] % 128:
        raise ValueError("LTX split RoPE requires nonempty [B,N,H*128] input")
    if x.dtype != torch.bfloat16:
        raise ValueError("LTX split RoPE input must be BF16")
    if not x.is_contiguous():
        raise ValueError("LTX split RoPE input must be contiguous")
    if x.numel() > 2**31 - 1:
        raise ValueError("LTX split RoPE input exceeds the indexing bound")
    batch, tokens, width = x.shape
    expected = (batch, width // 128, tokens, 64)
    for table in (cosine, sine):
        if tuple(table.shape) != expected:
            raise ValueError("LTX split RoPE tables must have exact [B,H,N,64] shape")
        if table.dtype != torch.float32:
            raise ValueError("LTX split RoPE tables must remain FP32")
        if table.device != x.device:
            raise ValueError("LTX split RoPE inputs must share one CUDA device")
        if any(stride < 0 for stride in table.stride()):
            raise ValueError("LTX split RoPE table strides must be nonnegative")
        if sum((size - 1) * stride for size, stride in zip(table.shape, table.stride())) > 2**31 - 1:
            raise ValueError("LTX split RoPE tables exceed the indexing bound")
    if torch.is_grad_enabled() and any(value.requires_grad for value in (x, cosine, sine)):
        raise ValueError("fused LTX split RoPE is inference-only")
    if x.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("fused LTX split RoPE requires NVIDIA CUDA; no eager fallback")
    if torch.cuda.get_device_capability(x.device)[0] < 8:
        raise ValueError("fused LTX split RoPE requires native BF16 conversion on sm_80 or newer")
    return cosine, sine


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def split_rotary_kernel(
        X, COS, SIN, OUT,
        N: tl.constexpr, H: tl.constexpr, PAIRS: tl.constexpr,
        CS0: tl.constexpr, CS1: tl.constexpr, CS2: tl.constexpr, CS3: tl.constexpr,
        SS0: tl.constexpr, SS1: tl.constexpr, SS2: tl.constexpr, SS3: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = pair < PAIRS
        channel = pair % 64
        head = (pair // 64) % H
        token = (pair // (64 * H)) % N
        batch = pair // (64 * H * N)
        base = ((batch * N + token) * H + head) * 128 + channel
        cos_offset = batch * CS0 + head * CS1 + token * CS2 + channel * CS3
        sin_offset = batch * SS0 + head * SS1 + token * SS2 + channel * SS3
        first = tl.load(X + base, valid, 0).to(tl.float32)
        second = tl.load(X + base + 64, valid, 0).to(tl.float32)
        cosine = tl.load(COS + cos_offset, valid, 0)
        sine = tl.load(SIN + sin_offset, valid, 0)
        first_out, second_out = tl.inline_asm_elementwise(
            asm="""{
                .reg .f32 p0, p1, p2, p3, negative_sine;
                mul.rn.f32 p0, $2, $4;
                mul.rn.f32 p1, $3, $4;
                neg.f32 negative_sine, $5;
                mul.rn.f32 p2, negative_sine, $3;
                mul.rn.f32 p3, $5, $2;
                add.rn.f32 $0, p0, p2;
                add.rn.f32 $1, p1, p3;
            }""",
            constraints="=f,=f,f,f,f,f",
            args=[first, second, cosine, sine],
            dtype=(tl.float32, tl.float32), is_pure=True, pack=1,
        )
        tl.store(OUT + base, first_out.to(tl.bfloat16), valid)
        tl.store(OUT + base + 64, second_out.to(tl.bfloat16), valid)

    _KERNEL = split_rotary_kernel
    return _KERNEL


def fused_rope(x, freqs):
    """Native two-argument split interface with strict guards and no fallback."""
    import torch

    cosine, sine = _validate(x, freqs)
    kernel = _get_kernel()
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    pairs, block = x.numel() // 2, 256
    with torch.cuda.device(x.device):
        kernel[((pairs + block - 1) // block,)](
            x, cosine, sine, output,
            N=x.shape[1], H=x.shape[2] // 128, PAIRS=pairs,
            CS0=cosine.stride(0), CS1=cosine.stride(1), CS2=cosine.stride(2), CS3=cosine.stride(3),
            SS0=sine.stride(0), SS1=sine.stride(1), SS2=sine.stride(2), SS3=sine.stride(3),
            BLOCK=block, num_warps=4, enable_fp_fusion=False,
        )
    return output
