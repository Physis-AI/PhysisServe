"""Opt-in H3 rotate-half kernel with the native BF16 rounding boundaries.

This is NOT Wan's adjacent-pair FP32 RoPE. H3 casts its tables to BF16, rounds
each product to BF16, then rounds their sum to BF16. The explicit PTX below
preserves those boundaries; there is no FP32/FMA replacement or eager fallback.
Input is logically BNHD with actual strides, including BHND-transposed and
token-subview storage. Output is a new contiguous BNHD tensor as in native H3.

The admitted head width is 128; batch size and strides remain independent of
model-executor batching support. CPU tests do not replace target-GPU parity.
References:
https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt
"""

_KERNEL = None
_INDEX_LIMIT = 2**31 - 1


def _validate(hidden_states, cos, sin):
    import torch

    if not all(isinstance(value, torch.Tensor) for value in (hidden_states, cos, sin)):
        raise TypeError("H3 RoPE inputs must be tensors")
    if any(value.layout != torch.strided for value in (hidden_states, cos, sin)):
        raise ValueError("H3 RoPE requires strided tensors")
    if hidden_states.ndim != 4 or any(size <= 0 for size in hidden_states.shape):
        raise ValueError("H3 RoPE requires nonempty [B,N,H,D] input")
    if hidden_states.shape[-1] != 128:
        raise ValueError("H3 RoPE requires the validated head_dim of 128")
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("H3 RoPE hidden states must be BF16")
    if cos.ndim != 2 or sin.ndim != 2 or cos.shape != sin.shape:
        raise ValueError("H3 RoPE cos/sin must have the same [N,rotary_dim] shape")
    rotary_dim = cos.shape[1]
    if cos.shape[0] != hidden_states.shape[1]:
        raise ValueError("H3 RoPE table token count must equal hidden-state token count")
    if rotary_dim % 2 or rotary_dim > hidden_states.shape[3]:
        raise ValueError("H3 RoPE rotary_dim must be even and no larger than head_dim")
    for table in (cos, sin):
        if table.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("H3 RoPE tables must be FP32 or BF16")
        if table.device != hidden_states.device:
            raise ValueError("H3 RoPE inputs must share one CUDA device")
    for value in (hidden_states, cos, sin):
        if any(stride < 0 for stride in value.stride()):
            raise ValueError("H3 RoPE strides must be nonnegative")
        # data_ptr already includes storage_offset: the kernel indexes relative
        # to that pointer, not the backing allocation's start. Large strided
        # views need this bound even when their logical numel is small.
        span = sum((size - 1) * stride for size, stride in zip(value.shape, value.stride()))
        if value.numel() > _INDEX_LIMIT or (value.numel() and span > _INDEX_LIMIT):
            raise ValueError("H3 RoPE view exceeds the kernel indexing bound")
    if torch.is_grad_enabled() and any(value.requires_grad for value in (hidden_states, cos, sin)):
        raise ValueError("fused H3 RoPE is inference-only")
    if hidden_states.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("fused H3 RoPE requires NVIDIA CUDA; no eager fallback")
    if torch.cuda.get_device_capability(hidden_states.device)[0] < 8:
        raise ValueError("fused H3 RoPE requires native BF16 conversion on sm_80 or newer")


def _get_kernel():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def rotary_kernel(
        X, COS, SIN, OUT,
        N: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
        ROTARY: tl.constexpr, ELEMENTS: tl.constexpr,
        XS0: tl.constexpr, XS1: tl.constexpr, XS2: tl.constexpr, XS3: tl.constexpr,
        CS0: tl.constexpr, CS1: tl.constexpr,
        SS0: tl.constexpr, SS1: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = index < ELEMENTS
        channel = index % D
        head = (index // D) % H
        token = (index // (D * H)) % N
        batch = index // (D * H * N)
        base = batch * XS0 + token * XS1 + head * XS2
        original = tl.load(X + base + channel * XS3, valid, 0)
        output = original
        if ROTARY > 0:
            rotating = valid & (channel < ROTARY)
            first_half = channel < ROTARY // 2
            partner_channel = tl.where(first_half, channel + ROTARY // 2, channel - ROTARY // 2)
            partner = tl.load(X + base + partner_channel * XS3, rotating, 0).to(tl.float32)
            # Negation of a finite BF16 value is exact, including signed zero.
            rotated = tl.where(first_half, -partner, partner)
            cosine = tl.load(COS + token * CS0 + channel * CS1, rotating, 0).to(tl.float32)
            sine = tl.load(SIN + token * SS0 + channel * SS1, rotating, 0).to(tl.float32)
            result = tl.inline_asm_elementwise(
                asm="""{
                    .reg .b16 cb, sb, pb, qb, rb;
                    .reg .f32 c, s, p, q, sum;
                    cvt.rn.bf16.f32 cb, $3;
                    cvt.rn.bf16.f32 sb, $4;
                    cvt.f32.bf16 c, cb;
                    cvt.f32.bf16 s, sb;
                    mul.rn.f32 p, $1, c;
                    cvt.rn.bf16.f32 pb, p;
                    cvt.f32.bf16 p, pb;
                    mul.rn.f32 q, $2, s;
                    cvt.rn.bf16.f32 qb, q;
                    cvt.f32.bf16 q, qb;
                    add.rn.f32 sum, p, q;
                    cvt.rn.bf16.f32 rb, sum;
                    cvt.f32.bf16 $0, rb;
                }""",
                constraints="=f,f,f,f,f",
                args=[original.to(tl.float32), rotated, cosine, sine],
                dtype=tl.float32, is_pure=True, pack=1,
            ).to(tl.bfloat16)
            # Keep pass-through channels in their original BF16 representation;
            # in particular, do not round-trip their NaNs through FP32.
            output = tl.where(channel < ROTARY, result, original)
        tl.store(OUT + index, output, valid)

    _KERNEL = rotary_kernel
    return _KERNEL


def fused_rope(hidden_states, cos, sin):
    """Leading rotate-half, native BF16 arithmetic, contiguous BNHD output."""
    import torch

    _validate(hidden_states, cos, sin)
    kernel = _get_kernel()
    output = torch.empty(hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)
    block, elements = 256, hidden_states.numel()
    with torch.cuda.device(hidden_states.device):
        kernel[((elements + block - 1) // block,)](
            hidden_states, cos, sin, output,
            N=hidden_states.shape[1], H=hidden_states.shape[2], D=hidden_states.shape[3],
            ROTARY=cos.shape[1], ELEMENTS=elements,
            XS0=hidden_states.stride(0), XS1=hidden_states.stride(1),
            XS2=hidden_states.stride(2), XS3=hidden_states.stride(3),
            CS0=cos.stride(0), CS1=cos.stride(1), SS0=sin.stride(0), SS1=sin.stride(1),
            BLOCK=block, num_warps=4, enable_fp_fusion=False,
        )
    return output
