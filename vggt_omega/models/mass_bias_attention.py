"""Inference-only Triton attention with an exact per-key additive log mass."""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _mass_bias_attention_fwd(
        q_ptr, k_ptr, v_ptr, bias_ptr, out_ptr,
        stride_qb: tl.constexpr, stride_qh: tl.constexpr, stride_qn: tl.constexpr,
        stride_kb: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr,
        stride_vb: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr,
        stride_ob: tl.constexpr, stride_oh: tl.constexpr, stride_on: tl.constexpr,
        n_heads: tl.constexpr, n_query, n_key,
        scale_log2: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
        CAST_BF16: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch_head = tl.program_id(1)
        batch = batch_head // n_heads
        head = batch_head - batch * n_heads
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        dims = tl.arange(0, HEAD_DIM)
        q = tl.load(
            q_ptr + batch * stride_qb + head * stride_qh
            + rows[:, None] * stride_qn + dims[None, :],
            mask=rows[:, None] < n_query,
            other=0.0,
        )
        if CAST_BF16:
            q = q.to(tl.bfloat16)
        running_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        running_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
        for start_n in range(0, n_key, BLOCK_N):
            cols = start_n + tl.arange(0, BLOCK_N)
            k = tl.load(
                k_ptr + batch * stride_kb + head * stride_kh
                + cols[:, None] * stride_kn + dims[None, :],
                mask=cols[:, None] < n_key,
                other=0.0,
            )
            if CAST_BF16:
                k = k.to(tl.bfloat16)
            logits = tl.dot(q, tl.trans(k)) * scale_log2
            log_mass = tl.load(bias_ptr + cols, mask=cols < n_key, other=-float("inf"))
            logits += log_mass[None, :] * 1.4426950408889634
            logits = tl.where(cols[None, :] < n_key, logits, -float("inf"))
            block_max = tl.max(logits, axis=1)
            new_max = tl.maximum(running_max, block_max)
            probabilities = tl.exp2(logits - new_max[:, None])
            correction = tl.exp2(running_max - new_max)
            running_sum = running_sum * correction + tl.sum(probabilities, axis=1)
            accumulator *= correction[:, None]
            v = tl.load(
                v_ptr + batch * stride_vb + head * stride_vh
                + cols[:, None] * stride_vn + dims[None, :],
                mask=cols[:, None] < n_key,
                other=0.0,
            )
            if CAST_BF16:
                v = v.to(tl.bfloat16)
            accumulator += tl.dot(probabilities.to(v.dtype), v)
            running_max = new_max
        accumulator /= running_sum[:, None]
        tl.store(
            out_ptr + batch * stride_ob + head * stride_oh
            + rows[:, None] * stride_on + dims[None, :],
            accumulator,
            mask=rows[:, None] < n_query,
        )


def mass_bias_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                        log_key_mass: torch.Tensor) -> torch.Tensor:
    """Compute softmax(QK^T/sqrt(d) + log_key_mass)V exactly."""
    if triton is None:
        raise RuntimeError("Triton is unavailable")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("q, k and v must be [batch, heads, tokens, channels]")
    if q.shape[-1] != 64 or k.shape[-1] != 64 or v.shape[-1] != 64:
        raise ValueError("the initial fused kernel supports head_dim=64 only")
    if k.shape != v.shape or q.shape[:2] != k.shape[:2]:
        raise ValueError("incompatible q/k/v shapes")
    # Match torch SDPA autocast semantics.  Q/K normalization and RoPE produce
    # fp32 tensors in Omega, while SDPA itself is autocast to bf16.  A custom
    # Triton call is not autocast automatically, so perform that cast here.
    autocast_bf16 = bool(
        torch.is_autocast_enabled("cuda") and q.dtype == torch.float32
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    # Materialising compact bf16 Q/K/V is currently faster than loading the
    # fp32 interleaved tensors in the custom kernel.  It is also exactly the
    # dtype conversion performed internally by autocast SDPA.
    if autocast_bf16:
        q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)
    output_dtype = q.dtype
    bias = log_key_mass.reshape(-1).to(q.dtype)
    if bias.numel() != k.shape[-2]:
        raise ValueError("log_key_mass length does not match key tokens")
    output = torch.empty(q.shape, device=q.device, dtype=output_dtype)
    # RTX 4090 / head_dim=64: 64x32 keeps the fp32 online-softmax state in
    # registers while exposing enough query parallelism for very long clips.
    block_m, block_n = 64, 32
    grid = (triton.cdiv(q.shape[-2], block_m), q.shape[0] * q.shape[1])
    _mass_bias_attention_fwd[grid](
        q, k, v, bias, output,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        output.stride(0), output.stride(1), output.stride(2),
        q.shape[1], q.shape[-2], k.shape[-2],
        1.0 / math.sqrt(q.shape[-1]) * 1.4426950408889634,
        BLOCK_M=block_m, BLOCK_N=block_n, HEAD_DIM=64,
        CAST_BF16=False,
        num_warps=4, num_stages=3,
    )
    return output
