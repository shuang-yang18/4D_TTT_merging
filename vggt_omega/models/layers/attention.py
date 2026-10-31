# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import List, Tuple
import atexit
import json
import os
from pathlib import Path

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from .utils import cat_keep_shapes, uncat_with_shapes


_ATTENTION_PROFILE = os.environ.get("VGGT_PROFILE_ATTENTION_DETAIL", "0") == "1"
_ATTENTION_PROFILE_EVENTS: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
_Q_COMPRESSION_DIAGNOSTICS: list[dict[str, float | int]] = []


def _attention_profile_begin():
    if not _ATTENTION_PROFILE or not torch.cuda.is_available():
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _attention_profile_end(label, begin):
    if begin is None:
        return
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    _ATTENTION_PROFILE_EVENTS.append((label, begin, end))


def _print_attention_profile():
    if not _ATTENTION_PROFILE_EVENTS:
        return
    torch.cuda.synchronize()
    totals = {}
    for label, begin, end in _ATTENTION_PROFILE_EVENTS:
        values = totals.setdefault(label, [0.0, 0])
        values[0] += begin.elapsed_time(end)
        values[1] += 1
    for label, (elapsed, calls) in sorted(totals.items()):
        print(
            f"[attention-detail-profile] {label} total_ms={elapsed:.3f} "
            f"calls={calls} mean_ms={elapsed / calls:.3f}", flush=True,
        )


if _ATTENTION_PROFILE:
    atexit.register(_print_attention_profile)


def _save_q_compression_diagnostics():
    path = os.environ.get("VGGT_Q_COMPRESSION_DIAGNOSTIC_PATH")
    if not path or not _Q_COMPRESSION_DIAGNOSTICS:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"layers": _Q_COMPRESSION_DIAGNOSTICS}, indent=2) + "\n",
        encoding="utf-8",
    )


if os.environ.get("VGGT_Q_COMPRESSION_DIAGNOSTIC_PATH"):
    atexit.register(_save_q_compression_diagnostics)


@torch.no_grad()
def _diagnose_compact_query(attention, dense_output, dense_q, compact_k, compact_v,
                            compact_bias, token_plan):
    """Compare full-Q output with the paired compact-Q quotient in one layer."""
    ordinal = getattr(attention, "q_compression_diagnostic_ordinal", None)
    requested = {
        int(value) for value in os.environ.get(
            "VGGT_Q_COMPRESSION_DIAGNOSTIC_LAYERS", "8,10,11,12",
        ).split(",") if value.strip()
    }
    if ordinal is None or ordinal not in requested or not token_plan.preserve_queries:
        return
    batch, heads, original_tokens, head_dim = dense_q.shape
    channels = heads * head_dim
    dense_q_flat = dense_q.transpose(1, 2).reshape(batch, original_tokens, channels)
    compact_q_flat = token_plan.compact(dense_q_flat)
    compact_q = compact_q_flat.reshape(
        batch, token_plan.kept_tokens, heads, head_dim,
    ).transpose(1, 2)
    from vggt_omega.models.mass_bias_attention import mass_bias_attention
    compact_output = mass_bias_attention(compact_q, compact_k, compact_v, compact_bias)
    compact_output_flat = compact_output.transpose(1, 2).reshape(
        batch, token_plan.kept_tokens, channels,
    )

    def restore(compact_flat):
        result = compact_flat.new_zeros((batch, original_tokens, channels))
        result[:, token_plan.keep[0]] = compact_flat
        if token_plan.source_original.shape[1]:
            parent = compact_flat[:, token_plan.source_parent_compact[0]]
            represented = (
                parent * token_plan.source_restore_weights[..., None].to(compact_flat.dtype)
            ).sum(dim=2)
            result[:, token_plan.source_original[0]] = represented.to(compact_flat.dtype)
        return result

    approximate = restore(compact_output_flat)
    full = dense_output.transpose(1, 2).reshape(batch, original_tokens, channels)
    # Camera/register tokens remain exact in both paths.
    if token_plan.special_query_original is not None:
        special = token_plan.special_query_original
        approximate[:, special] = full[:, special]
    restored_q = restore(compact_q_flat)
    source = token_plan.source_original[0]
    if source.numel() > 20000:
        source = source[torch.linspace(
            0, source.numel() - 1, 20000, device=source.device,
        ).round().long()]
    full = attention.proj(full[:, source]).float()
    approximate = attention.proj(approximate[:, source]).float()
    q_residual = (dense_q_flat[:, source] - restored_q[:, source]).float()
    difference = full - approximate
    reference_rms = full.square().mean().sqrt().clamp_min(1e-12)

    global_scale = (full * approximate).sum() / approximate.square().sum().clamp_min(1e-12)
    scaled_difference = full - global_scale * approximate
    channel_scale = (full * approximate).sum(dim=(0, 1)) / approximate.square().sum(
        dim=(0, 1),
    ).clamp_min(1e-12)
    channel_difference = full - approximate * channel_scale
    beta = (difference * q_residual).sum() / q_residual.square().sum().clamp_min(1e-12)
    q_scalar_difference = difference - beta * q_residual
    beta_channel = (difference * q_residual).sum(dim=(0, 1)) / q_residual.square().sum(
        dim=(0, 1),
    ).clamp_min(1e-12)
    q_channel_difference = difference - q_residual * beta_channel
    cosine = torch.nn.functional.cosine_similarity(full, approximate, dim=-1).mean()
    q_error = q_residual.square().mean(dim=-1).sqrt().flatten()
    update_error = difference.square().mean(dim=-1).sqrt().flatten()
    q_centered, update_centered = q_error - q_error.mean(), update_error - update_error.mean()
    correlation = (q_centered * update_centered).mean() / (
        q_centered.square().mean().sqrt() * update_centered.square().mean().sqrt()
    ).clamp_min(1e-12)
    _Q_COMPRESSION_DIAGNOSTICS.append({
        "global_ordinal": int(ordinal),
        "sampled_source_tokens": int(source.numel()),
        "kept_tokens": int(token_plan.kept_tokens),
        "original_tokens": int(original_tokens),
        "projected_update_relative_rms": float(difference.square().mean().sqrt() / reference_rms),
        "projected_update_cosine": float(cosine),
        "global_scale": float(global_scale),
        "after_global_scale_relative_rms": float(scaled_difference.square().mean().sqrt() / reference_rms),
        "after_channel_scale_relative_rms": float(channel_difference.square().mean().sqrt() / reference_rms),
        "q_reconstruction_relative_rms": float(
            q_residual.square().mean().sqrt()
            / dense_q_flat[:, source].float().square().mean().sqrt().clamp_min(1e-12)
        ),
        "q_error_update_error_correlation": float(correlation),
        "q_scalar_beta": float(beta),
        "after_q_scalar_compensation_relative_rms": float(
            q_scalar_difference.square().mean().sqrt() / reference_rms
        ),
        "after_q_channel_compensation_relative_rms": float(
            q_channel_difference.square().mean().sqrt() / reference_rms
        ),
    })


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            bias_mask = torch.ones_like(self.bias)
            bias_mask[o // 3 : 2 * o // 3].fill_(0)
            self.register_buffer("bias_mask", bias_mask)

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        use_qk_norm: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        # VGGT-Omega change: the aggregator checkpoint was trained with Q/K
        # normalization, while upstream DINOv3 attention does not expose it.
        self.use_qk_norm = use_qk_norm
        if self.use_qk_norm:
            self.q_norm = nn.LayerNorm(head_dim, eps=1e-5)
            self.k_norm = nn.LayerNorm(head_dim, eps=1e-5)
        else:
            self.q_norm = None
            self.k_norm = None

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None, token_plan=None) -> Tensor:
        if token_plan is not None and self.training:
            raise RuntimeError("token compaction is inference-only")
        fixed_config = getattr(self, "fixed_update_operator_config", None)
        if fixed_config is not None:
            if token_plan is not None or rope is not None or self.training:
                raise RuntimeError("fixed update operator requires plain inference global attention")
            from vggt_omega.models.sampled_update_operator import fixed_update_attention
            return fixed_update_attention(self, x, fixed_config)
        sampled_config = getattr(self, "sampled_update_operator_config", None)
        if sampled_config is not None:
            if token_plan is not None or rope is not None or self.training:
                raise RuntimeError("sampled update operator requires plain inference global attention")
            from vggt_omega.models.sampled_update_operator import sampled_update_attention
            return sampled_update_attention(self, x, sampled_config)
        dense_diagnostic = getattr(self, "dense_update_diagnostic_config", None)
        trajectory_probe = getattr(self, "baseline_trajectory_probe_config", None)
        source = x if dense_diagnostic is not None or trajectory_probe is not None else None
        profile = _attention_profile_begin() if token_plan is not None else None
        qkv = (
            token_plan.precomputed_qkv
            if token_plan is not None and getattr(token_plan, "precomputed_qkv", None) is not None
            else self.qkv(x)
        )
        _attention_profile_end("global.qkv_projection", profile)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope, token_plan=token_plan)
        profile = _attention_profile_begin() if token_plan is not None else None
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        _attention_profile_end("global.output_projection", profile)
        if trajectory_probe is not None:
            from vggt_omega.models.sampled_update_operator import capture_baseline_trajectory_probe
            capture_baseline_trajectory_probe(source, x, trajectory_probe)
        if dense_diagnostic is not None:
            from vggt_omega.models.sampled_update_operator import fit_dense_update_diagnostic
            fit_dense_update_diagnostic(source, x, dense_diagnostic)
        if token_plan is None:
            return x
        profile = _attention_profile_begin()
        restored = token_plan.restore(x)
        if getattr(token_plan, "source_query_correction", False) and not getattr(token_plan, "preserve_queries", False):
            source_output = token_plan.source_attention_output
            if source_output is None:
                raise RuntimeError("source-query correction requested without source attention output")
            corrected = self.proj(source_output)
            corrected = self.proj_drop(corrected)
            restored.index_copy_(1, token_plan.source_original[0], corrected)
            # Avoid retaining an attention-sized tensor beyond this block.
            token_plan.source_attention_output = None
        _attention_profile_end("global.update_restore", profile)
        return restored

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None, token_plan=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        profile = _attention_profile_begin() if token_plan is not None else None
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        _attention_profile_end("global.qk_norm_rope", profile)
        # Exact Any4D token-plan semantics: compact Q, K and V, run attention
        # exclusively on compact tokens, project, then broadcast the compact
        # update to every source token through ``restore`` in forward().
        if token_plan is not None:
            profile = _attention_profile_begin()
            dense_q, dense_k, dense_v = q, k, v
            def pack(value: Tensor) -> Tensor:
                value = token_plan.compact(
                    value.transpose(1, 2).reshape(B, N, C), merge=token_plan.pool_sources,
                )
                return value.reshape(B, token_plan.kept_tokens, self.num_heads, C // self.num_heads).transpose(1, 2)
            if getattr(token_plan, "preserve_queries", False):
                # Asymmetric attention: every original token keeps its own
                # query/output while only the scene-memory K/V bank is
                # compacted.  A dedicated key path permits a physical-dst key
                # with pooled values, without changing generic plan semantics.
                key = k.transpose(1, 2).reshape(B, N, C)
                if hasattr(token_plan, "compact_key"):
                    key = token_plan.compact_key(key)
                else:
                    key = token_plan.compact(key, merge=token_plan.pool_sources)
                k = key.reshape(B, token_plan.kept_tokens, self.num_heads, C // self.num_heads).transpose(1, 2)
                v = pack(v)
            else:
                query = q.transpose(1, 2).reshape(B, N, C)
                if hasattr(token_plan, "compact_query"):
                    query = token_plan.compact_query(query)
                    query_tokens = token_plan.query_kept_tokens
                    q = query.reshape(B, query_tokens, self.num_heads, C // self.num_heads).transpose(1, 2)
                    k, v = pack(k), pack(v)
                else:
                    q, k, v = pack(q), pack(k), pack(v)
            _attention_profile_end("global.qkv_compaction", profile)
        source_attention_output = None
        if (token_plan is not None and getattr(token_plan, "source_query_correction", False)
                and not getattr(token_plan, "preserve_queries", False)):
            profile = _attention_profile_begin()
            source_query = dense_q.index_select(2, token_plan.source_original[0])
            source_attention_output = torch.nn.functional.scaled_dot_product_attention(source_query, k, v)
            _attention_profile_end(
                f"global.source_correction_sdpa.q{source_query.shape[-2]}.k{k.shape[-2]}",
                profile,
            )
        compact_bias = None
        if token_plan is not None and getattr(token_plan, "key_log_mass", None) is not None:
            compact_bias = token_plan.key_log_mass[:, None, None, :].to(device=q.device, dtype=q.dtype)
        profile = _attention_profile_begin() if token_plan is not None else None
        use_fused_mass_bias = (
            compact_bias is not None
            and os.environ.get("VGGT_TRITON_MASS_BIAS_ATTENTION", "1") == "1"
            and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and q.shape[-1] == 64
        )
        if use_fused_mass_bias:
            from vggt_omega.models.mass_bias_attention import mass_bias_attention
            x = mass_bias_attention(q, k, v, compact_bias)
        else:
            x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=compact_bias)
        _attention_profile_end(
            f"global.main_sdpa.q{q.shape[-2]}.k{k.shape[-2]}.bias{int(compact_bias is not None)}"
            f".fused{int(use_fused_mass_bias)}.dtype{str(q.dtype).split('.')[-1]}",
            profile,
        )
        if token_plan is not None and getattr(token_plan, "dense_special_queries", False):
            profile = _attention_profile_begin()
            original = token_plan.special_query_original
            output_indices = (
                original if getattr(token_plan, "preserve_queries", False)
                else token_plan.special_query_compact
            )
            if original is None or output_indices is None:
                raise RuntimeError("dense-special attention plan has no special-token mapping")
            exact_special = torch.nn.functional.scaled_dot_product_attention(
                dense_q.index_select(2, original), dense_k, dense_v,
            )
            # Replace only the small camera/register query set.  Patch-query
            # attention remains compact and therefore retains the speedup.
            x.index_copy_(2, output_indices, exact_special)
            _attention_profile_end(
                f"global.special_sdpa.q{original.numel()}.k{dense_k.shape[-2]}", profile,
            )
        if token_plan is not None and os.environ.get("VGGT_Q_COMPRESSION_DIAGNOSTIC_PATH"):
            _diagnose_compact_query(
                self, x, dense_q, k, v, compact_bias, token_plan,
            )
        if source_attention_output is not None:
            token_plan.source_attention_output = source_attention_output.transpose(1, 2).reshape(
                B, token_plan.source_original.shape[1], C,
            )
        x = x.transpose(1, 2)
        output_tokens = (
            N if token_plan is not None and getattr(token_plan, "preserve_queries", False)
            else token_plan.query_kept_tokens if token_plan is not None and hasattr(token_plan, "query_kept_tokens")
            else token_plan.kept_tokens if token_plan is not None else N
        )
        x = x.reshape([B, output_tokens, C])
        return x


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
