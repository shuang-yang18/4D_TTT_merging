"""Optional Triton kernels for flow-free token-matching TTT.

Every public wrapper has a PyTorch caller-side fallback.  This module never
changes an algorithmic threshold or an output layout; it only fuses launch
heavy, fixed-shape CUDA work.
"""
from __future__ import annotations

import os

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU/development installations
    triton = None
    tl = None


def triton_ready(value: torch.Tensor) -> bool:
    return triton is not None and value.is_cuda


if triton is not None:
    @triton.jit
    def _row_l2_normalize_kernel(
        input_ptr, output_ptr, rows, columns: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        column = tl.arange(0, BLOCK_C)
        mask = (row[:, None] < rows) & (column[None, :] < columns)
        value = tl.load(
            input_ptr + row[:, None] * columns + column[None, :],
            mask=mask, other=0.0,
        ).to(tl.float32)
        inverse = tl.rsqrt(tl.maximum(tl.sum(value * value, axis=1), 1.0e-12))
        tl.store(
            output_ptr + row[:, None] * columns + column[None, :],
            value * inverse[:, None], mask=mask,
        )


    @triton.jit
    def _local_top2_match_kernel(
        source_ptr, target_ptr, best_ptr, second_ptr, dy_ptr, dx_ptr,
        rows, height: tl.constexpr, width: tl.constexpr,
        channels: tl.constexpr, radius_y: tl.constexpr, radius_x: tl.constexpr,
        reverse_order: tl.constexpr, BLOCK_C: tl.constexpr,
    ):
        row = tl.program_id(0)
        valid_row = row < rows
        patch_count = height * width
        pair = row // patch_count
        patch = row - pair * patch_count
        y = patch // width
        x = patch - y * width
        channel = tl.arange(0, BLOCK_C)
        channel_mask = channel < channels
        source = tl.load(
            source_ptr + row * channels + channel,
            mask=valid_row & channel_mask, other=0.0,
        ).to(tl.float32)
        best = -float("inf")
        second = -float("inf")
        best_y = 0
        best_x = 0
        span_x: tl.constexpr = 2 * radius_x + 1
        candidates: tl.constexpr = (2 * radius_y + 1) * span_x
        for candidate in tl.static_range(0, candidates):
            cy = candidate // span_x
            cx = candidate - cy * span_x
            if reverse_order:
                oy = radius_y - cy
                ox = radius_x - cx
            else:
                oy = cy - radius_y
                ox = cx - radius_x
            target_y = y + oy
            target_x = x + ox
            valid_target = (
                valid_row & (target_y >= 0) & (target_y < height)
                & (target_x >= 0) & (target_x < width)
            )
            target_row = pair * patch_count + target_y * width + target_x
            target = tl.load(
                target_ptr + target_row * channels + channel,
                mask=valid_target & channel_mask, other=0.0,
            ).to(tl.float32)
            similarity = tl.sum(source * target, axis=0)
            similarity = tl.where(valid_target, similarity, -float("inf"))
            better = similarity > best
            second = tl.where(better, best, tl.maximum(second, similarity))
            best = tl.where(better, similarity, best)
            best_y = tl.where(better, oy, best_y)
            best_x = tl.where(better, ox, best_x)
        tl.store(best_ptr + row, best, mask=valid_row)
        tl.store(second_ptr + row, second, mask=valid_row)
        tl.store(dy_ptr + row, best_y, mask=valid_row)
        tl.store(dx_ptr + row, best_x, mask=valid_row)


    @triton.jit
    def _integer_gather_warp_kernel(
        field_ptr, flow_ptr, output_ptr, elements, height: tl.constexpr,
        width: tl.constexpr, channels: tl.constexpr, BLOCK: tl.constexpr,
    ):
        element = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid_element = element < elements
        row = element // channels
        channel = element - row * channels
        patch_count = height * width
        batch = row // patch_count
        patch = row - batch * patch_count
        y = patch // width
        x = patch - y * width
        flow_x = tl.load(flow_ptr + row * 2, mask=valid_element, other=0.0).to(tl.int32)
        flow_y = tl.load(flow_ptr + row * 2 + 1, mask=valid_element, other=0.0).to(tl.int32)
        sample_x = tl.minimum(tl.maximum(x + flow_x, 0), width - 1)
        sample_y = tl.minimum(tl.maximum(y + flow_y, 0), height - 1)
        source_row = batch * patch_count + sample_y * width + sample_x
        value = tl.load(
            field_ptr + source_row * channels + channel,
            mask=valid_element, other=0.0,
        )
        tl.store(output_ptr + element, value, mask=valid_element)


    @triton.jit
    def _motion_cues_kernel(
        forward_ptr, backward_ptr, predicted_ptr, forward_conf_ptr,
        backward_conf_ptr, residual_ptr, confidence_ptr, boundary_ptr,
        rows, frames: tl.constexpr, height: tl.constexpr, width: tl.constexpr,
    ):
        row = tl.program_id(0)
        valid_row = row < rows
        patch_count = height * width
        frame = row // patch_count
        patch = row - frame * patch_count
        y = patch // width
        x = patch - y * width
        has_forward = valid_row & (frame + 1 < frames)
        has_backward = valid_row & (frame > 0)
        forward_row = frame * patch_count + patch
        backward_row = (frame - 1) * patch_count + patch
        fdx = tl.load(forward_ptr + forward_row * 2, mask=has_forward, other=0.0).to(tl.float32)
        fdy = tl.load(forward_ptr + forward_row * 2 + 1, mask=has_forward, other=0.0).to(tl.float32)
        bdx = tl.load(backward_ptr + backward_row * 2, mask=has_backward, other=0.0).to(tl.float32)
        bdy = tl.load(backward_ptr + backward_row * 2 + 1, mask=has_backward, other=0.0).to(tl.float32)
        pfdx = tl.load(predicted_ptr + row * 4, mask=has_forward, other=0.0).to(tl.float32)
        pfdy = tl.load(predicted_ptr + row * 4 + 1, mask=has_forward, other=0.0).to(tl.float32)
        pbdx = tl.load(predicted_ptr + row * 4 + 2, mask=has_backward, other=0.0).to(tl.float32)
        pbdy = tl.load(predicted_ptr + row * 4 + 3, mask=has_backward, other=0.0).to(tl.float32)
        f_residual = tl.sqrt((fdx - pfdx) * (fdx - pfdx) + (fdy - pfdy) * (fdy - pfdy))
        b_residual = tl.sqrt((bdx - pbdx) * (bdx - pbdx) + (bdy - pbdy) * (bdy - pbdy))
        count = has_forward.to(tl.float32) + has_backward.to(tl.float32)

        f_mean_x = 0.0
        f_mean_y = 0.0
        b_mean_x = 0.0
        b_mean_y = 0.0
        for oy in tl.static_range(-1, 2):
            for ox in tl.static_range(-1, 2):
                neighbor_y = y + oy
                neighbor_x = x + ox
                inside = (
                    (neighbor_y >= 0) & (neighbor_y < height)
                    & (neighbor_x >= 0) & (neighbor_x < width)
                )
                neighbor_patch = neighbor_y * width + neighbor_x
                f_neighbor = frame * patch_count + neighbor_patch
                b_neighbor = (frame - 1) * patch_count + neighbor_patch
                f_mean_x += tl.load(
                    forward_ptr + f_neighbor * 2,
                    mask=has_forward & inside, other=0.0,
                ).to(tl.float32) / 9.0
                f_mean_y += tl.load(
                    forward_ptr + f_neighbor * 2 + 1,
                    mask=has_forward & inside, other=0.0,
                ).to(tl.float32) / 9.0
                b_mean_x += tl.load(
                    backward_ptr + b_neighbor * 2,
                    mask=has_backward & inside, other=0.0,
                ).to(tl.float32) / 9.0
                b_mean_y += tl.load(
                    backward_ptr + b_neighbor * 2 + 1,
                    mask=has_backward & inside, other=0.0,
                ).to(tl.float32) / 9.0
        f_boundary = tl.sqrt((fdx - f_mean_x) * (fdx - f_mean_x) + (fdy - f_mean_y) * (fdy - f_mean_y))
        b_boundary = tl.sqrt((bdx - b_mean_x) * (bdx - b_mean_x) + (bdy - b_mean_y) * (bdy - b_mean_y))
        f_conf = tl.load(forward_conf_ptr + forward_row, mask=has_forward, other=1.0).to(tl.float32)
        b_conf = tl.load(backward_conf_ptr + backward_row, mask=has_backward, other=1.0).to(tl.float32)
        tl.store(residual_ptr + row, (f_residual + b_residual) / tl.maximum(count, 1.0), mask=valid_row)
        tl.store(confidence_ptr + row, tl.minimum(f_conf, b_conf), mask=valid_row)
        tl.store(boundary_ptr + row, (f_boundary + b_boundary) / tl.maximum(count, 1.0), mask=valid_row)


    @triton.jit
    def _sensitivity_labels_kernel(
        residual_ptr, confidence_ptr, boundary_ptr, sensitivity_ptr,
        low_confidence_ptr, raw_event_ptr, rows, patch_count: tl.constexpr,
        uncertain_confidence: tl.constexpr, residual_scale, boundary_scale,
        residual_threshold, boundary_threshold,
    ):
        row = tl.program_id(0)
        valid = row < rows
        frame = row // patch_count
        later = valid & (frame > 0)
        residual = tl.load(residual_ptr + row, mask=valid, other=0.0).to(tl.float32)
        confidence = tl.load(confidence_ptr + row, mask=valid, other=1.0).to(tl.float32)
        boundary = tl.load(boundary_ptr + row, mask=valid, other=0.0).to(tl.float32)
        residual_scale_value = tl.load(residual_scale).to(tl.float32)
        boundary_scale_value = tl.load(boundary_scale).to(tl.float32)
        residual_threshold_value = tl.load(residual_threshold).to(tl.float32)
        boundary_threshold_value = tl.load(boundary_threshold).to(tl.float32)
        novelty = 1.0 - confidence
        residual_unit = tl.minimum(tl.maximum(residual / tl.maximum(residual_scale_value, 1.0e-6), 0.0), 1.0)
        boundary_unit = tl.minimum(tl.maximum(boundary / tl.maximum(boundary_scale_value, 1.0e-6), 0.0), 1.0)
        sensitivity = tl.minimum(tl.maximum(
            0.45 * novelty + 0.35 * residual_unit + 0.20 * boundary_unit, 0.0,
        ), 1.0)
        low = later & (confidence < uncertain_confidence)
        event = later & (
            ((residual >= residual_threshold_value) & (confidence >= uncertain_confidence))
            | (boundary >= boundary_threshold_value) | (novelty >= 0.55)
        )
        tl.store(sensitivity_ptr + row, sensitivity, mask=valid)
        tl.store(low_confidence_ptr + row, low.to(tl.int8), mask=valid)
        tl.store(raw_event_ptr + row, event.to(tl.int8), mask=valid)


def normalize_rows(value: torch.Tensor) -> torch.Tensor:
    """FP32 row normalization with one CUDA launch."""
    if (
        not triton_ready(value)
        or os.environ.get("VGGT_TRITON_NORMALIZE", "0") != "1"
    ):
        return torch.nn.functional.normalize(value.float(), dim=-1, eps=1e-6)
    flat = value.reshape(-1, value.shape[-1]).contiguous()
    output = torch.empty(flat.shape, device=flat.device, dtype=torch.float32)
    block_m = 8 if flat.shape[1] <= 128 else 2
    block_c = triton.next_power_of_2(flat.shape[1])
    _row_l2_normalize_kernel[(triton.cdiv(len(flat), block_m),)](
        flat, output, len(flat), columns=flat.shape[1],
        BLOCK_M=block_m, BLOCK_C=block_c, num_warps=4,
    )
    return output.reshape(value.shape)


def local_top2_match(
    source: torch.Tensor, target: torch.Tensor, radius_y: int, radius_x: int,
    *, reverse_order: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused exact-window cosine Top-2 for ``[pairs,H,W,C]`` descriptors."""
    if not triton_ready(source):
        raise RuntimeError("local_top2_match requires CUDA Triton")
    if source.shape != target.shape or source.ndim != 4:
        raise ValueError("source and target descriptors must share [pairs,H,W,C]")
    pairs, height, width, channels = source.shape
    rows = pairs * height * width
    source = source.contiguous()
    target = target.contiguous()
    best = torch.empty(rows, device=source.device, dtype=torch.float32)
    second = torch.empty_like(best)
    dy = torch.empty(rows, device=source.device, dtype=torch.int32)
    dx = torch.empty_like(dy)
    _local_top2_match_kernel[(rows,)](
        source, target, best, second, dy, dx, rows,
        height=height, width=width, channels=channels,
        radius_y=radius_y, radius_x=radius_x, reverse_order=reverse_order,
        BLOCK_C=triton.next_power_of_2(channels), num_warps=4,
    )
    shape = (pairs, height, width)
    return best.view(shape), second.view(shape), dy.view(shape).long(), dx.view(shape).long()


def integer_gather_warp(field: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Border-padded integer warp equivalent to align-corners grid_sample."""
    if not triton_ready(field):
        raise RuntimeError("integer_gather_warp requires CUDA Triton")
    if field.ndim != 4 or flow.shape != (*field.shape[:3], 2):
        raise ValueError("field/flow must be [B,H,W,C] and [B,H,W,2]")
    batch, height, width, channels = field.shape
    rows = batch * height * width
    elements = rows * channels
    output = torch.empty_like(field.contiguous())
    block = 256
    _integer_gather_warp_kernel[(triton.cdiv(elements, block),)](
        field.contiguous(), flow.contiguous(), output, elements,
        height=height, width=width, channels=channels,
        BLOCK=block, num_warps=4,
    )
    return output


def fused_motion_cues(
    forward_flow: torch.Tensor, backward_flow: torch.Tensor,
    predicted: torch.Tensor, forward_confidence: torch.Tensor,
    backward_confidence: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse residual aggregation, confidence propagation and 3x3 boundaries."""
    if not triton_ready(predicted):
        raise RuntimeError("fused_motion_cues requires CUDA Triton")
    frames, height, width, channels = predicted.shape
    if channels != 4:
        raise ValueError("predicted motion must have four channels")
    rows = frames * height * width
    outputs = [torch.empty(rows, device=predicted.device, dtype=torch.float32) for _ in range(3)]
    _motion_cues_kernel[(rows,)](
        forward_flow.contiguous(), backward_flow.contiguous(), predicted.contiguous(),
        forward_confidence.contiguous(), backward_confidence.contiguous(),
        *outputs, rows, frames=frames, height=height, width=width, num_warps=1,
    )
    shape = (frames, height, width)
    return tuple(value.view(shape) for value in outputs)


def fused_sensitivity_labels(
    residual: torch.Tensor, confidence: torch.Tensor, boundary: torch.Tensor,
    *, uncertain_confidence: float, residual_scale: torch.Tensor,
    boundary_scale: torch.Tensor, residual_threshold: torch.Tensor,
    boundary_threshold: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse cue normalization, sensitivity and preliminary label predicates."""
    if not triton_ready(residual):
        raise RuntimeError("fused_sensitivity_labels requires CUDA Triton")
    frames, height, width = residual.shape
    rows = residual.numel()
    sensitivity = torch.empty_like(residual, dtype=torch.float32)
    low = torch.empty_like(residual, dtype=torch.int8)
    event = torch.empty_like(residual, dtype=torch.int8)
    _sensitivity_labels_kernel[(rows,)](
        residual.contiguous(), confidence.contiguous(), boundary.contiguous(),
        sensitivity, low, event, rows, patch_count=height * width,
        uncertain_confidence=float(uncertain_confidence),
        residual_scale=residual_scale, boundary_scale=boundary_scale,
        residual_threshold=residual_threshold, boundary_threshold=boundary_threshold,
        num_warps=1,
    )
    return sensitivity, low.bool(), event.bool()
