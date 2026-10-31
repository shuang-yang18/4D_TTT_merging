"""Training-free test-time optimization of patch-aligned world XYZ+t geometry."""
from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .ttt_kernels import integer_gather_warp, triton_ready


_NVOFA_BUILD_DIR = Path(
    "/data/mmc_syang/NvidiaOpticalFlowSDKforPytorch/build-numpy"
)


def uniform_anchor_indices(frames: int, anchor_frames: int) -> tuple[int, ...]:
    """Return distinct uniform anchors including both temporal endpoints."""
    if frames < 2 or not 2 <= anchor_frames <= frames:
        raise ValueError("anchor_frames must lie in [2, frames]")
    indices = torch.linspace(0, frames - 1, anchor_frames).round().long().unique()
    if len(indices) != anchor_frames:
        raise RuntimeError("uniform anchor construction produced duplicate frames")
    return tuple(int(value) for value in indices)


def _identity_grid(
    height: int, width: int, *, device: torch.device, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype), indexing="ij",
    )
    grid = torch.stack((xx, yy), dim=-1)
    grid[..., 0] = grid[..., 0] * (2.0 / max(width - 1, 1)) - 1.0
    grid[..., 1] = grid[..., 1] * (2.0 / max(height - 1, 1)) - 1.0
    return grid


def _flow_grid(flow: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert patch-pixel flow to a grid_sample grid and in-bounds mask."""
    if flow.ndim != 4 or flow.shape[-1] != 2:
        raise ValueError(f"flow must be [pairs,H,W,2], got {tuple(flow.shape)}")
    _, height, width, _ = flow.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=torch.float32),
        torch.arange(width, device=flow.device, dtype=torch.float32), indexing="ij",
    )
    xy = torch.stack((xx, yy), dim=-1)[None] + flow.float()
    valid = (
        (xy[..., 0] >= 0) & (xy[..., 0] <= width - 1) &
        (xy[..., 1] >= 0) & (xy[..., 1] <= height - 1)
    )
    xy[..., 0] = xy[..., 0] * (2.0 / max(width - 1, 1)) - 1.0
    xy[..., 1] = xy[..., 1] * (2.0 / max(height - 1, 1)) - 1.0
    return xy, valid


def _warp(
    field: torch.Tensor, grid: torch.Tensor, integer_flow: torch.Tensor | None = None,
) -> torch.Tensor:
    """Warp [B,H,W,C] fields with [B,H,W,2] normalized sampling grids."""
    if (
        integer_flow is not None and triton_ready(field)
        # grid_sample is faster on the current CUDA stack; the direct gather
        # remains opt-in for architectures where integer sampling wins.
        and os.environ.get("VGGT_TRITON_INTEGER_GATHER", "0") == "1"
    ):
        return integer_gather_warp(field, integer_flow)
    return F.grid_sample(
        field.permute(0, 3, 1, 2), grid, mode="bilinear",
        padding_mode="border", align_corners=True,
    ).permute(0, 2, 3, 1)


@torch.no_grad()
def optimize_anchor_geometry_ttt(
    anchor_xyz: torch.Tensor,
    anchor_indices: tuple[int, ...],
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    backward_confidence: torch.Tensor,
    *,
    forward_confidence: torch.Tensor | None = None,
    time_weight: float = .25,
    iterations: int = 6,
    smoothness: float = 1.0,
    data_weight: float = .25,
    relaxation: float = .5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Propagate sparse world geometry and minimize a flow-consistency energy.

    ``forward_flow[t]`` maps patch coordinates from frame ``t`` to ``t+1``;
    ``backward_flow[t]`` maps coordinates from ``t+1`` to ``t``. The optimized
    variables are per-patch world coordinates for this test clip. Anchor
    predictions are clamped after every Jacobi step, so the procedure cannot
    drift away from the sparse VGGT geometry gauge.
    """
    if anchor_xyz.ndim != 4 or anchor_xyz.shape[-1] != 3:
        raise ValueError("anchor_xyz must be [anchors,H,W,3]")
    if (iterations < 0 or smoothness < 0 or data_weight <= 0 or time_weight < 0
            or not 0 < relaxation <= 1):
        raise ValueError("invalid TTT optimization hyperparameters")
    frames = forward_flow.shape[0] + 1
    anchors, height, width, _ = anchor_xyz.shape
    expected_flow = (frames - 1, height, width, 2)
    if forward_flow.shape != expected_flow or backward_flow.shape != expected_flow:
        raise ValueError(f"flows must have shape {expected_flow}")
    expected_confidence = (frames - 1, height, width)
    if backward_confidence.shape != expected_confidence:
        raise ValueError("backward_confidence must be [frames-1,H,W]")
    if forward_confidence is not None and forward_confidence.shape != expected_confidence:
        raise ValueError("forward_confidence must be [frames-1,H,W]")
    if (len(anchor_indices) != anchors
            or tuple(sorted(set(anchor_indices))) != anchor_indices):
        raise ValueError("anchor indices must be distinct, sorted, and match anchor_xyz")
    if anchor_indices[0] != 0 or anchor_indices[-1] != frames - 1:
        raise ValueError("TTT anchors must include the first and final frames")

    device = anchor_xyz.device
    forward_flow = forward_flow.to(device=device, dtype=torch.float32)
    backward_flow = backward_flow.to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(anchor_xyz).all()):
        raise ValueError("anchor geometry must be finite")
    if not bool(torch.isfinite(forward_flow).all() and torch.isfinite(backward_flow).all()):
        raise ValueError("TTT flows must be finite")
    if not bool(torch.isfinite(backward_confidence).all()):
        raise ValueError("backward confidence must be finite")
    if forward_confidence is not None and not bool(torch.isfinite(forward_confidence).all()):
        raise ValueError("forward confidence must be finite")
    forward_grid, forward_valid = _flow_grid(forward_flow)
    backward_grid, backward_valid = _flow_grid(backward_flow)
    integer_correspondence = (
        triton_ready(forward_flow)
        and os.environ.get("VGGT_TRITON_INTEGER_GATHER", "0") == "1"
        and bool(torch.equal(forward_flow, forward_flow.round()))
        and bool(torch.equal(backward_flow, backward_flow.round()))
    )
    forward_integer = forward_flow if integer_correspondence else None
    backward_integer = backward_flow if integer_correspondence else None
    backward_conf = backward_confidence.to(
        device=device, dtype=torch.float32,
    ).clamp(0, 1) * backward_valid.float()
    if forward_confidence is None:
        forward_confidence = _warp(
            backward_conf[..., None], forward_grid, forward_integer,
        )[..., 0]
    forward_conf = forward_confidence.to(
        device=device, dtype=torch.float32,
    ).clamp(0, 1) * forward_valid.float()
    anchor_by_frame = {frame: index for index, frame in enumerate(anchor_indices)}

    # Bidirectional initialization inside every anchor interval. Long-range
    # confidence is retained, while a small floor lets disoccluded patches use
    # the same-position border fallback instead of becoming undefined.
    initial = torch.empty((frames, height, width, 3), device=device, dtype=torch.float32)
    initial_confidence = torch.empty((frames, height, width), device=device, dtype=torch.float32)
    for left, right in zip(anchor_indices[:-1], anchor_indices[1:]):
        left_fields = {left: anchor_xyz[anchor_by_frame[left]].float()}
        left_confidence = {left: torch.ones((height, width), device=device)}
        for frame in range(left + 1, right + 1):
            integer_slice = None if backward_integer is None else backward_integer[frame - 1:frame]
            warped = _warp(
                left_fields[frame - 1][None], backward_grid[frame - 1:frame], integer_slice,
            )[0]
            warped_conf = _warp(
                left_confidence[frame - 1][None, ..., None], backward_grid[frame - 1:frame],
                integer_slice,
            )[0, ..., 0]
            pair = backward_conf[frame - 1]
            left_fields[frame] = warped
            left_confidence[frame] = (warped_conf * pair).clamp_min(.02)

        right_fields = {right: anchor_xyz[anchor_by_frame[right]].float()}
        right_confidence = {right: torch.ones((height, width), device=device)}
        for frame in range(right - 1, left - 1, -1):
            integer_slice = None if forward_integer is None else forward_integer[frame:frame + 1]
            warped = _warp(
                right_fields[frame + 1][None], forward_grid[frame:frame + 1], integer_slice,
            )[0]
            warped_conf = _warp(
                right_confidence[frame + 1][None, ..., None], forward_grid[frame:frame + 1],
                integer_slice,
            )[0, ..., 0]
            pair = forward_conf[frame]
            right_fields[frame] = warped
            right_confidence[frame] = (warped_conf * pair).clamp_min(.02)

        span = right - left
        for frame in range(left, right + 1):
            if frame in anchor_by_frame:
                initial[frame] = anchor_xyz[anchor_by_frame[frame]].float()
                initial_confidence[frame].fill_(1.0)
                continue
            left_prior = float(right - frame) / span
            right_prior = float(frame - left) / span
            left_weight = left_confidence[frame] * left_prior
            right_weight = right_confidence[frame] * right_prior
            total = (left_weight + right_weight).clamp_min(1e-6)
            initial[frame] = (
                left_fields[frame] * left_weight[..., None]
                + right_fields[frame] * right_weight[..., None]
            ) / total[..., None]
            initial_confidence[frame] = total.clamp(max=1.0)

    geometry = initial.clone()
    anchor_mask = torch.zeros(frames, device=device, dtype=torch.bool)
    anchor_mask[list(anchor_indices)] = True
    initial_previous = _warp(initial[:-1], backward_grid, backward_integer)
    initial_residual = (initial[1:] - initial_previous).norm(dim=-1)
    initial_consistency = (
        (initial_residual * backward_conf).sum() / backward_conf.sum().clamp_min(1e-6)
    )

    def objective(value: torch.Tensor) -> torch.Tensor:
        previous_value = _warp(value[:-1], backward_grid, backward_integer)
        flow_error = (value[1:] - previous_value).square().sum(dim=-1)
        flow_term = (flow_error * backward_conf).sum() / backward_conf.sum().clamp_min(1e-6)
        data_error = (value - initial).square().sum(dim=-1)
        data_term = (
            (data_error * initial_confidence).sum()
            / initial_confidence.sum().clamp_min(1e-6)
        )
        return smoothness * flow_term + data_weight * data_term

    initial_objective = objective(geometry)
    current_objective = initial_objective
    completed_iterations = 0
    for _ in range(iterations):
        numerator = initial * (data_weight * initial_confidence[..., None])
        denominator = data_weight * initial_confidence
        previous = _warp(geometry[:-1], backward_grid, backward_integer)
        next_value = _warp(geometry[1:], forward_grid, forward_integer)
        numerator[1:] += smoothness * previous * backward_conf[..., None]
        denominator[1:] += smoothness * backward_conf
        numerator[:-1] += smoothness * next_value * forward_conf[..., None]
        denominator[:-1] += smoothness * forward_conf
        proposal = numerator / denominator.clamp_min(1e-6)[..., None]
        proposal[anchor_mask] = anchor_xyz.float()
        accepted = False
        step = relaxation
        for _line_search in range(5):
            candidate = torch.lerp(geometry, proposal, step)
            candidate[anchor_mask] = anchor_xyz.float()
            candidate_objective = objective(candidate)
            if bool(candidate_objective <= current_objective + 1e-8):
                geometry = candidate
                current_objective = candidate_objective
                completed_iterations += 1
                accepted = True
                break
            step *= .5
        if not accepted:
            break

    finite = torch.isfinite(geometry).all(dim=-1)
    if not bool(finite.all()):
        geometry[~finite] = initial[~finite]
    geometry = geometry.clamp(-6, 6)
    frame_time = torch.linspace(-.5, .5, frames, device=device, dtype=torch.float32)
    frame_time = frame_time[:, None, None, None].expand(frames, height, width, 1)
    descriptor = torch.cat((geometry, time_weight * frame_time), dim=-1).reshape(-1, 4).contiguous()

    with torch.no_grad():
        previous = _warp(geometry[:-1], backward_grid, backward_integer)
        residual = (geometry[1:] - previous).norm(dim=-1)
        weighted_residual = (residual * backward_conf).sum() / backward_conf.sum().clamp_min(1e-6)
    return descriptor, {
        "frames": float(frames),
        "anchor_frames": float(anchors),
        "anchor_ratio": float(anchors / frames),
        "ttt_iterations": float(iterations),
        "ttt_smoothness": float(smoothness),
        "ttt_data_weight": float(data_weight),
        "ttt_relaxation": float(relaxation),
        "ttt_completed_iterations": float(completed_iterations),
        "flow_valid_ratio": float(
            .5 * ((backward_conf > .05).float().mean()
                  + (forward_conf > .05).float().mean())
        ),
        "integer_gather_propagation": bool(integer_correspondence),
        "initial_ttt_objective": float(initial_objective),
        "final_ttt_objective": float(current_objective),
        "ttt_objective_improvement": float(
            (initial_objective - current_objective) / initial_objective.clamp_min(1e-8)
        ),
        "initial_flow_consistency_residual": float(initial_consistency),
        "flow_consistency_residual": float(weighted_residual),
        "flow_consistency_improvement": float(
            (initial_consistency - weighted_residual) / initial_consistency.clamp_min(1e-8)
        ),
        "time_weight": float(time_weight),
    }


def _nvofa_module():
    if _NVOFA_BUILD_DIR.is_dir() and str(_NVOFA_BUILD_DIR) not in sys.path:
        sys.path.insert(0, str(_NVOFA_BUILD_DIR))
    try:
        return importlib.import_module("nvof_numpy")
    except ImportError:
        return None


def _abgr(gray: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack((np.full_like(gray, 255), gray, gray, gray), axis=-1)
    )


@torch.no_grad()
def adjacent_patch_flows(
    images: torch.Tensor,
    patch_hw: tuple[int, int],
    *,
    backend: str = "auto",
    gpu_id: int = 0,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    dict[str, float | str],
]:
    """Measure adjacent bidirectional flow and reduce it to patch coordinates."""
    if images.ndim != 4 or images.shape[1] < 3:
        raise ValueError("images must be [T,C,H,W]")
    if not images.is_cuda:
        raise ValueError("adjacent_patch_flows expects CUDA images")
    if backend not in {"auto", "nvofa", "dis"}:
        raise ValueError("flow backend must be auto, nvofa, or dis")
    frames = images.shape[0]
    patch_height, patch_width = patch_hw
    flow_height, flow_width = patch_height * 4, patch_width * 4
    started = time.perf_counter()
    rgb = F.interpolate(images[:, :3].float(), size=(flow_height, flow_width), mode="area")
    gray = (rgb[:, 0] * .299 + rgb[:, 1] * .587 + rgb[:, 2] * .114).clamp(0, 1)
    gray_host = gray.mul(255).round().to(torch.uint8).cpu().numpy()
    module = _nvofa_module() if backend in {"auto", "nvofa"} else None
    if backend == "nvofa" and module is None:
        raise RuntimeError("NVOFA requested but nvof_numpy is unavailable")
    engine = None if module is None else module.NumpyNVOpticalFlow(
        flow_width, flow_height, gpu_id, "fast",
    )
    resolved_backend = "nvofa" if engine is not None else "dis"
    forward_rows, backward_rows = [], []
    dis = None
    if engine is None:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        dis.setFinestScale(1)
        dis.setVariationalRefinementIterations(0)
    for frame in range(frames - 1):
        if engine is not None:
            forward, backward = engine.compute_bidirectional(
                _abgr(gray_host[frame]), _abgr(gray_host[frame + 1]),
            )
        else:
            forward = dis.calc(gray_host[frame], gray_host[frame + 1], None)
            backward = dis.calc(gray_host[frame + 1], gray_host[frame], None)
        forward_rows.append(torch.from_numpy(np.ascontiguousarray(forward)))
        backward_rows.append(torch.from_numpy(np.ascontiguousarray(backward)))
    forward = torch.stack(forward_rows).to(images.device, dtype=torch.float32)
    backward = torch.stack(backward_rows).to(images.device, dtype=torch.float32)

    # Average flow over each 4x4 flow cell and convert from flow-image pixels
    # to patch-grid pixels. Average pooling preserves sub-patch displacement.
    def reduce_flow(value: torch.Tensor) -> torch.Tensor:
        value = F.avg_pool2d(value.permute(0, 3, 1, 2), 4, stride=4)
        value[:, 0] *= patch_width / flow_width
        value[:, 1] *= patch_height / flow_height
        return value.permute(0, 2, 3, 1).contiguous()

    forward_patch = reduce_flow(forward)
    backward_patch = reduce_flow(backward)
    forward_grid, forward_valid = _flow_grid(forward_patch)
    backward_grid, backward_valid = _flow_grid(backward_patch)
    forward_at_previous = _warp(forward_patch, backward_grid)
    forward_valid_at_previous = _warp(
        forward_valid[..., None].float(), backward_grid,
    )[..., 0] >= .999
    fb_error = (backward_patch + forward_at_previous).norm(dim=-1)
    backward_confidence = torch.exp(-.5 * (fb_error / .75).square())
    backward_confidence *= (forward_valid_at_previous & backward_valid).float()

    backward_at_next = _warp(backward_patch, forward_grid)
    backward_valid_at_next = _warp(
        backward_valid[..., None].float(), forward_grid,
    )[..., 0] >= .999
    bf_error = (forward_patch + backward_at_next).norm(dim=-1)
    forward_confidence = torch.exp(-.5 * (bf_error / .75).square())
    forward_confidence *= (backward_valid_at_next & forward_valid).float()
    torch.cuda.synchronize(images.device)
    return forward_patch, backward_patch, forward_confidence, backward_confidence, {
        "flow_backend": resolved_backend,
        "flow_ms": (time.perf_counter() - started) * 1000.0,
        "flow_height": float(flow_height),
        "flow_width": float(flow_width),
        "pair_count": float(frames - 1),
        "forward_confident_ratio": float((forward_confidence > .05).float().mean()),
        "backward_confident_ratio": float((backward_confidence > .05).float().mean()),
    }
