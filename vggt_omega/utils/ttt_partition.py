"""Adaptive stable/sensitive patch partition from TTT world geometry and flow."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .ttt_4d import _flow_grid, _warp


def _robust_unit(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask].float()
    if not selected.numel():
        return torch.zeros_like(value, dtype=torch.float32)
    scale = torch.quantile(selected, .90).clamp_min(1e-6)
    return (value.float() / scale).clamp(0, 1)


@torch.no_grad()
def ttt_motion_visibility_cues(
    world: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    backward_confidence: torch.Tensor,
    *,
    forward_confidence: torch.Tensor | None = None,
    motion_floor: float = .04,
    heterogeneity_floor: float = .025,
    novelty_threshold: float = .55,
    mad_multiplier: float = 3.0,
    uncertain_confidence: float = .10,
    max_uncertain_ratio: float = .02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | str]]:
    """Compute the shared stable/sensitive/uncertain TTT cues.

    ``backward_confidence[t]`` is defined on frame ``t+1`` coordinates,
    whereas ``forward_confidence[t]`` is defined on frame ``t`` coordinates.
    Keeping both maps avoids the former source/destination coordinate mix-up.
    """
    if world.ndim != 4 or world.shape[-1] != 3:
        raise ValueError("world geometry must be [frames,H,W,3]")
    frames, height, width, _ = world.shape
    expected_flow = (frames - 1, height, width, 2)
    expected_confidence = (frames - 1, height, width)
    if forward_flow.shape != expected_flow or backward_flow.shape != expected_flow:
        raise ValueError(f"flows must have shape {expected_flow}")
    if backward_confidence.shape != expected_confidence:
        raise ValueError(f"backward confidence must have shape {expected_confidence}")
    if forward_confidence is not None and forward_confidence.shape != expected_confidence:
        raise ValueError(f"forward confidence must have shape {expected_confidence}")
    if not all(torch.isfinite(value).all() for value in (
        world, forward_flow, backward_flow, backward_confidence,
    )):
        raise ValueError("TTT geometry, flow, and confidence must be finite")

    backward_grid, backward_valid = _flow_grid(backward_flow)
    forward_grid, forward_valid = _flow_grid(forward_flow)
    backward_confidence = backward_confidence.to(
        device=world.device, dtype=torch.float32,
    ).clamp(0, 1) * backward_valid.float()
    if forward_confidence is None:
        # Compatibility fallback for callers with only target-grid confidence:
        # transport it onto the source frame instead of reusing it unwarped.
        forward_confidence = _warp(
            backward_confidence[..., None], forward_grid,
        )[..., 0]
    forward_confidence = forward_confidence.to(
        device=world.device, dtype=torch.float32,
    ).clamp(0, 1) * forward_valid.float()

    previous_world = _warp(world[:-1], backward_grid)
    motion = torch.zeros((frames, height, width), device=world.device)
    motion[1:] = (world[1:] - previous_world).norm(dim=-1)

    confidence = torch.ones_like(motion)
    confidence[1:] = backward_confidence
    confidence[:-1] = torch.minimum(confidence[:-1], forward_confidence)
    novelty = 1.0 - confidence
    motion_average = F.avg_pool2d(motion[:, None], 3, stride=1, padding=1)[:, 0]
    flow_average = F.avg_pool2d(
        forward_flow.permute(0, 3, 1, 2), 3, stride=1, padding=1,
    ).permute(0, 2, 3, 1)
    flow_boundary = torch.zeros_like(motion)
    flow_boundary[:-1] = (forward_flow - flow_average).norm(dim=-1)
    heterogeneity = (motion - motion_average).abs() + .25 * flow_boundary

    later = torch.zeros_like(motion, dtype=torch.bool)
    later[1:] = True
    reliable = later & (confidence >= uncertain_confidence)

    def threshold(value: torch.Tensor, floor: float) -> torch.Tensor:
        selected = value[reliable]
        if not selected.numel():
            return value.new_tensor(floor)
        median = selected.median()
        mad = (selected - median).abs().median()
        return torch.maximum(
            value.new_tensor(floor), median + mad_multiplier * 1.4826 * mad,
        )

    motion_threshold = threshold(motion, motion_floor)
    heterogeneity_threshold = threshold(heterogeneity, heterogeneity_floor)
    sensitivity = (
        .45 * novelty
        + .30 * _robust_unit(motion, later)
        + .25 * _robust_unit(heterogeneity, later)
    ).clamp(0, 1)
    low_confidence = (confidence < uncertain_confidence) & later
    uncertain = torch.zeros_like(low_confidence)
    candidates = low_confidence.flatten().nonzero().flatten()
    budget = min(
        len(candidates),
        max(1, int(round(max_uncertain_ratio * motion.numel())))
        if max_uncertain_ratio > 0 else 0,
    )
    if budget:
        score = sensitivity.flatten().index_select(0, candidates)
        protected = candidates.index_select(0, score.topk(budget).indices)
        uncertain.flatten()[protected] = True
    event = (
        ((motion >= motion_threshold) & (heterogeneity >= heterogeneity_threshold))
        | (novelty >= novelty_threshold)
        | low_confidence
    ) & later & ~uncertain
    labels = torch.zeros_like(motion, dtype=torch.long)
    labels[event] = 1
    labels[uncertain] = 2
    labels[0] = 0
    return labels.flatten(), sensitivity.flatten(), confidence, {
        "partition": "adaptive_bidirectional_motion_visibility_v2",
        "motion_threshold": float(motion_threshold),
        "heterogeneity_threshold": float(heterogeneity_threshold),
        "novelty_threshold": float(novelty_threshold),
        "stable_ratio": float((labels == 0).float().mean()),
        "sensitive_ratio": float((labels == 1).float().mean()),
        "uncertain_ratio": float((labels == 2).float().mean()),
        "low_confidence_ratio": float(low_confidence.float().mean()),
        "motion_mean": float(motion.mean()),
    }


@torch.no_grad()
def adaptive_ttt_partition(
    geometry: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    backward_confidence: torch.Tensor,
    patch_hw: tuple[int, int],
    *,
    forward_confidence: torch.Tensor | None = None,
    motion_floor: float = .04,
    heterogeneity_floor: float = .025,
    novelty_threshold: float = .55,
    mad_multiplier: float = 3.0,
    uncertain_confidence: float = .10,
    max_uncertain_ratio: float = .02,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | str]]:
    """Return flattened labels (stable=0, sensitive=1, uncertain=2)."""
    height, width = patch_hw
    if geometry.ndim != 2 or geometry.shape[-1] != 4:
        raise ValueError("geometry must be [frames*patches,4]")
    if len(geometry) % (height * width):
        raise ValueError("geometry length must be divisible by patches per frame")
    frames = len(geometry) // (height * width)
    world = geometry.reshape(frames, height, width, 4)[..., :3].float()
    labels, sensitivity, _, stats = ttt_motion_visibility_cues(
        world, forward_flow, backward_flow, backward_confidence,
        forward_confidence=forward_confidence,
        motion_floor=motion_floor,
        heterogeneity_floor=heterogeneity_floor,
        novelty_threshold=novelty_threshold,
        mad_multiplier=mad_multiplier,
        uncertain_confidence=uncertain_confidence,
        max_uncertain_ratio=max_uncertain_ratio,
    )
    return labels, sensitivity, stats
