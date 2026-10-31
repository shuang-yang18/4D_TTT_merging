"""Sequence-specific test-time training for patch-aligned 4-D geometry.

Unlike :mod:`vggt_omega.utils.ttt_4d`, this module updates the parameters of a
small neural adapter by forward/backward optimisation on every test clip.  The
VGGT-Omega backbone remains frozen.  No dataset labels or offline adapter
checkpoint are used.
"""
from __future__ import annotations

import math
import time

import torch
from torch import nn
import torch.nn.functional as F

from .ttt_4d import _flow_grid, _warp, optimize_anchor_geometry_ttt
from .ttt_partition import ttt_motion_visibility_cues


class Sequence4DAdapter(nn.Module):
    """Small residual geometry and sensitivity model reset for every clip."""

    def __init__(self, input_dim: int = 8, hidden_dim: int = 32) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.geometry_head = nn.Linear(hidden_dim, 3)
        self.sensitivity_head = nn.Linear(hidden_dim, 1)
        # The initial adapter is exactly the observation-only geometry.  This
        # makes a failed/zero-step adaptation a safe identity operation.
        nn.init.zeros_(self.geometry_head.weight)
        nn.init.zeros_(self.geometry_head.bias)
        nn.init.zeros_(self.sensitivity_head.weight)
        nn.init.constant_(self.sensitivity_head.bias, math.log(.2 / .8))

    def forward(
        self, inputs: torch.Tensor, initial_xyz: torch.Tensor, xyz_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(inputs)
        # Bound a sequence-specific correction to 25% of the robust scene
        # scale.  The regularizer below normally keeps it substantially lower.
        correction = .25 * xyz_scale * torch.tanh(self.geometry_head(hidden))
        sensitivity_logit = self.sensitivity_head(hidden)[..., 0]
        return initial_xyz + correction, sensitivity_logit


def adapt_anchor_geometry_strict_ttt(
    anchor_xyz: torch.Tensor,
    anchor_indices: tuple[int, ...],
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    backward_confidence: torch.Tensor,
    *,
    forward_confidence: torch.Tensor | None = None,
    time_weight: float = .25,
    steps: int = 12,
    learning_rate: float = 3e-3,
    hidden_dim: int = 32,
    anchor_weight: float = 4.0,
    flow_weight: float = 1.0,
    classification_weight: float = .25,
    correction_weight: float = .05,
    spatial_weight: float = .02,
    uncertain_confidence: float = .10,
    max_uncertain_ratio: float = .02,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | str]]:
    """Adapt a fresh neural 4-D model on one clip and return its predictions.

    Returns ``(descriptor, labels, sensitivity, statistics)``.  Labels use
    stable=0, sensitive=1 and uncertain=2.  Every call constructs a new model,
    so no information or parameters leak across test sequences.
    """
    if steps < 1 or learning_rate <= 0 or hidden_dim < 4:
        raise ValueError("strict TTT needs positive steps/lr and hidden_dim >= 4")
    started = time.perf_counter()
    device = anchor_xyz.device
    # Callers normally obtain anchor predictions under inference_mode.  Such
    # tensors cannot be saved by autograd even when requires_grad is false;
    # clone outside that context to make an ordinary immutable observation.
    anchor_xyz = anchor_xyz.clone()
    frames = forward_flow.shape[0] + 1
    height, width = forward_flow.shape[1:3]
    if anchor_xyz.shape != (len(anchor_indices), height, width, 3):
        raise ValueError("anchor geometry and flow grids do not match")

    # iterations=0 exposes the same bidirectional anchor interpolation used by
    # the old state optimiser without performing any state update.
    initial_descriptor, initial_stats = optimize_anchor_geometry_ttt(
        anchor_xyz, anchor_indices, forward_flow, backward_flow, backward_confidence,
        forward_confidence=forward_confidence,
        time_weight=time_weight, iterations=0, smoothness=.5,
        data_weight=.5, relaxation=.25,
    )
    initial_xyz = initial_descriptor[:, :3].reshape(frames, height, width, 3).float()
    pseudo_labels, initial_sensitivity, confidence, cue_stats = ttt_motion_visibility_cues(
        initial_xyz, forward_flow, backward_flow, backward_confidence,
        forward_confidence=forward_confidence,
        uncertain_confidence=uncertain_confidence,
        max_uncertain_ratio=max_uncertain_ratio,
    )
    pseudo_event = pseudo_labels.reshape(frames, height, width) == 1
    uncertain = pseudo_labels.reshape(frames, height, width) == 2
    initial_sensitivity = initial_sensitivity.reshape(frames, height, width)
    target_stats = {
        "pseudo_sensitive_ratio": float(pseudo_event.float().mean()),
        "pseudo_uncertain_ratio": float(uncertain.float().mean()),
        "pseudo_motion_threshold": cue_stats["motion_threshold"],
        "pseudo_heterogeneity_threshold": cue_stats["heterogeneity_threshold"],
        "pseudo_motion_mean": cue_stats["motion_mean"],
    }

    xyz_centre = initial_xyz.reshape(-1, 3).median(dim=0).values
    xyz_scale = torch.quantile(
        (initial_xyz - xyz_centre).norm(dim=-1).flatten(), .90,
    ).clamp_min(1e-3)
    xyz_unit = (initial_xyz - xyz_centre) / xyz_scale
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device), indexing="ij",
    )
    spatial = torch.stack((xx, yy), dim=-1)[None].expand(frames, -1, -1, -1)
    temporal = torch.linspace(-1, 1, frames, device=device)[:, None, None, None]
    temporal = temporal.expand(-1, height, width, -1)
    flow_magnitude = torch.zeros((frames, height, width), device=device)
    flow_magnitude[:-1] += forward_flow.norm(dim=-1)
    flow_magnitude[1:] += backward_flow.norm(dim=-1)
    flow_magnitude /= 2.0
    flow_scale = torch.quantile(flow_magnitude.flatten(), .90).clamp_min(1e-6)
    inputs = torch.cat((
        xyz_unit, spatial, temporal, confidence[..., None],
        (flow_magnitude / flow_scale).clamp(0, 2)[..., None],
        initial_sensitivity[..., None],
    ), dim=-1).detach()

    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        model = Sequence4DAdapter(inputs.shape[-1], hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    backward_grid, backward_valid = _flow_grid(backward_flow)
    pair_weight = backward_confidence.float() * backward_valid.float()
    anchor_mask = torch.zeros(frames, device=device, dtype=torch.bool)
    anchor_mask[list(anchor_indices)] = True
    reliable_static = (~pseudo_event[1:] & ~uncertain[1:]).float() * pair_weight
    target = pseudo_event.float()
    positive = target.sum().clamp_min(1)
    negative = (1 - target).sum().clamp_min(1)
    positive_weight = (negative / positive).clamp(1, 8)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    losses: list[float] = []
    initial_flow_residual = None

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        world, sensitivity_logit = model(inputs, initial_xyz, xyz_scale)
        previous = _warp(world[:-1], backward_grid)
        residual = (world[1:] - previous).norm(dim=-1)
        if initial_flow_residual is None:
            initial_flow_residual = float(
                (residual.detach() * reliable_static).sum()
                / reliable_static.sum().clamp_min(1e-6)
            )
        flow_loss = (
            F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction="none", beta=.05)
            * reliable_static
        ).sum() / reliable_static.sum().clamp_min(1e-6)
        anchor_loss = F.smooth_l1_loss(
            world[anchor_mask], anchor_xyz.float(), beta=.05,
        )
        correction = world - initial_xyz
        correction_loss = correction.square().mean() / xyz_scale.square()
        horizontal = correction[:, :, 1:] - correction[:, :, :-1]
        vertical = correction[:, 1:] - correction[:, :-1]
        spatial_loss = (
            horizontal.square().mean() + vertical.square().mean()
        ) / xyz_scale.square()
        classification_loss = F.binary_cross_entropy_with_logits(
            sensitivity_logit, target, pos_weight=positive_weight,
        )
        parameter_drift = sum(
            (parameter - initial).square().mean()
            for parameter, initial in zip(model.parameters(), initial_parameters)
        )
        loss = (
            flow_weight * flow_loss
            + anchor_weight * anchor_loss
            + classification_weight * classification_loss
            + correction_weight * correction_loss
            + spatial_weight * spatial_loss
            + 1e-5 * parameter_drift
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("strict TTT produced a non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    model.eval()
    with torch.no_grad():
        world, logits = model(inputs, initial_xyz, xyz_scale)
        sensitivity = logits.sigmoid()
        labels = torch.zeros_like(logits, dtype=torch.long)
        # A fixed 0.5 boundary is poorly calibrated after only a handful of
        # online gradient steps.  Retain the observation-only event *rate* as
        # a sequence prior, while letting the trained adapter decide which
        # tokens occupy that budget.  This prevents all-stable/all-dynamic
        # collapse without copying the pseudo labels token by token.
        eligible = torch.ones_like(uncertain)
        eligible[0] = False
        eligible &= ~uncertain
        event_budget = min(int(pseudo_event.sum()), int(eligible.sum()))
        if event_budget:
            eligible_ids = eligible.flatten().nonzero().flatten()
            chosen = eligible_ids.index_select(
                0, sensitivity.flatten().index_select(0, eligible_ids).topk(event_budget).indices,
            )
            labels.flatten()[chosen] = 1
        labels[uncertain] = 2
        labels[0] = 0
        previous = _warp(world[:-1], backward_grid)
        final_residual = (world[1:] - previous).norm(dim=-1)
        final_flow_residual = float(
            (final_residual * reliable_static).sum()
            / reliable_static.sum().clamp_min(1e-6)
        )
        frame_time = torch.linspace(-.5, .5, frames, device=device)
        frame_time = frame_time[:, None, None, None].expand(frames, height, width, 1)
        descriptor = torch.cat((world, time_weight * frame_time), dim=-1)

    parameter_delta = math.sqrt(sum(
        float((parameter.detach() - initial).square().sum())
        for parameter, initial in zip(model.parameters(), initial_parameters)
    ))
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    return (
        descriptor.reshape(-1, 4).contiguous(),
        labels.flatten().contiguous(),
        sensitivity.flatten().contiguous(),
        {
            "ttt_mode": "strict_sequence_adapter",
            "adapter_reset_per_sequence": True,
            "backbone_frozen": True,
            "frames": float(frames),
            "anchor_frames": float(len(anchor_indices)),
            "anchor_ratio": float(len(anchor_indices) / frames),
            "time_weight": float(time_weight),
            "adapter_trainable_parameters": float(trainable_parameters),
            "adapter_hidden_dim": float(hidden_dim),
            "adapter_steps": float(steps),
            "adapter_learning_rate": float(learning_rate),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_reduction_fraction": (losses[0] - losses[-1]) / max(losses[0], 1e-8),
            "parameter_delta_l2": parameter_delta,
            "initial_static_flow_residual": float(initial_flow_residual or 0.0),
            "final_static_flow_residual": final_flow_residual,
            "static_flow_residual_reduction_fraction": (
                (float(initial_flow_residual or 0.0) - final_flow_residual)
                / max(float(initial_flow_residual or 0.0), 1e-8)
            ),
            "stable_ratio": float((labels == 0).float().mean()),
            "sensitive_ratio": float((labels == 1).float().mean()),
            "uncertain_ratio": float((labels == 2).float().mean()),
            "sensitivity_mean": float(sensitivity.mean()),
            "ttt_wall_ms": (time.perf_counter() - started) * 1000.0,
            **target_stats,
            "initial_geometry_source": "bidirectional_anchor_interpolation",
            "initial_flow_valid_ratio": initial_stats["flow_valid_ratio"],
        },
    )
