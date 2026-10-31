"""Flow-free adjacent-frame correspondence from VGGT patch tokens.

The routines in this module deliberately operate on the patch grid rather
than the image grid.  They replace an external dense optical-flow network
with local cosine matching of the patch descriptors already available from
VGGT-Omega's frozen image encoder.
"""
from __future__ import annotations

import math
import os
import time

import torch
from torch import nn
import torch.nn.functional as F

from .ttt_4d import optimize_anchor_geometry_ttt
from .ttt_kernels import (
    fused_motion_cues,
    fused_sensitivity_labels,
    local_top2_match,
    normalize_rows,
    triton_ready,
)
from vggt_omega.models.sparse_convex_kernels import four_support_projected_gradient


_PROJECTION_CACHE: dict[tuple[str, int, int, int], torch.Tensor] = {}


class FixedBasisDifficultyPredictor(nn.Module):
    """Per-sequence predictor of descriptor-space convex difficulty."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(channels, 64), nn.SiLU(),
            nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value).squeeze(-1)


@torch.no_grad()
def _descriptor_proxy_residual(
    descriptors: torch.Tensor, train_ids: torch.Tensor,
) -> torch.Tensor:
    frames, height, width, channels = descriptors.shape
    patches = height * width
    device = descriptors.device
    generator = torch.Generator(device="cpu").manual_seed(0)
    rows, cols = torch.arange(0, height, 2), torch.arange(0, width, 2)
    choice = torch.rand(frames - 1, len(rows), len(cols), 2, generator=generator)
    row = rows[None, :, None] + (
        choice[..., 0] * (height - rows).clamp(max=2)[None, :, None]
    ).long()
    col = cols[None, None, :] + (
        choice[..., 1] * (width - cols).clamp(max=2)[None, None, :]
    ).long()
    position = (row * width + col).reshape(frames - 1, -1).to(device)
    stable = torch.cat((
        torch.arange(patches, device=device),
        (position + torch.arange(1, frames, device=device)[:, None] * patches).flatten(),
    )).unique()
    stable_frame = torch.div(stable, patches, rounding_mode="floor")
    train_frame = torch.div(train_ids, patches, rounding_mode="floor")
    flat = descriptors.reshape(-1, channels).float()
    result = torch.zeros(len(train_ids), device=device)
    for begin in range(0, frames, 8):
        end = min(begin + 8, frames)
        selected = ((train_frame >= begin) & (train_frame < end)).nonzero().flatten()
        if not selected.numel():
            continue
        local_mask = (
            (stable_frame == 0)
            | ((stable_frame >= max(1, begin - 1))
               & (stable_frame <= min(frames - 1, end)))
        )
        local_basis = stable.index_select(0, local_mask.nonzero().flatten())
        source_ids = train_ids.index_select(0, selected)
        source = flat.index_select(0, source_ids)
        score = source.to(torch.bfloat16) @ flat.index_select(
            0, local_basis,
        ).to(torch.bfloat16).T
        local = score.topk(4, dim=-1).indices
        support_ids = local_basis.index_select(0, local.reshape(-1)).reshape(-1, 4)
        supports = flat.index_select(0, support_ids.reshape(-1)).reshape(
            len(selected), 4, channels,
        )
        gram = supports @ supports.transpose(1, 2)
        rhs = (supports @ source.unsqueeze(-1)).squeeze(-1)
        weights = four_support_projected_gradient(gram, rhs, steps=32)
        reconstruction = (supports * weights.unsqueeze(-1)).sum(dim=1)
        result.index_copy_(0, selected, (source - reconstruction).norm(dim=-1))
    return result


def select_fixed_dynamic_bases_ttt(
    descriptors: torch.Tensor, labels: torch.Tensor, sensitivity: torch.Tensor,
    geometry: torch.Tensor, *, budget: int = 20000, sample_stride: int = 10,
    steps: int = 500, learning_rate: float = 3e-3, seed: int = 0,
    acceptance_cosine: float = .90,
) -> tuple[torch.Tensor, dict[str, float | int | str]]:
    """Fit a predictor and retain only absolutely difficult event bases.

    ``budget`` is the upper bound at 300 frames rather than a mandatory Top-K
    quota; the realised cap scales linearly with sequence length.  For a
    unit source descriptor and convex reconstruction ``y``,
    ``||x-y|| <= sqrt(1-tau**2)`` is sufficient for cosine ``>= tau``.
    This gives an absolute, sequence-independent residual cutoff while the
    existing full-dimensional first-layer certificate still promotes any
    false negatives.
    """
    started = time.perf_counter()
    frames, height, width, channels = descriptors.shape
    patches = height * width
    total = frames * patches
    if labels.numel() != total or sensitivity.numel() != total:
        raise ValueError("fixed-basis TTT labels must align with descriptors")
    if geometry.shape != (total, 4):
        raise ValueError("fixed-basis TTT geometry must be [T*P,4]")
    budget_at_300_frames = max(1, int(budget))
    budget = min(
        max(1, int(round(budget_at_300_frames * frames / 300.0))), total,
    )
    if not 0.0 < acceptance_cosine <= 1.0:
        raise ValueError("acceptance_cosine must lie in (0,1]")
    device = descriptors.device
    train_ids = torch.arange(0, total, sample_stride, device=device)
    target = _descriptor_proxy_residual(descriptors, train_ids)
    one_hot = F.one_hot(labels.long(), num_classes=3).float()
    normalized_geometry = geometry.float()
    normalized_geometry = (
        normalized_geometry - normalized_geometry.mean(0, keepdim=True)
    ) / normalized_geometry.std(0, keepdim=True).clamp_min(1e-5)
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device), indexing="ij",
    )
    spatial = torch.stack((yy, xx), dim=-1).reshape(1, patches, 2).expand(
        frames, -1, -1,
    ).reshape(total, 2)
    sequence_time = torch.linspace(-1, 1, frames, device=device).reshape(
        frames, 1, 1,
    ).expand(-1, patches, -1).reshape(total, 1)
    features = torch.cat((
        descriptors.reshape(total, channels).float(), sensitivity[:, None].float(),
        one_hot, normalized_geometry, spatial, sequence_time,
    ), dim=-1)
    fork_devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=fork_devices):
        torch.manual_seed(seed)
        model = FixedBasisDifficultyPredictor(features.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4,
        fused=device.type == "cuda",
    )
    train_features = features.index_select(0, train_ids)
    with torch.enable_grad():
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(train_features)
            loss = F.smooth_l1_loss(prediction, target, beta=.1)
            loss.backward()
            optimizer.step()
    scores = []
    with torch.no_grad():
        for begin in range(0, total, 32768):
            scores.append(model(features[begin:begin + 32768]))
        score = torch.cat(scores)
        # The tiny per-sequence regressor is deliberately conservative: its
        # Smooth-L1 estimate shrinks the upper residual tail.  Trigger at half
        # the sufficient cosine-error bound, then let ``budget`` cap highly
        # dynamic sequences.  Unlike a percentile/Top-K rule this remains an
        # absolute cutoff, so genuinely static sequences can select fewer.
        safety_factor = float(os.environ.get(
            "VGGT_TTT_FIXED_BASIS_THRESHOLD_FACTOR", "0.4",
        ))
        if not 0.0 < safety_factor <= 1.0:
            raise ValueError("fixed-basis threshold factor must lie in (0,1]")
        sufficient_bound = math.sqrt(max(0.0, 1.0 - acceptance_cosine ** 2))
        default_threshold = safety_factor * sufficient_bound
        residual_threshold = float(os.environ.get(
            "VGGT_TTT_FIXED_BASIS_RESIDUAL_THRESHOLD", default_threshold,
        ))
        if residual_threshold < 0:
            raise ValueError("fixed-basis residual threshold must be nonnegative")
        event_mask = labels != 0
        candidate = (event_mask & (score >= residual_threshold)).nonzero().flatten()
        selected_count = min(budget, len(candidate))
        if selected_count:
            selected = candidate.index_select(
                0, score.index_select(0, candidate).topk(selected_count).indices,
            )
        else:
            selected = candidate
        mask = torch.zeros(total, device=device, dtype=torch.bool)
        mask[selected] = True
        event_scores = score[event_mask]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return mask, {
        "fixed_basis_ttt": "adaptive_descriptor_convex_difficulty_v2",
        # Backward-compatible field: now records the realised rather than
        # requested count.  ``fixed_basis_budget_max`` is the hard cap.
        "fixed_basis_budget": int(len(selected)),
        "fixed_basis_budget_max": int(budget),
        "fixed_basis_budget_at_300_frames": int(budget_at_300_frames),
        "fixed_basis_budget_scale_frames": int(frames),
        "fixed_basis_selected": int(len(selected)),
        "fixed_basis_candidate_count": int(len(candidate)),
        "fixed_basis_event_token_count": int(event_mask.sum()),
        "fixed_basis_residual_threshold": float(residual_threshold),
        "fixed_basis_sufficient_residual_bound": float(sufficient_bound),
        "fixed_basis_threshold_factor": float(safety_factor),
        "fixed_basis_score_mean": float(score.mean()),
        "fixed_basis_score_p95": float(torch.quantile(score, .95)),
        "fixed_basis_score_max": float(score.max()),
        "fixed_basis_event_score_p70": (
            float(torch.quantile(event_scores, .70)) if event_scores.numel() else 0.0
        ),
        "fixed_basis_event_score_p75": (
            float(torch.quantile(event_scores, .75)) if event_scores.numel() else 0.0
        ),
        "fixed_basis_sample_stride": int(sample_stride),
        "fixed_basis_training_samples": int(len(train_ids)),
        "fixed_basis_training_steps": int(steps),
        "fixed_basis_final_loss": float(loss.detach()),
        "fixed_basis_ttt_ms": (time.perf_counter() - started) * 1000.0,
    }


@torch.no_grad()
def extract_vggt_patch_descriptors(
    aggregator,
    images: torch.Tensor,
    patch_hw: tuple[int, int],
    *,
    descriptor_dim: int = 64,
    chunk_frames: int = 32,
    projection_seed: int = 0,
    projection_mode: str = "random",
    return_full_tokens: bool = False,
) -> tuple[torch.Tensor, dict[str, float | str], torch.Tensor | None]:
    """Extract deterministic low-dimensional descriptors for every patch.

    ``images`` is ``[T,C,H,W]``.  The frozen VGGT patch encoder is evaluated
    in chunks and its 1024-D output is reduced by a fixed Rademacher
    projection.  No parameter is trained and the global RNG state is not
    modified.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must be [T,3,H,W]")
    if descriptor_dim < 8 or chunk_frames < 1:
        raise ValueError("descriptor_dim must be >= 8 and chunk_frames positive")
    if projection_mode not in {"subsample", "random"}:
        raise ValueError("projection_mode must be subsample or random")
    if images.shape[-2] // aggregator.patch_size != patch_hw[0] or images.shape[-1] // aggregator.patch_size != patch_hw[1]:
        raise ValueError("patch_hw does not match image size and patch size")

    started = time.perf_counter()
    device = images.device
    mean = aggregator._resnet_mean[0].to(device=device, dtype=torch.float32)
    std = aggregator._resnet_std[0].to(device=device, dtype=torch.float32)
    rows: list[torch.Tensor] = []
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    for begin in range(0, len(images), chunk_frames):
        normalized = (images[begin:begin + chunk_frames].float() - mean) / std
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=device.type == "cuda"):
            tokens = aggregator.patch_embed(normalized)
            if isinstance(tokens, dict):
                tokens = tokens["x_norm_patchtokens"]
        rows.append(tokens.detach())
    encoded = torch.cat(rows, dim=0).contiguous()
    expected_patches = patch_hw[0] * patch_hw[1]
    if encoded.ndim != 3 or encoded.shape[1] != expected_patches:
        raise RuntimeError(
            f"patch encoder returned {tuple(encoded.shape)}, expected [T,{expected_patches},C]"
        )

    channels = encoded.shape[-1]
    if descriptor_dim < channels:
        if projection_mode == "subsample":
            channel_ids = torch.linspace(
                0, channels - 1, descriptor_dim, device=device,
            ).round().long().unique()
            if len(channel_ids) != descriptor_dim:
                raise RuntimeError("descriptor channel sampling produced duplicates")
            projected = encoded.index_select(-1, channel_ids)
        else:
            # Construct on CPU with an isolated generator so descriptor
            # extraction cannot change model or experiment RNG state.
            generator = torch.Generator(device="cpu")
            generator.manual_seed(projection_seed)
            cache_key = (str(device), channels, descriptor_dim, projection_seed)
            projection = _PROJECTION_CACHE.get(cache_key)
            if projection is None:
                projection_cpu = torch.empty((channels, descriptor_dim), dtype=torch.float32)
                projection_cpu.bernoulli_(.5, generator=generator).mul_(2).sub_(1)
                projection_cpu.mul_(1.0 / math.sqrt(descriptor_dim))
                projection = projection_cpu.to(device=device)
                _PROJECTION_CACHE[cache_key] = projection
            projected = encoded.float() @ projection
    else:
        projected = encoded
    descriptors = normalize_rows(projected).reshape(
        len(images), patch_hw[0], patch_hw[1], -1,
    ).contiguous()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return descriptors, {
        "descriptor_source": "frozen_vggt_patch_encoder",
        "descriptor_dim": float(descriptors.shape[-1]),
        "descriptor_chunk_frames": float(chunk_frames),
        "descriptor_projection_seed": float(projection_seed),
        "descriptor_projection_mode": projection_mode,
        "descriptor_ms": (time.perf_counter() - started) * 1000.0,
        "patch_tokens_reused_by_main_forward": bool(return_full_tokens),
    }, (encoded if return_full_tokens else None)


def _update_two_best(
    best: torch.Tensor,
    second: torch.Tensor,
    best_dy: torch.Tensor,
    best_dx: torch.Tensor,
    similarity: torch.Tensor,
    dy: int,
    dx: int,
    row_slice: slice,
    column_slice: slice,
) -> None:
    current_best = best[:, row_slice, column_slice]
    current_second = second[:, row_slice, column_slice]
    better = similarity > current_best
    updated_second = torch.where(
        better, current_best, torch.maximum(current_second, similarity),
    )
    best[:, row_slice, column_slice] = torch.where(better, similarity, current_best)
    second[:, row_slice, column_slice] = updated_second
    best_dy[:, row_slice, column_slice] = torch.where(
        better, torch.full_like(best_dy[:, row_slice, column_slice], dy),
        best_dy[:, row_slice, column_slice],
    )
    best_dx[:, row_slice, column_slice] = torch.where(
        better, torch.full_like(best_dx[:, row_slice, column_slice], dx),
        best_dx[:, row_slice, column_slice],
    )


@torch.no_grad()
def adjacent_token_correspondences(
    descriptors: torch.Tensor,
    *,
    search_radius: int = 4,
    minimum_similarity: float = .35,
    cycle_scale: float = 1.5,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    dict[str, float | str],
]:
    """Return bidirectional patch-grid displacement and confidence.

    Matching is exact within a fixed local window.  Forward and backward top
    two candidates are accumulated in one sweep over displacements, avoiding
    a quadratic all-token affinity matrix.  Confidence combines cosine
    similarity, the top-1 margin and a soft forward/backward cycle error.
    """
    if descriptors.ndim != 4:
        raise ValueError("descriptors must be [T,H,W,C]")
    if len(descriptors) < 2 or search_radius < 0 or cycle_scale <= 0:
        raise ValueError("token matching needs >=2 frames and valid search settings")
    if not bool(torch.isfinite(descriptors).all()):
        raise ValueError("token descriptors must be finite")
    started = time.perf_counter()
    frames, height, width, _ = descriptors.shape
    pairs = frames - 1
    source, target = descriptors[:-1].float(), descriptors[1:].float()
    shape = (pairs, height, width)
    radius_y = min(search_radius, height - 1)
    radius_x = min(search_radius, width - 1)
    fused_match = (
        triton_ready(descriptors)
        and os.environ.get("VGGT_TRITON_LOCAL_MATCH", "1") != "0"
    )
    if fused_match:
        forward_best, forward_second, forward_dy, forward_dx = local_top2_match(
            source, target, radius_y, radius_x,
        )
        backward_best, backward_second, backward_dy, backward_dx = local_top2_match(
            target, source, radius_y, radius_x, reverse_order=True,
        )
    else:
        negative = torch.finfo(torch.float32).min
        forward_best = torch.full(shape, negative, device=descriptors.device)
        forward_second = torch.full_like(forward_best, negative)
        backward_best = torch.full_like(forward_best, negative)
        backward_second = torch.full_like(forward_best, negative)
        forward_dy = torch.zeros(shape, device=descriptors.device, dtype=torch.long)
        forward_dx = torch.zeros_like(forward_dy)
        backward_dy = torch.zeros_like(forward_dy)
        backward_dx = torch.zeros_like(forward_dy)
        for dy in range(-radius_y, radius_y + 1):
            source_y = slice(max(0, -dy), min(height, height - dy))
            target_y = slice(max(0, dy), min(height, height + dy))
            for dx in range(-radius_x, radius_x + 1):
                source_x = slice(max(0, -dx), min(width, width - dx))
                target_x = slice(max(0, dx), min(width, width + dx))
                similarity = (
                    source[:, source_y, source_x]
                    * target[:, target_y, target_x]
                ).sum(dim=-1)
                _update_two_best(
                    forward_best, forward_second, forward_dy, forward_dx,
                    similarity, dy, dx, source_y, source_x,
                )
                _update_two_best(
                    backward_best, backward_second, backward_dy, backward_dx,
                    similarity, -dy, -dx, target_y, target_x,
                )

    yy, xx = torch.meshgrid(
        torch.arange(height, device=descriptors.device),
        torch.arange(width, device=descriptors.device), indexing="ij",
    )
    yy = yy[None].expand(pairs, -1, -1)
    xx = xx[None].expand(pairs, -1, -1)
    pair_index = torch.arange(pairs, device=descriptors.device)[:, None, None]

    forward_target_y = (yy + forward_dy).clamp(0, height - 1)
    forward_target_x = (xx + forward_dx).clamp(0, width - 1)
    return_backward_dy = backward_dy[pair_index, forward_target_y, forward_target_x]
    return_backward_dx = backward_dx[pair_index, forward_target_y, forward_target_x]
    forward_cycle = torch.sqrt(
        (forward_dy + return_backward_dy).float().square()
        + (forward_dx + return_backward_dx).float().square()
    )

    backward_source_y = (yy + backward_dy).clamp(0, height - 1)
    backward_source_x = (xx + backward_dx).clamp(0, width - 1)
    return_forward_dy = forward_dy[pair_index, backward_source_y, backward_source_x]
    return_forward_dx = forward_dx[pair_index, backward_source_y, backward_source_x]
    backward_cycle = torch.sqrt(
        (backward_dy + return_forward_dy).float().square()
        + (backward_dx + return_forward_dx).float().square()
    )

    def confidence(best: torch.Tensor, second: torch.Tensor, cycle: torch.Tensor) -> torch.Tensor:
        similarity = ((best - minimum_similarity) / (1.0 - minimum_similarity)).clamp(0, 1)
        margin = torch.sigmoid(((best - second) - .005) / .01)
        cycle_term = torch.exp(-.5 * (cycle / cycle_scale).square())
        return (similarity * margin * cycle_term).clamp(0, 1)

    forward_confidence = confidence(forward_best, forward_second, forward_cycle)
    backward_confidence = confidence(backward_best, backward_second, backward_cycle)
    forward_flow = torch.stack((forward_dx, forward_dy), dim=-1).float()
    backward_flow = torch.stack((backward_dx, backward_dy), dim=-1).float()
    if descriptors.device.type == "cuda":
        torch.cuda.synchronize(descriptors.device)
    return forward_flow, backward_flow, forward_confidence, backward_confidence, {
        "flow_backend": "token_match",
        "correspondence_backend": "local_patch_token_match",
        "flow_ms": (time.perf_counter() - started) * 1000.0,
        "token_match_ms": (time.perf_counter() - started) * 1000.0,
        "token_match_search_radius": float(search_radius),
        "token_match_candidate_count": float((2 * radius_y + 1) * (2 * radius_x + 1)),
        "token_match_fused_kernel": fused_match,
        "token_match_forward_similarity_mean": float(forward_best.mean()),
        "token_match_backward_similarity_mean": float(backward_best.mean()),
        "token_match_forward_cycle_mean": float(forward_cycle.mean()),
        "token_match_backward_cycle_mean": float(backward_cycle.mean()),
        "forward_confident_ratio": float((forward_confidence > .05).float().mean()),
        "backward_confident_ratio": float((backward_confidence > .05).float().mean()),
        "pair_count": float(pairs),
    }


class SequenceTokenMotionAdapter(nn.Module):
    """Small per-clip model of the dominant, stable token motion field."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 4),
        )
        # Zero initial motion is a deterministic and conservative prior.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, inputs: torch.Tensor, motion_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        motion = motion_scale * torch.tanh(self.network(inputs))
        return motion[..., :2], motion[..., 2:]


def _robust_unit(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    selected = value[mask]
    if not selected.numel():
        return torch.zeros_like(value)
    scale = torch.quantile(selected.float(), .90).clamp_min(1e-6)
    return (value.float() / scale).clamp(0, 1)


def _motion_adapter_loss(
    model: SequenceTokenMotionAdapter,
    inputs: torch.Tensor,
    forward_input_ids: torch.Tensor,
    backward_input_ids: torch.Tensor,
    forward_targets: torch.Tensor,
    backward_targets: torch.Tensor,
    forward_weights: torch.Tensor,
    backward_weights: torch.Tensor,
    initial_parameters: list[torch.Tensor],
    motion_scale: float,
) -> torch.Tensor:
    predicted_forward, _ = model(inputs.index_select(0, forward_input_ids), motion_scale)
    _, predicted_backward = model(inputs.index_select(0, backward_input_ids), motion_scale)
    forward_error = F.smooth_l1_loss(
        predicted_forward, forward_targets, reduction="none", beta=.25,
    ).sum(dim=-1)
    backward_error = F.smooth_l1_loss(
        predicted_backward, backward_targets, reduction="none", beta=.25,
    ).sum(dim=-1)
    data_loss = (
        (forward_error * forward_weights).sum()
        / forward_weights.sum().clamp_min(1e-6)
        + (backward_error * backward_weights).sum()
        / backward_weights.sum().clamp_min(1e-6)
    ) * .5
    parameter_drift = sum(
        (parameter - initial).square().mean()
        for parameter, initial in zip(model.parameters(), initial_parameters)
    )
    return data_loss + 1e-5 * parameter_drift


def _fit_motion_adapter(
    model: SequenceTokenMotionAdapter,
    inputs: torch.Tensor,
    forward_input_ids: torch.Tensor,
    backward_input_ids: torch.Tensor,
    forward_targets: torch.Tensor,
    backward_targets: torch.Tensor,
    forward_weights: torch.Tensor,
    backward_weights: torch.Tensor,
    initial_parameters: list[torch.Tensor],
    motion_scale: float,
    steps: int,
    learning_rate: float,
) -> tuple[list[float], str]:
    """Fit the per-sequence adapter, graph-capturing the fixed CUDA step."""
    use_graph = (
        inputs.is_cuda
        and os.environ.get("VGGT_TTT_CUDA_GRAPH", "1") != "0"
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4,
        fused=inputs.is_cuda, capturable=use_graph,
    )

    def step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        loss = _motion_adapter_loss(
            model, inputs, forward_input_ids, backward_input_ids,
            forward_targets, backward_targets, forward_weights, backward_weights,
            initial_parameters, motion_scale,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
        optimizer.step()
        return loss

    if not use_graph:
        losses = []
        for _ in range(steps):
            loss = step()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("token-motion TTT produced a non-finite loss")
            losses.append(float(loss.detach()))
        return losses, "eager_adamw"

    # Allocate fused-AdamW state outside capture, then restore the exact model
    # and zero optimizer state.  This warm-up has no algorithmic effect.
    saved_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    warmup_stream = torch.cuda.Stream(device=inputs.device)
    warmup_stream.wait_stream(torch.cuda.current_stream(inputs.device))
    with torch.cuda.stream(warmup_stream):
        warmup_loss = step()
    torch.cuda.current_stream(inputs.device).wait_stream(warmup_stream)
    if not bool(torch.isfinite(warmup_loss)):
        raise RuntimeError("token-motion TTT graph warm-up produced a non-finite loss")
    del warmup_loss
    with torch.no_grad():
        for parameter, saved in zip(model.parameters(), saved_parameters):
            parameter.copy_(saved)
        for state in optimizer.state.values():
            for value in state.values():
                if torch.is_tensor(value):
                    value.zero_()
    optimizer.zero_grad(set_to_none=True)

    graph = torch.cuda.CUDAGraph()
    loss_slot = torch.empty((), device=inputs.device, dtype=torch.float32)
    with torch.cuda.graph(graph):
        captured_loss = step()
        loss_slot.copy_(captured_loss.detach())
    initial_loss = float(loss_slot)
    # Capture records and materializes the static buffers, but optimizer state
    # advances only on replay.  Execute exactly the requested update count.
    for _ in range(steps):
        graph.replay()
    final_loss = float(loss_slot)
    if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
        raise RuntimeError("token-motion TTT CUDA graph produced a non-finite loss")
    if steps == 1:
        return [initial_loss], "cuda_graph_fused_adamw"
    return [initial_loss, final_loss], "cuda_graph_fused_adamw"


@torch.enable_grad()
def adapt_token_motion_ttt(
    anchor_xyz: torch.Tensor,
    anchor_indices: tuple[int, ...],
    descriptors: torch.Tensor,
    forward_flow: torch.Tensor,
    backward_flow: torch.Tensor,
    forward_confidence: torch.Tensor,
    backward_confidence: torch.Tensor,
    *,
    time_weight: float = .25,
    steps: int = 12,
    learning_rate: float = 3e-3,
    hidden_dim: int = 32,
    sample_ratio: float = .10,
    max_samples: int = 32768,
    uncertain_confidence: float = .10,
    max_uncertain_ratio: float = .02,
    max_sensitive_ratio: float = .20,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float | str | bool]]:
    """Fit a sequence motion prior and derive flow-free 4-D sensitivity.

    The adapter is reset for every clip and trained only against reliable
    token correspondences.  It has no classification head and never receives
    binary pseudo labels.  Tokens become sensitive when their observed motion
    is poorly explained by the learned dominant field, when they lie on a
    motion boundary, or when their correspondence is unreliable.
    """
    started = time.perf_counter()
    if steps < 1 or learning_rate <= 0 or hidden_dim < 4:
        raise ValueError("token-motion TTT needs positive steps/lr and hidden_dim >= 4")
    if (not 0 < sample_ratio <= 1 or max_samples < 1
            or not 0 < max_sensitive_ratio < 1):
        raise ValueError("invalid token-motion TTT sampling settings")
    frames, height, width, descriptor_dim = descriptors.shape
    expected_flow = (frames - 1, height, width, 2)
    if forward_flow.shape != expected_flow or backward_flow.shape != expected_flow:
        raise ValueError("token correspondence and descriptor grids do not match")
    if anchor_xyz.shape != (len(anchor_indices), height, width, 3):
        raise ValueError("anchor geometry and token descriptor grids do not match")
    device = descriptors.device

    # Token correspondences also propagate sparse anchor geometry.  No dense
    # optical flow or learned flow model is involved in this interpolation.
    geometry, geometry_stats = optimize_anchor_geometry_ttt(
        anchor_xyz.clone(), anchor_indices, forward_flow, backward_flow,
        backward_confidence, forward_confidence=forward_confidence,
        time_weight=time_weight, iterations=0, smoothness=.5,
        data_weight=.5, relaxation=.25,
    )

    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, height, device=device),
        torch.linspace(-1, 1, width, device=device), indexing="ij",
    )
    spatial = torch.stack((xx, yy), dim=-1)[None].expand(frames, -1, -1, -1)
    temporal = torch.linspace(-1, 1, frames, device=device)[:, None, None, None]
    temporal = temporal.expand(-1, height, width, -1)
    inputs = torch.cat((descriptors.float(), spatial, temporal), dim=-1).reshape(
        frames * height * width, descriptor_dim + 3,
    ).detach()
    motion_scale = max(
        1.0,
        float(torch.quantile(forward_flow.norm(dim=-1).flatten(), .95)),
        float(torch.quantile(backward_flow.norm(dim=-1).flatten(), .95)),
    )

    # Select reliable observations deterministically; this keeps adaptation
    # linear in token count and independent of the global RNG state.
    pair_tokens = (frames - 1) * height * width
    sample_count = min(max_samples, max(1, int(round(sample_ratio * pair_tokens))))
    forward_ids = forward_confidence.flatten().topk(sample_count).indices
    backward_ids = backward_confidence.flatten().topk(sample_count).indices
    patch_count = height * width
    forward_frame = torch.div(forward_ids, patch_count, rounding_mode="floor")
    backward_pair_frame = torch.div(backward_ids, patch_count, rounding_mode="floor")
    forward_input_ids = forward_ids
    backward_input_ids = (backward_pair_frame + 1) * patch_count + backward_ids.remainder(patch_count)
    forward_targets = forward_flow.reshape(-1, 2).index_select(0, forward_ids)
    backward_targets = backward_flow.reshape(-1, 2).index_select(0, backward_ids)
    forward_weights = forward_confidence.flatten().index_select(0, forward_ids)
    backward_weights = backward_confidence.flatten().index_select(0, backward_ids)

    rng_devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(seed)
        model = SequenceTokenMotionAdapter(inputs.shape[-1], hidden_dim).to(device)
    initial_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    losses, optimizer_backend = _fit_motion_adapter(
        model, inputs, forward_input_ids, backward_input_ids,
        forward_targets, backward_targets, forward_weights, backward_weights,
        initial_parameters, motion_scale, steps, learning_rate,
    )

    model.eval()
    with torch.no_grad():
        chunks: list[torch.Tensor] = []
        prediction_chunk = 65536
        for begin in range(0, len(inputs), prediction_chunk):
            predicted_forward, predicted_backward = model(
                inputs[begin:begin + prediction_chunk], motion_scale,
            )
            chunks.append(torch.cat((predicted_forward, predicted_backward), dim=-1))
        predicted = torch.cat(chunks).reshape(frames, height, width, 4)
        fused_postprocess = (
            triton_ready(predicted)
            and os.environ.get("VGGT_TRITON_TTT_POSTPROCESS", "1") != "0"
        )
        if fused_postprocess:
            residual, confidence, boundary = fused_motion_cues(
                forward_flow, backward_flow, predicted,
                forward_confidence, backward_confidence,
            )
        else:
            predicted_forward, predicted_backward = predicted[..., :2], predicted[..., 2:]
            forward_residual = (forward_flow - predicted_forward[:-1]).norm(dim=-1)
            backward_residual = (backward_flow - predicted_backward[1:]).norm(dim=-1)
            residual = torch.zeros((frames, height, width), device=device)
            residual_count = torch.zeros_like(residual)
            residual[:-1] += forward_residual
            residual_count[:-1] += 1
            residual[1:] += backward_residual
            residual_count[1:] += 1
            residual /= residual_count.clamp_min(1)
            confidence = torch.ones_like(residual)
            confidence[:-1] = torch.minimum(confidence[:-1], forward_confidence)
            confidence[1:] = torch.minimum(confidence[1:], backward_confidence)
            forward_average = F.avg_pool2d(
                forward_flow.permute(0, 3, 1, 2), 3, stride=1, padding=1,
            ).permute(0, 2, 3, 1)
            backward_average = F.avg_pool2d(
                backward_flow.permute(0, 3, 1, 2), 3, stride=1, padding=1,
            ).permute(0, 2, 3, 1)
            boundary = torch.zeros_like(residual)
            boundary[:-1] += (forward_flow - forward_average).norm(dim=-1)
            boundary[1:] += (backward_flow - backward_average).norm(dim=-1)
            boundary /= residual_count.clamp_min(1)
        novelty = 1.0 - confidence

        later = torch.ones_like(residual, dtype=torch.bool)
        later[0] = False
        reliable = later & (confidence >= uncertain_confidence)

        def robust_threshold(value: torch.Tensor, floor: float) -> torch.Tensor:
            selected = value[reliable]
            if not selected.numel():
                return value.new_tensor(floor)
            median = selected.median()
            mad = (selected - median).abs().median()
            return torch.maximum(
                value.new_tensor(floor), median + 3.0 * 1.4826 * mad,
            )

        residual_threshold = robust_threshold(residual, .5)
        boundary_threshold = robust_threshold(boundary, .5)
        residual_selected = residual[later]
        boundary_selected = boundary[later]
        residual_scale = (
            torch.quantile(residual_selected.float(), .90).clamp_min(1e-6)
            if residual_selected.numel() else residual.new_tensor(1.0)
        )
        boundary_scale = (
            torch.quantile(boundary_selected.float(), .90).clamp_min(1e-6)
            if boundary_selected.numel() else boundary.new_tensor(1.0)
        )
        if fused_postprocess:
            sensitivity, low_confidence, raw_event = fused_sensitivity_labels(
                residual, confidence, boundary,
                uncertain_confidence=uncertain_confidence,
                residual_scale=residual_scale, boundary_scale=boundary_scale,
                residual_threshold=residual_threshold,
                boundary_threshold=boundary_threshold,
            )
        else:
            sensitivity = (
                .45 * novelty
                + .35 * _robust_unit(residual, later)
                + .20 * _robust_unit(boundary, later)
            ).clamp(0, 1)
            low_confidence = later & (confidence < uncertain_confidence)
            raw_event = later & (
                ((residual >= residual_threshold) & (confidence >= uncertain_confidence))
                | (boundary >= boundary_threshold)
                | (novelty >= .55)
            )
        uncertain = torch.zeros_like(low_confidence)
        uncertain_candidates = low_confidence.flatten().nonzero().flatten()
        uncertain_budget = min(
            len(uncertain_candidates),
            max(1, int(round(max_uncertain_ratio * residual.numel())))
            if max_uncertain_ratio > 0 else 0,
        )
        if uncertain_budget:
            chosen = uncertain_candidates.index_select(
                0, sensitivity.flatten().index_select(
                    0, uncertain_candidates,
                ).topk(uncertain_budget).indices,
            )
            uncertain.flatten()[chosen] = True
        raw_event &= ~uncertain
        # Token-level correspondences are intentionally conservative: weak
        # texture and ordinary parallax can trigger the raw robust tests.  A
        # clip-level cap retains only the strongest unexplained events while
        # allowing genuinely simple clips to use fewer sensitive tokens.
        event = raw_event.clone()
        sensitive_budget = int(round(max_sensitive_ratio * residual.numel()))
        if int(event.sum()) > sensitive_budget:
            candidate = event.flatten().nonzero().flatten()
            keep = candidate.index_select(
                0, sensitivity.flatten().index_select(0, candidate).topk(
                    sensitive_budget,
                ).indices,
            )
            event.zero_()
            event.flatten()[keep] = True
        labels = torch.zeros_like(residual, dtype=torch.long)
        labels[event] = 1
        labels[uncertain] = 2
        labels[0] = 0

    parameter_delta = math.sqrt(sum(
        float((parameter.detach() - initial).square().sum())
        for parameter, initial in zip(model.parameters(), initial_parameters)
    ))
    return (
        geometry,
        labels.flatten().contiguous(),
        sensitivity.flatten().contiguous(),
        {
            "ttt_mode": "flow_free_sequence_token_motion_adapter",
            "adapter_reset_per_sequence": True,
            "backbone_frozen": True,
            "binary_pseudo_labels_used": False,
            "adapter_trainable_parameters": float(sum(p.numel() for p in model.parameters())),
            "adapter_hidden_dim": float(hidden_dim),
            "adapter_steps": float(steps),
            "adapter_learning_rate": float(learning_rate),
            "adapter_optimizer_backend": optimizer_backend,
            "ttt_fused_postprocess": fused_postprocess,
            "adapter_sample_ratio": float(sample_ratio),
            "adapter_sample_count_per_direction": float(sample_count),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "loss_reduction_fraction": (losses[0] - losses[-1]) / max(losses[0], 1e-8),
            "parameter_delta_l2": parameter_delta,
            "motion_scale": motion_scale,
            "motion_residual_mean": float(residual.mean()),
            "motion_residual_threshold": float(residual_threshold),
            "motion_boundary_mean": float(boundary.mean()),
            "motion_boundary_threshold": float(boundary_threshold),
            "confidence_mean": float(confidence.mean()),
            "raw_sensitive_ratio": float(raw_event.float().mean()),
            "max_sensitive_ratio": float(max_sensitive_ratio),
            "stable_ratio": float((labels == 0).float().mean()),
            "sensitive_ratio": float((labels == 1).float().mean()),
            "uncertain_ratio": float((labels == 2).float().mean()),
            "sensitivity_mean": float(sensitivity.mean()),
            "initial_geometry_source": "bidirectional_token_match_anchor_interpolation",
            "initial_flow_valid_ratio": geometry_stats["flow_valid_ratio"],
            "integer_gather_propagation": geometry_stats.get(
                "integer_gather_propagation", False,
            ),
            "ttt_wall_ms": (time.perf_counter() - started) * 1000.0,
        },
    )
