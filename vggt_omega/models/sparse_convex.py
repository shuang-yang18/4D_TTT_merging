"""Inference-only sparse-convex attention plan for VGGT-Omega."""
from __future__ import annotations

import atexit
import os

import torch
import torch.nn.functional as F

from .token_merging import (
    FAST_MATCH_SOURCE_CHUNK,
    WeightedBilinearTokenPlan,
    deterministic_weighted_normalizer,
)
from .sparse_convex_kernels import (
    four_support_active_set,
    four_support_projected_gradient,
    streaming_top4,
    triton_acceleration_available,
)

_ACTIVE_SETS = torch.tensor((
    (1., 1., 1., 1.),
    (1., 1., 1., 0.), (1., 1., 0., 1.), (1., 0., 1., 1.), (0., 1., 1., 1.),
    (1., 1., 0., 0.), (1., 0., 1., 0.), (1., 0., 0., 1.),
    (0., 1., 1., 0.), (0., 1., 0., 1.), (0., 0., 1., 1.),
    (1., 0., 0., 0.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.),
))
_RIDGE = 1e-4
_EARLY_MARGIN = 1e-2
_LOCAL_FRAME_GROUP = 8
_HASH_PROJECTIONS: dict[tuple[str, int, int], torch.Tensor] = {}
_PROFILE_ENABLED = os.environ.get("VGGT_PROFILE_SPARSE_CONVEX", "0") == "1"
_PROFILE_EVENTS: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
_TTT_TRACE_PATH = os.environ.get("VGGT_TTT_TRACE_PATH")
_TTT_TRACE_RECORDS: list[dict[str, torch.Tensor | int | float]] = []
_TTT_FIXED_DYNAMIC_CACHE: dict[str, torch.Tensor] = {}


def _profile_begin() -> torch.cuda.Event | None:
    if not _PROFILE_ENABLED or not torch.cuda.is_available():
        return None
    event = torch.cuda.Event(enable_timing=True)
    event.record()
    return event


def _profile_end(label: str, begin: torch.cuda.Event | None) -> None:
    if begin is None:
        return
    end = torch.cuda.Event(enable_timing=True)
    end.record()
    _PROFILE_EVENTS.append((label, begin, end))


def _print_profile() -> None:
    if not _PROFILE_EVENTS:
        return
    try:
        torch.cuda.synchronize()
        totals: dict[str, list[float]] = {}
        for label, begin, end in _PROFILE_EVENTS:
            values = totals.setdefault(label, [0.0, 0.0])
            values[0] += begin.elapsed_time(end)
            values[1] += 1
        for label, (total, calls) in sorted(totals.items()):
            print(f"[sparse-convex-profile] {label} total_ms={total:.3f} "
                  f"calls={int(calls)} mean_ms={total / calls:.3f}", flush=True)
    except Exception as error:  # pragma: no cover - interpreter shutdown safety
        print(f"[sparse-convex-profile] unavailable: {error}", flush=True)


if _PROFILE_ENABLED:
    atexit.register(_print_profile)


def _save_ttt_trace() -> None:
    if not _TTT_TRACE_PATH or not _TTT_TRACE_RECORDS:
        return
    torch.save({"layers": _TTT_TRACE_RECORDS}, _TTT_TRACE_PATH)


if _TTT_TRACE_PATH:
    atexit.register(_save_ttt_trace)


def _simplex_project(value: torch.Tensor) -> torch.Tensor:
    ordered, _ = value.sort(dim=-1, descending=True)
    cumulative = ordered.cumsum(dim=-1) - 1
    count = torch.arange(1, value.shape[-1] + 1, device=value.device, dtype=value.dtype)
    active = ordered - cumulative / count > 0
    rho = active.sum(dim=-1, keepdim=True).clamp_min(1) - 1
    theta = cumulative.gather(-1, rho) / (rho.to(value.dtype) + 1)
    return (value - theta).clamp_min(0)


def _joint_metric_routes(
    left_features: torch.Tensor,
    left_geometry: torch.Tensor,
    right_features: torch.Tensor,
    right_geometry: torch.Tensor,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor feature cosine minus weighted squared 4-D distance as one GEMM.

    The returned matrices satisfy ``left_route @ right_route.T ==
    left_features @ right_features.T - weight * ||g_l-g_r||^2`` up to the
    selected floating-point precision.  Keeping the geometry term inside the
    existing GEMM avoids materialising a second source-by-basis distance
    matrix on long clips.
    """
    if weight <= 0:
        return left_features, right_features
    scale = (2.0 * weight) ** .5
    left_norm = -weight * left_geometry.float().square().sum(dim=-1, keepdim=True)
    right_norm = -weight * right_geometry.float().square().sum(dim=-1, keepdim=True)
    left_route = torch.cat((
        left_features.float(), scale * left_geometry.float(), left_norm,
        torch.ones_like(left_norm),
    ), dim=-1)
    right_route = torch.cat((
        right_features.float(), scale * right_geometry.float(),
        torch.ones_like(right_norm), right_norm,
    ), dim=-1)
    return left_route, right_route


@torch.no_grad()
def _adaptive_fit(
    features: torch.Tensor, basis_ids: torch.Tensor, *, max_support: int,
    mode: str, coefficient_limit: float, chunk: int = 1024, steps: int = 32,
) -> dict[str, torch.Tensor]:
    basis = features.index_select(0, basis_ids)
    all_ids, all_weights, all_cosine, all_residual, all_norm = [], [], [], [], []
    for begin in range(0, len(features), chunk):
        profile = _profile_begin()
        source = features[begin:begin + chunk].float()
        score = source.to(torch.bfloat16) @ basis.to(torch.bfloat16).T
        k = min(max_support, len(basis_ids))
        ranking = score.abs() if mode == "adaptive_linear" else score
        local = ranking.topk(k, dim=-1).indices
        selected_ids = basis_ids.index_select(0, local.reshape(-1)).reshape(local.shape)
        supports = features.index_select(0, selected_ids.reshape(-1)).reshape(len(source), k, -1).float()
        _profile_end("legacy.support_search", profile)
        profile = _profile_begin()
        alpha = torch.zeros((len(source), k), device=source.device, dtype=torch.float32)
        if mode == "adaptive_convex":
            alpha[:, 0] = 1.0
        gram = supports @ supports.transpose(1, 2)
        rhs = (supports @ source.unsqueeze(-1)).squeeze(-1)
        step = (.95 / gram.abs().sum(dim=-1).amax(dim=-1).clamp_min(1e-4)).unsqueeze(-1)
        for _ in range(steps):
            gradient = (gram @ alpha.unsqueeze(-1)).squeeze(-1) - rhs
            proposal = alpha - step * gradient
            alpha = (_simplex_project(proposal) if mode == "adaptive_convex"
                     else proposal.clamp(-coefficient_limit, coefficient_limit))
        reconstruction = (supports * alpha.unsqueeze(-1)).sum(dim=1)
        all_ids.append(F.pad(selected_ids, (0, max_support - k), value=-1))
        all_weights.append(F.pad(alpha, (0, max_support - k)))
        all_cosine.append(F.cosine_similarity(source, reconstruction, dim=-1, eps=1e-8))
        all_residual.append((source - reconstruction).norm(dim=-1))
        all_norm.append(reconstruction.norm(dim=-1))
        _profile_end("legacy.convex_and_certificate", profile)
    return {"support_ids": torch.cat(all_ids), "weights": torch.cat(all_weights),
            "cosine": torch.cat(all_cosine), "residual": torch.cat(all_residual),
            "reconstruction_norm": torch.cat(all_norm)}


@torch.no_grad()
def _adaptive_fit_fused(
    features: torch.Tensor, basis_ids: torch.Tensor, *, max_support: int,
    threshold: float, source_ids: torch.Tensor | None = None,
    geometry: torch.Tensor | None = None, geometry_weight: float = 0.0,
    geometry_guard_tolerance: float | None = None,
    chunk: int = 1024, steps: int = 32,
) -> dict[str, torch.Tensor]:
    """Exact Top-4 retrieval with easy Top-1 acceptance and fused hard solves.

    A source whose best physical basis token clears ``threshold + margin`` has
    an already feasible one-hot convex representation.  Matching the released
    fixed-topology planner, only the remaining hard rows pay for a four-way
    solve.  ``source_ids`` excludes tokens that are guaranteed to stay
    physical, avoiding searches whose results would be discarded.
    """
    if max_support != 4:
        raise ValueError("the fused K-means convex planner currently requires four supports")
    if geometry_weight < 0 or (geometry_weight > 0 and geometry is None):
        raise ValueError("a nonnegative geometry weight and matching 4-D descriptor are required")
    if geometry_guard_tolerance is not None and geometry_guard_tolerance < 0:
        raise ValueError("the 4-D feature guard tolerance must be nonnegative")
    if geometry is not None and geometry.shape != (len(features), 4):
        raise ValueError(f"geometry must have shape {(len(features), 4)}, got {tuple(geometry.shape)}")
    guard_enabled = geometry_weight > 0 and geometry_guard_tolerance is not None
    basis = features.index_select(0, basis_ids)
    if geometry_weight:
        basis_geometry = geometry.index_select(0, basis_ids)
        if guard_enabled:
            # Search once in feature space, then let 4-D geometry rerank only
            # a small semantic shortlist. This is both safer and cheaper than
            # two source-by-dictionary GEMMs.
            basis_bf16 = basis.to(torch.bfloat16).T.contiguous()
        else:
            _, basis_route = _joint_metric_routes(
                basis, basis_geometry, basis, basis_geometry, geometry_weight,
            )
            basis_bf16 = basis_route.to(torch.bfloat16).T.contiguous()
    else:
        basis_bf16 = basis.to(torch.bfloat16).T.contiguous()
    if source_ids is None:
        source_ids = torch.arange(len(features), device=features.device)
    easy_setting = os.environ.get("VGGT_FUSED_EASY_THRESHOLD")
    # The accuracy-validated default solves every candidate.  The one-hot
    # shortcut is an explicit ablation enabled by setting a numeric threshold.
    easy_enabled = easy_setting is not None and easy_setting.lower() not in {"off", "none", "disabled"}
    easy_threshold = (
        float(easy_setting) if easy_enabled
        else 1.0
    )
    if easy_enabled and not threshold <= easy_threshold <= 1.0:
        raise ValueError(
            "VGGT_FUSED_EASY_THRESHOLD must be off or lie between the acceptance threshold and 1"
        )
    all_ids, all_weights, all_cosine, all_residual, all_norm = [], [], [], [], []
    easy_counts = []
    guard_counts, guard_feature_deltas, guard_geometry_gains = [], [], []
    triton_top4 = (
        features.is_cuda and triton_acceleration_available()
        # The streaming kernel bounds score memory, but cuBLAS is faster when
        # the full chunk fits.  Keep it opt-in for genuinely memory-bound runs.
        and os.environ.get("VGGT_TRITON_STREAMING_TOP4", "0") == "1"
    )
    triton_active_set = (
        features.is_cuda and triton_acceleration_available()
        and os.environ.get("VGGT_TRITON_ACTIVE_SET", "0") != "0"
    )

    def solve_weights(gram: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        if triton_active_set:
            return four_support_active_set(gram, rhs, ridge=_RIDGE)
        return four_support_projected_gradient(gram, rhs, steps=steps)

    for begin in range(0, len(source_ids), chunk):
        profile = _profile_begin()
        query_ids = source_ids[begin:begin + chunk]
        source = features.index_select(0, query_ids).float()
        if geometry_weight:
            source_geometry = geometry.index_select(0, query_ids)
            if guard_enabled:
                score = source.to(torch.bfloat16) @ basis_bf16
            else:
                source_route, _ = _joint_metric_routes(
                    source, source_geometry, basis, basis_geometry, geometry_weight,
                )
                if triton_top4:
                    value, local = streaming_top4(source_route, basis_bf16)
                    score = None
                else:
                    score = source_route.to(torch.bfloat16) @ basis_bf16
        else:
            if triton_top4:
                value, local = streaming_top4(source, basis_bf16)
                score = None
            else:
                score = source.to(torch.bfloat16) @ basis_bf16
        if guard_enabled:
            shortlist = min(16, score.shape[-1])
            feature_value, feature_local = score.topk(shortlist, dim=-1)
            candidate_geometry = basis_geometry.index_select(
                0, feature_local.reshape(-1),
            ).reshape(len(source), shortlist, 4)
            joint_value = feature_value.float() - geometry_weight * (
                source_geometry[:, None].float() - candidate_geometry.float()
            ).square().sum(dim=-1)
            joint_in_shortlist = joint_value.topk(4, dim=-1).indices
            # Canonicalise every selected set by feature score. The finite-step
            # simplex solver starts from support zero, so allowing geometry to
            # permute an otherwise identical set would create a numerical
            # change masquerading as a 4-D selection change.
            selected_feature_value = feature_value.gather(1, joint_in_shortlist)
            canonical_order = selected_feature_value.argsort(
                dim=-1, descending=True, stable=True,
            )
            joint_in_shortlist = joint_in_shortlist.gather(1, canonical_order)
            local = feature_local.gather(1, joint_in_shortlist)
            value = feature_value.gather(1, joint_in_shortlist)
            # Ask Top-4 directly rather than slicing Top-16: BF16 ties may be
            # ordered differently by topk for different k, and the fallback
            # must be bit-for-bit equivalent to the released feature route.
            feature_top4 = score.topk(4, dim=-1).indices
            feature_selected_ids = basis_ids.index_select(
                0, feature_top4.reshape(-1),
            ).reshape(len(source), 4)
        elif score is not None:
            value, local = score.topk(4, dim=-1)
        selected_ids = basis_ids.index_select(0, local.reshape(-1)).reshape(local.shape)
        _profile_end("fused.support_search", profile)
        profile = _profile_begin()

        # Pareto-safe 4-D retrieval: compare the geometry-aware candidate set
        # against the feature-only Top-4 under the same physical dictionary.
        # A 4-D route is accepted only when its fitted feature residual/cosine
        # is no worse (up to the explicit tolerance) and its 4-D residual is
        # strictly smaller. This gives each token an immediate feature-only
        # fallback instead of trusting sequence-level flow quality blindly.
        if guard_enabled:
            def solve(
                candidate_ids: torch.Tensor,
                solve_source: torch.Tensor,
                solve_geometry: torch.Tensor,
            ) -> tuple[torch.Tensor, ...]:
                rows = len(candidate_ids)
                supports = features.index_select(0, candidate_ids.reshape(-1)).reshape(
                    rows, 4, -1,
                ).float()
                gram = supports @ supports.transpose(1, 2)
                rhs = (supports @ solve_source.unsqueeze(-1)).squeeze(-1)
                weights = solve_weights(gram, rhs)
                reconstruction = (supports * weights.unsqueeze(-1)).sum(dim=1)
                candidate_geometry = geometry.index_select(
                    0, candidate_ids.reshape(-1),
                ).reshape(rows, 4, 4).float()
                geometry_reconstruction = (
                    candidate_geometry * weights.unsqueeze(-1)
                ).sum(dim=1)
                geometry_residual = (
                    solve_geometry.float() - geometry_reconstruction
                ).norm(dim=-1)
                return (
                    weights,
                    F.cosine_similarity(solve_source, reconstruction, dim=-1, eps=1e-8),
                    (solve_source - reconstruction).norm(dim=-1),
                    reconstruction.norm(dim=-1),
                    geometry_residual,
                )

            feature_only = solve(feature_selected_ids, source, source_geometry)
            # Most semantic shortlists choose exactly the feature Top-4. Reuse
            # that solve and pay for a second four-way solve only when geometry
            # actually changes the physical support set.
            changed_set = (selected_ids != feature_selected_ids).any(dim=-1)
            changed = changed_set.nonzero().flatten()
            joint = [value.clone() for value in feature_only]
            if changed.numel():
                changed_joint = solve(
                    selected_ids.index_select(0, changed),
                    source.index_select(0, changed),
                    source_geometry.index_select(0, changed),
                )
                for destination, update in zip(joint, changed_joint):
                    destination.index_copy_(0, changed, update)
            tolerance = float(geometry_guard_tolerance)
            choose_4d = (
                (joint[1] + tolerance >= feature_only[1])
                & (joint[2] <= feature_only[2] + tolerance)
                & (joint[4] < feature_only[4])
                & changed_set
            )
            selected_ids = torch.where(
                choose_4d[:, None], selected_ids, feature_selected_ids,
            )
            alpha = torch.where(choose_4d[:, None], joint[0], feature_only[0])
            cosine = torch.where(choose_4d, joint[1], feature_only[1])
            residual = torch.where(choose_4d, joint[2], feature_only[2])
            reconstruction_norm = torch.where(choose_4d, joint[3], feature_only[3])
            guard_counts.append(choose_4d.sum())
            guard_feature_deltas.append((joint[1] - feature_only[1]).sum())
            guard_geometry_gains.append((feature_only[4] - joint[4]).sum())
            all_ids.append(selected_ids)
            all_weights.append(alpha)
            all_cosine.append(cosine)
            all_residual.append(residual)
            all_norm.append(reconstruction_norm)
            easy_counts.append(torch.zeros((), device=source.device, dtype=torch.long))
            _profile_end("fused.convex_and_certificate", profile)
            continue

        # The margin makes the BF16 Top-1 decision safely feasible at the FP32
        # acceptance threshold. Easy rows need neither a support gather nor a
        # four-variable solve.
        easy = value[:, 0] >= easy_threshold if easy_enabled else torch.zeros_like(
            value[:, 0], dtype=torch.bool,
        )
        hard = (~easy).nonzero().flatten()
        alpha = torch.zeros((len(source), 4), device=source.device, dtype=torch.float32)
        alpha[easy, 0] = 1.0
        cosine = value[:, 0].float()
        residual = (2.0 - 2.0 * cosine).clamp_min(0.0).sqrt()
        reconstruction_norm = torch.ones_like(cosine)
        if hard.numel():
            hard_source = source.index_select(0, hard)
            hard_selected = selected_ids.index_select(0, hard)
            hard_supports = features.index_select(0, hard_selected.reshape(-1)).reshape(
                len(hard), 4, -1,
            ).float()
            gram = hard_supports @ hard_supports.transpose(1, 2)
            rhs = (hard_supports @ hard_source.unsqueeze(-1)).squeeze(-1)
            hard_alpha = solve_weights(gram, rhs)
            reconstruction = (hard_supports * hard_alpha.unsqueeze(-1)).sum(dim=1)
            alpha.index_copy_(0, hard, hard_alpha)
            cosine.index_copy_(
                0, hard, F.cosine_similarity(hard_source, reconstruction, dim=-1, eps=1e-8),
            )
            residual.index_copy_(0, hard, (hard_source - reconstruction).norm(dim=-1))
            reconstruction_norm.index_copy_(0, hard, reconstruction.norm(dim=-1))
        all_ids.append(selected_ids)
        all_weights.append(alpha)
        all_cosine.append(cosine)
        all_residual.append(residual)
        all_norm.append(reconstruction_norm)
        easy_counts.append(easy.sum())
        _profile_end("fused.convex_and_certificate", profile)
    easy_total = int(torch.stack(easy_counts).sum()) if easy_counts else 0
    guard_selected = int(torch.stack(guard_counts).sum()) if guard_counts else 0
    guard_candidates = len(source_ids) if guard_counts else 0
    return {"support_ids": torch.cat(all_ids), "weights": torch.cat(all_weights),
            "cosine": torch.cat(all_cosine), "residual": torch.cat(all_residual),
            "reconstruction_norm": torch.cat(all_norm),
            "easy_enabled": easy_enabled,
            "easy_threshold": easy_threshold,
            "easy_candidates": easy_total,
            "hard_candidates": len(source_ids) - easy_total,
            "streaming_top4_kernel": triton_top4,
            "active_set_kernel": triton_active_set,
            "geometry_guard_enabled": bool(guard_counts),
            "geometry_guard_selected": guard_selected,
            "geometry_guard_fallback": guard_candidates - guard_selected,
            "geometry_guard_selection_ratio": (
                guard_selected / guard_candidates if guard_candidates else 0.0
            ),
            "geometry_guard_feature_cosine_delta_mean": (
                float(torch.stack(guard_feature_deltas).sum()) / guard_candidates
                if guard_candidates else 0.0
            ),
            "geometry_guard_4d_residual_gain_mean": (
                float(torch.stack(guard_geometry_gains).sum()) / guard_candidates
                if guard_candidates else 0.0
            )}


@torch.no_grad()
def _adaptive_fit_temporal_local(
    features: torch.Tensor, basis_ids: torch.Tensor, *, frames: int,
    patches: int, threshold: float, source_ids: torch.Tensor,
    geometry: torch.Tensor | None = None, geometry_weight: float = 0.0,
    geometry_guard_tolerance: float | None = None,
    frame_group: int = _LOCAL_FRAME_GROUP,
) -> dict[str, torch.Tensor]:
    """Batched local Top-4 retrieval, one convex solve, and global fallback."""
    if source_ids.ndim != 1 or basis_ids.ndim != 1:
        raise ValueError("temporal-local fitting expects flat source and basis ids")
    count = len(source_ids)
    if not count:
        empty_ids = source_ids.reshape(0, 1).expand(0, 4)
        scalar = features.new_empty(0, dtype=torch.float32)
        return {
            "support_ids": empty_ids,
            "weights": scalar.reshape(0, 1).expand(0, 4),
            "cosine": scalar, "residual": scalar,
            "reconstruction_norm": scalar, "easy_candidates": 0,
            "hard_candidates": 0, "streaming_top4_kernel": False,
            "active_set_kernel": False, "geometry_guard_enabled": False,
            "geometry_guard_selected": 0, "local_search": True,
            "local_candidate_mean": 0.0, "global_fallback_candidates": 0,
        }

    support_ids = basis_ids[0].expand(count, 4).clone()
    feature_support_ids = support_ids.clone()
    weights = torch.empty((count, 4), device=features.device, dtype=torch.float32)
    cosine = torch.empty(count, device=features.device, dtype=torch.float32)
    residual = torch.empty_like(cosine)
    reconstruction_norm = torch.empty_like(cosine)
    assigned = torch.zeros(count, device=features.device, dtype=torch.bool)
    source_frame = torch.div(source_ids, patches, rounding_mode="floor")
    basis_frame = torch.div(basis_ids, patches, rounding_mode="floor")
    local_candidate_total = 0
    local_source_total = 0
    active_set_kernel = (
        features.is_cuda and triton_acceleration_available()
        and os.environ.get("VGGT_TRITON_ACTIVE_SET", "0") != "0"
    )
    streaming_top4_kernel = False
    guard_enabled = (
        geometry_weight > 0 and geometry is not None
        and geometry_guard_tolerance is not None
    )
    geometry_guard_selected = 0

    for begin in range(1, frames, frame_group):
        end = min(begin + frame_group, frames)
        rows = ((source_frame >= begin) & (source_frame < end)).nonzero().flatten()
        if not rows.numel():
            continue
        # Match the released fixed-topology temporal neighbourhood: the full
        # reference frame plus one preceding and one following boundary frame.
        first_local_frame = max(1, begin - 1)
        last_local_frame = min(frames - 1, end)
        local_mask = ((basis_frame == 0)
                      | ((basis_frame >= first_local_frame)
                         & (basis_frame <= last_local_frame)))
        local_basis = basis_ids.index_select(0, local_mask.nonzero().flatten())
        profile = _profile_begin()
        query_ids = source_ids.index_select(0, rows)
        source = features.index_select(0, query_ids).float()
        local_feature = features.index_select(0, local_basis)
        if geometry_weight and not guard_enabled:
            source_route, basis_route = _joint_metric_routes(
                source, geometry.index_select(0, query_ids),
                local_feature, geometry.index_select(0, local_basis),
                geometry_weight,
            )
            score = source_route.to(torch.bfloat16) @ basis_route.to(torch.bfloat16).T
            local_index = score.topk(4, dim=-1).indices
            selected = local_basis.index_select(
                0, local_index.reshape(-1),
            ).reshape(len(rows), 4)
            support_ids.index_copy_(0, rows, selected)
            feature_support_ids.index_copy_(0, rows, selected)
        elif guard_enabled:
            score = source.to(torch.bfloat16) @ local_feature.to(torch.bfloat16).T
            shortlist = min(16, len(local_basis))
            feature_value, feature_local = score.topk(shortlist, dim=-1)
            feature_top4 = score.topk(4, dim=-1).indices
            feature_selected = local_basis.index_select(
                0, feature_top4.reshape(-1),
            ).reshape(len(rows), 4)
            source_geometry = geometry.index_select(0, query_ids)
            candidate_geometry = geometry.index_select(
                0, local_basis.index_select(0, feature_local.reshape(-1)),
            ).reshape(len(rows), shortlist, 4)
            joint_value = feature_value.float() - geometry_weight * (
                source_geometry[:, None].float() - candidate_geometry.float()
            ).square().sum(dim=-1)
            joint_in_shortlist = joint_value.topk(4, dim=-1).indices
            selected_feature_value = feature_value.gather(1, joint_in_shortlist)
            canonical_order = selected_feature_value.argsort(
                dim=-1, descending=True, stable=True,
            )
            joint_in_shortlist = joint_in_shortlist.gather(1, canonical_order)
            joint_local = feature_local.gather(1, joint_in_shortlist)
            joint_selected = local_basis.index_select(
                0, joint_local.reshape(-1),
            ).reshape(len(rows), 4)
            feature_support_ids.index_copy_(0, rows, feature_selected)
            support_ids.index_copy_(0, rows, joint_selected)
        else:
            score = source.to(torch.bfloat16) @ local_feature.to(torch.bfloat16).T
            local_index = score.topk(4, dim=-1).indices
            selected = local_basis.index_select(
                0, local_index.reshape(-1),
            ).reshape(len(rows), 4)
            support_ids.index_copy_(0, rows, selected)
            feature_support_ids.index_copy_(0, rows, selected)
        _profile_end("local.support_search", profile)
        assigned[rows] = True
        local_candidate_total += int(len(local_basis)) * int(len(rows))
        local_source_total += int(len(rows))

    def solve(
        candidate_ids: torch.Tensor, solve_source: torch.Tensor,
        solve_geometry: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        rows = len(candidate_ids)
        supports = features.index_select(0, candidate_ids.reshape(-1)).reshape(
            rows, 4, -1,
        ).float()
        gram = supports @ supports.transpose(1, 2)
        rhs = (supports @ solve_source.unsqueeze(-1)).squeeze(-1)
        if active_set_kernel:
            alpha = four_support_active_set(gram, rhs, ridge=_RIDGE)
        else:
            alpha = four_support_projected_gradient(gram, rhs, steps=32)
        reconstruction = (supports * alpha.unsqueeze(-1)).sum(dim=1)
        result: tuple[torch.Tensor, ...] = (
            alpha,
            F.cosine_similarity(solve_source, reconstruction, dim=-1, eps=1e-8),
            (solve_source - reconstruction).norm(dim=-1),
            reconstruction.norm(dim=-1),
        )
        if solve_geometry is not None:
            support_geometry = geometry.index_select(
                0, candidate_ids.reshape(-1),
            ).reshape(rows, 4, 4).float()
            geometry_reconstruction = (
                support_geometry * alpha.unsqueeze(-1)
            ).sum(dim=1)
            result += ((solve_geometry.float() - geometry_reconstruction).norm(dim=-1),)
        return result

    assigned_rows = assigned.nonzero().flatten()
    solve_chunk = max(256, int(os.environ.get("VGGT_TTT_FIT_CHUNK", "4096")))
    for chunk_begin in range(0, len(assigned_rows), solve_chunk):
        profile = _profile_begin()
        rows = assigned_rows[chunk_begin:chunk_begin + solve_chunk]
        solve_source = features.index_select(
            0, source_ids.index_select(0, rows),
        ).float()
        solve_geometry = (
            geometry.index_select(0, source_ids.index_select(0, rows))
            if guard_enabled else None
        )
        feature_ids = feature_support_ids.index_select(0, rows)
        feature_fit = solve(feature_ids, solve_source, solve_geometry)
        chosen_ids = support_ids.index_select(0, rows)
        chosen_fit = [value.clone() for value in feature_fit]
        choose_4d = torch.zeros(len(rows), device=features.device, dtype=torch.bool)
        if guard_enabled:
            changed = (chosen_ids != feature_ids).any(dim=-1).nonzero().flatten()
            if changed.numel():
                changed_fit = solve(
                    chosen_ids.index_select(0, changed),
                    solve_source.index_select(0, changed),
                    solve_geometry.index_select(0, changed),
                )
                for destination, update in zip(chosen_fit, changed_fit):
                    destination.index_copy_(0, changed, update)
            tolerance = float(geometry_guard_tolerance)
            choose_4d = (
                (chosen_fit[1] + tolerance >= feature_fit[1])
                & (chosen_fit[2] <= feature_fit[2] + tolerance)
                & (chosen_fit[4] < feature_fit[4])
                & (chosen_ids != feature_ids).any(dim=-1)
            )
            chosen_ids = torch.where(choose_4d[:, None], chosen_ids, feature_ids)
        else:
            chosen_fit = list(feature_fit)
        support_ids.index_copy_(0, rows, chosen_ids)
        weights.index_copy_(
            0, rows, torch.where(choose_4d[:, None], chosen_fit[0], feature_fit[0])
            if guard_enabled else chosen_fit[0],
        )
        cosine.index_copy_(
            0, rows, torch.where(choose_4d, chosen_fit[1], feature_fit[1])
            if guard_enabled else chosen_fit[1],
        )
        residual.index_copy_(
            0, rows, torch.where(choose_4d, chosen_fit[2], feature_fit[2])
            if guard_enabled else chosen_fit[2],
        )
        reconstruction_norm.index_copy_(
            0, rows, torch.where(choose_4d, chosen_fit[3], feature_fit[3])
            if guard_enabled else chosen_fit[3],
        )
        geometry_guard_selected += int(choose_4d.sum())
        _profile_end("local.convex_and_certificate", profile)

    # An unexpected source in the reference frame remains unassigned and is
    # deliberately sent through the exact global route below.
    residual_limit = (2.0 * (1.0 - threshold)) ** .5
    certified = (
        assigned & (cosine >= threshold) & (residual <= residual_limit)
        & (reconstruction_norm >= .80)
    )
    fallback_rows = (~certified).nonzero().flatten()
    if fallback_rows.numel():
        global_fit = _adaptive_fit_fused(
            features, basis_ids, max_support=4, threshold=threshold,
            source_ids=source_ids.index_select(0, fallback_rows), geometry=geometry,
            geometry_weight=geometry_weight,
            geometry_guard_tolerance=geometry_guard_tolerance,
            chunk=max(256, int(os.environ.get("VGGT_TTT_FIT_CHUNK", "4096"))),
        )
        support_ids.index_copy_(0, fallback_rows, global_fit["support_ids"])
        weights.index_copy_(0, fallback_rows, global_fit["weights"])
        cosine.index_copy_(0, fallback_rows, global_fit["cosine"])
        residual.index_copy_(0, fallback_rows, global_fit["residual"])
        reconstruction_norm.index_copy_(
            0, fallback_rows, global_fit["reconstruction_norm"],
        )
        geometry_guard_selected += int(global_fit.get("geometry_guard_selected", 0))

    return {
        "support_ids": support_ids, "weights": weights, "cosine": cosine,
        "residual": residual, "reconstruction_norm": reconstruction_norm,
        "easy_candidates": 0, "hard_candidates": count,
        "streaming_top4_kernel": streaming_top4_kernel,
        "active_set_kernel": active_set_kernel,
        "geometry_guard_enabled": guard_enabled,
        "geometry_guard_selected": geometry_guard_selected,
        "local_search": True,
        "local_candidate_mean": (
            local_candidate_total / local_source_total if local_source_total else 0.0
        ),
        "global_fallback_candidates": int(fallback_rows.numel()),
    }


@torch.no_grad()
def _fast_hierarchical_convex_fit(
    features: torch.Tensor, projected: torch.Tensor, basis_ids: torch.Tensor, *,
    max_support: int, threshold: float, candidate_count: int = 64,
    chunk: int = 256, steps: int = 32,
) -> dict[str, torch.Tensor]:
    """Projected retrieval plus certified 1/2/9-support convex fitting."""
    basis_projected = projected.index_select(0, basis_ids)
    candidate_count = min(candidate_count, len(basis_ids))
    all_ids, all_weights, all_cosine, all_residual, all_norm = [], [], [], [], []
    residual_limit = (2.0 * (1.0 - threshold)) ** .5
    for begin in range(0, len(features), chunk):
        source = features[begin:begin + chunk].float()
        low_score = (projected[begin:begin + chunk].to(torch.bfloat16)
                     @ basis_projected.to(torch.bfloat16).T)
        candidate_local = low_score.topk(candidate_count, dim=-1).indices
        candidate_ids = basis_ids.index_select(0, candidate_local.reshape(-1)).reshape(candidate_local.shape)
        candidate = features.index_select(0, candidate_ids.reshape(-1)).reshape(
            len(source), candidate_count, -1,
        )
        exact_score = torch.einsum("nd,nkd->nk", source.to(torch.bfloat16), candidate.to(torch.bfloat16))
        local = exact_score.topk(min(max_support, candidate_count), dim=-1).indices
        selected_ids = candidate_ids.gather(1, local)
        supports = features.index_select(0, selected_ids.reshape(-1)).reshape(
            len(source), local.shape[1], -1,
        ).float()
        k = supports.shape[1]
        alpha = torch.zeros((len(source), k), device=source.device, dtype=torch.float32)
        alpha[:, 0] = 1.0
        reconstruction = supports[:, 0].clone()
        cosine = F.cosine_similarity(source, reconstruction, dim=-1, eps=1e-8)
        residual = (source - reconstruction).norm(dim=-1)
        norm = reconstruction.norm(dim=-1)
        passed = (cosine >= threshold) & (residual <= residual_limit) & (norm >= .80)

        # The exact closest point on the segment between the top two atoms.
        two = (~passed).nonzero().flatten()
        if len(two) and k >= 2:
            first, second = supports[two, 0], supports[two, 1]
            direction = first - second
            amount = (((source[two] - second) * direction).sum(-1)
                      / direction.square().sum(-1).clamp_min(1e-8)).clamp(0, 1)
            alpha[two] = 0
            alpha[two, 0], alpha[two, 1] = amount, 1.0 - amount
            reconstruction[two] = (first * amount[:, None] + second * (1.0 - amount)[:, None])
            cosine[two] = F.cosine_similarity(source[two], reconstruction[two], dim=-1, eps=1e-8)
            residual[two] = (source[two] - reconstruction[two]).norm(dim=-1)
            norm[two] = reconstruction[two].norm(dim=-1)
            passed[two] = ((cosine[two] >= threshold) & (residual[two] <= residual_limit)
                           & (norm[two] >= .80))

        # Only genuinely hard rows pay for the full projected-gradient solve.
        hard = (~passed).nonzero().flatten()
        if len(hard):
            hard_source, hard_support = source[hard], supports[hard]
            hard_alpha = alpha[hard]
            gram = hard_support @ hard_support.transpose(1, 2)
            rhs = (hard_support @ hard_source.unsqueeze(-1)).squeeze(-1)
            step = (.95 / gram.abs().sum(-1).amax(-1).clamp_min(1e-4)).unsqueeze(-1)
            for _ in range(steps):
                gradient = (gram @ hard_alpha.unsqueeze(-1)).squeeze(-1) - rhs
                hard_alpha = _simplex_project(hard_alpha - step * gradient)
            alpha[hard] = hard_alpha
            reconstruction[hard] = (hard_support * hard_alpha.unsqueeze(-1)).sum(1)
            cosine[hard] = F.cosine_similarity(hard_source, reconstruction[hard], dim=-1, eps=1e-8)
            residual[hard] = (hard_source - reconstruction[hard]).norm(dim=-1)
            norm[hard] = reconstruction[hard].norm(dim=-1)
        all_ids.append(F.pad(selected_ids, (0, max_support - k), value=-1))
        all_weights.append(F.pad(alpha, (0, max_support - k)))
        all_cosine.append(cosine); all_residual.append(residual); all_norm.append(norm)
    return {"support_ids": torch.cat(all_ids), "weights": torch.cat(all_weights),
            "cosine": torch.cat(all_cosine), "residual": torch.cat(all_residual),
            "reconstruction_norm": torch.cat(all_norm)}


def _fixed_hash_projection(features: torch.Tensor, dimensions: int = 32) -> torch.Tensor:
    key = (str(features.device), features.shape[-1], dimensions)
    projection = _HASH_PROJECTIONS.get(key)
    if projection is None:
        generator = torch.Generator(device=features.device).manual_seed(20261028)
        projection = torch.randn(features.shape[-1], dimensions, generator=generator,
                                 device=features.device, dtype=torch.float32)
        projection = F.normalize(projection, dim=0, eps=1e-8).to(torch.bfloat16)
        _HASH_PROJECTIONS[key] = projection
    return projection


@torch.no_grad()
def _hash_four_convex_fit(
    features: torch.Tensor, route: torch.Tensor, codes: torch.Tensor,
    representative: torch.Tensor, *, hash_bits: int, steps: int = 32,
    chunk: int = 512,
) -> dict[str, torch.Tensor]:
    """Multi-probe hash candidates followed by exact-space four-way convex fit."""
    all_ids, all_weights, all_cosine, all_residual, all_norm = [], [], [], [], []
    bit_order = route[:, :hash_bits].abs().argsort(dim=-1)
    flips = (1 << bit_order).to(codes.dtype)
    candidate_codes = torch.cat((codes[:, None], codes[:, None].bitwise_xor(flips)), dim=1)
    candidate_ids = representative.index_select(0, candidate_codes.reshape(-1)).reshape_as(candidate_codes)
    own = representative.index_select(0, codes)
    candidate_ids = torch.where(candidate_ids >= 0, candidate_ids, own[:, None])
    for begin in range(0, len(features), chunk):
        source = features[begin:begin + chunk].float()
        candidates = candidate_ids[begin:begin + chunk]
        candidate_feature = features.index_select(0, candidates.reshape(-1)).reshape(
            len(source), candidates.shape[1], -1,
        )
        exact = torch.einsum("nd,nkd->nk", source.to(torch.bfloat16),
                             candidate_feature.to(torch.bfloat16))
        local = exact.topk(4, dim=-1).indices
        selected_ids = candidates.gather(1, local)
        supports = features.index_select(0, selected_ids.reshape(-1)).reshape(
            len(source), 4, -1,
        ).float()
        alpha = torch.zeros((len(source), 4), device=source.device, dtype=torch.float32)
        alpha[:, 0] = 1.0
        gram = supports @ supports.transpose(1, 2)
        rhs = (supports @ source.unsqueeze(-1)).squeeze(-1)
        step = (.95 / gram.abs().sum(-1).amax(-1).clamp_min(1e-4)).unsqueeze(-1)
        for _ in range(steps):
            gradient = (gram @ alpha.unsqueeze(-1)).squeeze(-1) - rhs
            alpha = _simplex_project(alpha - step * gradient)
        reconstruction = (supports * alpha.unsqueeze(-1)).sum(1)
        all_ids.append(selected_ids); all_weights.append(alpha)
        all_cosine.append(F.cosine_similarity(source, reconstruction, dim=-1, eps=1e-8))
        all_residual.append((source - reconstruction).norm(dim=-1))
        all_norm.append(reconstruction.norm(dim=-1))
    return {"support_ids": torch.cat(all_ids), "weights": torch.cat(all_weights),
            "cosine": torch.cat(all_cosine), "residual": torch.cat(all_residual),
            "reconstruction_norm": torch.cat(all_norm)}


@torch.no_grad()
def build_adaptive_basis_plan(
    tokens: torch.Tensor, *, frames: int, tokens_per_frame: int, patch_token_start: int,
    patch_hw: tuple[int, int], threshold: float, mode: str, max_support: int = 9,
    coefficient_limit: float = 1.0, promotion_batch: int = 512, max_rounds: int = 128,
    preserve_queries: bool = False, preserve_destination_keys: bool | None = None,
) -> WeightedBilinearTokenPlan:
    """Residual-greedy physical basis with convex or signed-box restoration."""
    if tokens.shape[0] != 1 or mode not in {"adaptive_convex", "adaptive_linear"}:
        raise ValueError("adaptive basis plan requires batch one and a valid adaptive mode")
    height, width = patch_hw
    patches = height * width
    total = tokens.shape[1]
    if total != frames * tokens_per_frame or patch_token_start + patches != tokens_per_frame:
        raise ValueError("invalid adaptive basis token layout")
    device = tokens.device
    patch_ids = (torch.arange(frames, device=device)[:, None] * tokens_per_frame
                 + patch_token_start + torch.arange(patches, device=device)[None]).flatten()
    features = F.normalize(tokens[0].index_select(0, patch_ids).float(), dim=-1, eps=1e-8).to(torch.bfloat16)
    mean = F.normalize(features.float().mean(dim=0), dim=0, eps=1e-8)
    basis_mask = torch.zeros(len(features), device=device, dtype=torch.bool)
    basis_mask[int((features.float() @ mean).argmax())] = True
    residual_limit = (2.0 * (1.0 - threshold)) ** .5
    fit = None
    rounds = 0
    for rounds in range(max_rounds):
        basis_local = basis_mask.nonzero().flatten()
        fit = _adaptive_fit(features, basis_local, max_support=max_support, mode=mode,
                            coefficient_limit=coefficient_limit)
        accepted = ((fit["cosine"] >= threshold) & (fit["residual"] <= residual_limit)
                    & (fit["reconstruction_norm"] >= .80))
        accepted[basis_mask] = True
        failures = (~accepted).nonzero().flatten()
        if not len(failures):
            break
        amount = min(promotion_batch, len(failures))
        error = fit["residual"].index_select(0, failures)
        basis_mask[failures.index_select(0, error.topk(amount).indices)] = True
    else:
        raise RuntimeError("adaptive basis search did not converge")
    assert fit is not None
    basis_local = basis_mask.nonzero().flatten()
    source_local = (~basis_mask).nonzero().flatten()
    basis_original = patch_ids.index_select(0, basis_local)
    source_original = patch_ids.index_select(0, source_local)
    support_local = fit["support_ids"].index_select(0, source_local)
    support_original = patch_ids.index_select(0, support_local.reshape(-1)).reshape_as(support_local)
    weights = fit["weights"].index_select(0, source_local).float()
    special = (torch.arange(frames, device=device)[:, None] * tokens_per_frame
               + torch.arange(patch_token_start, device=device)[None]).flatten()
    keep = torch.cat((special, basis_original)).sort().values.view(1, -1)
    identity = torch.arange(total, device=device).view(1, -1)
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=device).view(1, -1))
    parent_compact = original_to_compact.gather(1, support_original.reshape(1, -1)).reshape_as(support_original)
    if (parent_compact < 0).any():
        raise RuntimeError("adaptive support must remain physical")
    mass = torch.ones((1, keep.shape[1], 1), device=device, dtype=torch.float32)
    for support_index in range(max_support):
        mass.scatter_add_(1, parent_compact[None, :, support_index, None],
                          weights[None, :, support_index, None])
    if (mass <= 0).any():
        raise RuntimeError("signed adaptive pooling produced non-positive evidence mass")
    special_compact = original_to_compact[0, special]
    selected_cosine = fit["cosine"].index_select(0, source_local)
    return WeightedBilinearTokenPlan(
        keep=keep, source_original=source_original.view(1, -1),
        source_parent_compact=parent_compact.view(1, len(source_local), max_support),
        source_pool_weights=weights.view(1, len(source_local), max_support),
        source_restore_weights=weights.view(1, len(source_local), max_support),
        original_tokens=total, normalizer=mass, preserve_queries=preserve_queries,
        preserve_destination_keys=(preserve_queries if preserve_destination_keys is None
                                   else preserve_destination_keys),
        dense_special_queries=True, special_query_original=special,
        special_query_compact=special_compact, key_log_mass=mass[..., 0].log(),
        metadata={"planner": mode + "_basis_v1", "threshold": float(threshold),
                  "max_support": int(max_support), "coefficient_limit": float(coefficient_limit),
                  "merge_ratio": float(len(source_local) / total),
                  "patch_basis_ratio": float(len(basis_local) / len(features)),
                  "fit_mean": float(selected_cosine.mean()) if len(source_local) else 0.0,
                  "search_rounds": int(rounds + 1), "preserve_queries": bool(preserve_queries)},
    )


@torch.no_grad()
def build_kmeans_basis_plan(
    tokens: torch.Tensor, *, frames: int, tokens_per_frame: int, patch_token_start: int,
    patch_hw: tuple[int, int], threshold: float, basis_ratio: float,
    max_support: int = 9, kmeans_iterations: int = 5, projection_dim: int = 64,
    preserve_queries: bool = False, preserve_destination_keys: bool | None = None,
    fast_solver: bool = False, fused_solver: bool = False, fused_batching: bool = False,
    geometry: torch.Tensor | None = None, selection_geometry_weight: float = 0.0,
    support_geometry_weight: float = 0.0,
    support_geometry_guard_tolerance: float | None = None,
) -> WeightedBilinearTokenPlan:
    """First-frame-preserving spherical K-means medoids with convex restoration."""
    if tokens.shape[0] != 1 or frames < 2 or not 0 < basis_ratio < 1:
        raise ValueError("K-means basis plan requires batch one, multiple frames, and ratio in (0,1)")
    if fused_batching and not fused_solver:
        raise ValueError("fused batching requires the fused four-support solver")
    if selection_geometry_weight < 0 or support_geometry_weight < 0:
        raise ValueError("4-D geometry weights must be nonnegative")
    if support_geometry_guard_tolerance is not None and support_geometry_guard_tolerance < 0:
        raise ValueError("4-D feature guard tolerance must be nonnegative")
    height, width = patch_hw
    patches = height * width
    total, device = tokens.shape[1], tokens.device
    if total != frames * tokens_per_frame or patch_token_start + patches != tokens_per_frame:
        raise ValueError("invalid K-means basis token layout")
    patch_ids = (torch.arange(frames, device=device)[:, None] * tokens_per_frame
                 + patch_token_start + torch.arange(patches, device=device)[None]).flatten()
    features = F.normalize(tokens[0].index_select(0, patch_ids).float(), dim=-1, eps=1e-8)
    if geometry is not None:
        geometry = geometry.to(device=device, dtype=torch.float32)
        if geometry.shape != (len(features), 4):
            raise ValueError(f"geometry must have shape {(len(features), 4)}, got {tuple(geometry.shape)}")
    if (selection_geometry_weight > 0 or support_geometry_weight > 0) and geometry is None:
        raise ValueError("geometry-aware K-means requires a patch-aligned 4-D descriptor")
    later = features[patches:]
    later_geometry = None if geometry is None else geometry[patches:]
    k = max(1, min(len(later), int(round(len(later) * basis_ratio))))

    # A fixed random projection makes online clustering affordable while
    # preserving the normalized-state geometry used by the restoration fit.
    profile = _profile_begin()
    generator = torch.Generator(device=device).manual_seed(0)
    projection = torch.randn(features.shape[1], min(projection_dim, features.shape[1]),
                             generator=generator, device=device, dtype=torch.float32)
    projection = F.normalize(projection, dim=0, eps=1e-8)
    projected_all = F.normalize(features @ projection, dim=-1, eps=1e-8)
    projected = projected_all[patches:]
    # The validated short-clip regime uses the original deterministic 8192-token
    # coreset. Expanding it to 2*k changed the physical medoids enough to damage
    # pose/point-cloud quality despite almost identical feature-fit scores. For
    # long clips k may exceed 8192, so raise the floor only to k: this preserves
    # distinct seeds and the requested basis ratio without the harmful 2*k
    # distribution shift. In that regime refinement is deliberately minimal.
    train_override = os.environ.get("VGGT_KMEANS_TRAIN_TOKENS")
    train_count = (
        min(len(later), int(train_override))
        if train_override is not None
        else min(len(later), max(8192, k))
    )
    if train_count < k:
        raise ValueError(
            f"K-means training tokens ({train_count}) must not be fewer than centers ({k})"
        )
    train_ids = torch.linspace(0, len(later) - 1, train_count, device=device).round().long().unique()
    train = projected.index_select(0, train_ids)
    train_geometry = None if later_geometry is None else later_geometry.index_select(0, train_ids)
    _profile_end(("fused" if fused_solver else "legacy") + ".projection", profile)
    profile = _profile_begin()
    seed_ids = torch.linspace(0, len(train) - 1, k, device=device).round().long()
    centers = train.index_select(0, seed_ids).clone()
    centers_geometry = None if train_geometry is None else train_geometry.index_select(0, seed_ids).clone()
    assignment = torch.zeros(len(train), device=device, dtype=torch.long)
    selection_guard_enabled = (
        selection_geometry_weight > 0 and support_geometry_guard_tolerance is not None
    )
    for _ in range(kmeans_iterations):
        assignment_chunk = 8192 if fused_batching else 1024
        if selection_geometry_weight and not selection_guard_enabled:
            _, center_route = _joint_metric_routes(
                centers, centers_geometry, centers, centers_geometry, selection_geometry_weight,
            )
            centers_bf16 = center_route.to(torch.bfloat16).T.contiguous()
        else:
            centers_bf16 = centers.to(torch.bfloat16).T.contiguous()
        for begin in range(0, len(train), assignment_chunk):
            train_part = train[begin:begin + assignment_chunk]
            if selection_geometry_weight and not selection_guard_enabled:
                geometry_part = train_geometry[begin:begin + assignment_chunk]
                train_route, _ = _joint_metric_routes(
                    train_part, geometry_part, centers, centers_geometry,
                    selection_geometry_weight,
                )
                score = train_route.to(torch.bfloat16) @ centers_bf16
            else:
                score = train_part.to(torch.bfloat16) @ centers_bf16
            assignment[begin:begin + assignment_chunk] = score.argmax(dim=-1)

        # Deterministic segmented means replace atomic index_add_. Besides
        # reproducibility, sorting makes each cluster contiguous for reduction.
        order = assignment.argsort(stable=True)
        counts = torch.bincount(assignment, minlength=k)
        means = torch.segment_reduce(train.index_select(0, order), reduce="mean", lengths=counts)
        nonempty = counts > 0
        centers[nonempty] = F.normalize(means[nonempty], dim=-1, eps=1e-8)
        if selection_geometry_weight:
            geometry_means = torch.segment_reduce(
                train_geometry.index_select(0, order), reduce="mean", lengths=counts,
            )
            centers_geometry[nonempty] = geometry_means[nonempty]
    _profile_end(("fused" if fused_solver else "legacy") + ".kmeans", profile)

    # Project virtual centers back to physical training tokens (medoids).
    profile = _profile_begin()
    assigned_centers = centers.index_select(0, assignment)
    feature_similarity = (train * assigned_centers).sum(dim=-1)
    similarity = feature_similarity
    if selection_geometry_weight:
        assigned_geometry = centers_geometry.index_select(0, assignment)
        similarity = similarity - selection_geometry_weight * (
            train_geometry - assigned_geometry
        ).square().sum(dim=-1)
        if selection_guard_enabled:
            # Preserve the feature-defined clusters and admit 4-D medoids only
            # from the near-optimal feature shell of each cluster. Geometry
            # therefore resolves semantic ambiguity instead of redefining it.
            ordered_feature_similarity = feature_similarity.index_select(0, order)
            best_feature_similarity = torch.segment_reduce(
                ordered_feature_similarity, reduce="max", lengths=counts, initial=-torch.inf,
            )
            feature_eligible = feature_similarity >= (
                best_feature_similarity.index_select(0, assignment)
                - float(support_geometry_guard_tolerance)
            )
            similarity = torch.where(
                feature_eligible, similarity, torch.full_like(similarity, -torch.inf),
            )
    if fused_solver:
        # Reuse the final stable cluster order. Segmented max/min avoids CUDA
        # atomic scatter reductions and fixes the earliest-token tie break.
        ordered_similarity = similarity.index_select(0, order)
        best_similarity = torch.segment_reduce(
            ordered_similarity, reduce="max", lengths=counts, initial=-torch.inf,
        )
        train_position = torch.arange(len(train), device=device)
        candidate = torch.where(
            similarity == best_similarity.index_select(0, assignment),
            train_position,
            len(train),
        )
        medoid_train = torch.segment_reduce(
            candidate.index_select(0, order).float(),
            reduce="min",
            lengths=counts,
            initial=float(len(train)),
        ).long()
    else:
        assignment_cpu, similarity_cpu = assignment.cpu(), similarity.cpu()
        medoid_train_list = []
        for cluster in range(k):
            members = (assignment_cpu == cluster).nonzero().flatten()
            medoid_train_list.append(int(members[similarity_cpu.index_select(0, members).argmax()])
                                     if len(members) else len(train))
        medoid_train = torch.tensor(medoid_train_list, device=device)

    # Empty clusters need a distinct physical representative. A naive seed
    # fallback can duplicate a medoid already selected by a non-empty cluster,
    # silently shrinking the basis after unique(). Prefer each empty cluster's
    # own seed when unused, then fill conflicts from the remaining train set.
    empty_cluster = medoid_train == len(train)
    if empty_cluster.any():
        used = torch.zeros(len(train), device=device, dtype=torch.bool)
        used[medoid_train[~empty_cluster]] = True
        empty_ids = empty_cluster.nonzero().flatten()
        desired = seed_ids.index_select(0, empty_ids)
        free_desired = ~used.index_select(0, desired)
        medoid_train[empty_ids[free_desired]] = desired[free_desired]
        used[desired[free_desired]] = True
        conflicts = empty_ids[~free_desired]
        if conflicts.numel():
            available = (~used).nonzero().flatten()
            medoid_train[conflicts] = available[:len(conflicts)]
    medoid_later = train_ids.index_select(0, medoid_train)
    medoid_checksum = (
        ((medoid_later.long() + 1) * (torch.arange(k, device=device, dtype=torch.long) + 1)).sum()
        % 2_147_483_647
    )
    basis_local = torch.cat((torch.arange(patches, device=device), patches + medoid_later)).unique()
    if len(basis_local) != patches + k:
        raise RuntimeError(
            f"K-means basis collapsed: requested {patches + k} distinct tokens, got {len(basis_local)}"
        )
    _profile_end(("fused" if fused_solver else "legacy") + ".medoid", profile)
    profile = _profile_begin()
    candidate_mask = torch.ones(len(features), device=device, dtype=torch.bool)
    candidate_mask[:patches] = False
    candidate_mask[basis_local] = False
    candidate_local = candidate_mask.nonzero().flatten()
    fit = (_adaptive_fit_fused(
        features, basis_local, max_support=max_support, threshold=threshold,
        source_ids=candidate_local, chunk=8192 if fused_batching else 1024,
        geometry=geometry, geometry_weight=support_geometry_weight,
        geometry_guard_tolerance=support_geometry_guard_tolerance,
    ) if fused_solver else _fast_hierarchical_convex_fit(
        features, projected_all, basis_local, max_support=max_support, threshold=threshold,
    ) if fast_solver else _adaptive_fit(
        features, basis_local, max_support=max_support, mode="adaptive_convex", coefficient_limit=1.0,
    ))
    _profile_end(("fused" if fused_solver else "legacy") + ".fit_total", profile)
    residual_limit = (2.0 * (1.0 - threshold)) ** .5
    accepted = ((fit["cosine"] >= threshold) & (fit["residual"] <= residual_limit)
                & (fit["reconstruction_norm"] >= .80))
    if fused_solver:
        source_local = candidate_local.index_select(0, accepted.nonzero().flatten())
        kept_mask = torch.ones(len(features), device=device, dtype=torch.bool)
        kept_mask[source_local] = False
        keep_local = kept_mask.nonzero().flatten()
    else:
        accepted[:patches] = False
        accepted[basis_local] = False
        source_local = accepted.nonzero().flatten()
        keep_local = (~accepted).nonzero().flatten()
    basis_original = patch_ids.index_select(0, keep_local)
    source_original = patch_ids.index_select(0, source_local)
    selected_fit = accepted.nonzero().flatten() if fused_solver else source_local
    support_local = fit["support_ids"].index_select(0, selected_fit)
    support_original = patch_ids.index_select(0, support_local.reshape(-1)).reshape_as(support_local)
    weights = fit["weights"].index_select(0, selected_fit).float()
    special = (torch.arange(frames, device=device)[:, None] * tokens_per_frame
               + torch.arange(patch_token_start, device=device)[None]).flatten()
    keep = torch.cat((special, basis_original)).sort().values.view(1, -1)
    identity = torch.arange(total, device=device).view(1, -1)
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=device).view(1, -1))
    parent_compact = original_to_compact.gather(1, support_original.reshape(1, -1)).reshape_as(support_original)
    if (parent_compact < 0).any():
        raise RuntimeError("K-means support must remain physical")
    mass = deterministic_weighted_normalizer(
        parent_compact.view(1, len(source_local), max_support),
        weights.view(1, len(source_local), max_support),
        keep.shape[1],
    )
    special_compact = original_to_compact[0, special]
    selected_cosine = fit["cosine"].index_select(0, selected_fit)
    return WeightedBilinearTokenPlan(
        keep=keep, source_original=source_original.view(1, -1),
        source_parent_compact=parent_compact.view(1, len(source_local), max_support),
        source_pool_weights=weights.view(1, len(source_local), max_support),
        source_restore_weights=weights.view(1, len(source_local), max_support),
        original_tokens=total, normalizer=mass, preserve_queries=preserve_queries,
        preserve_destination_keys=(preserve_queries if preserve_destination_keys is None
                                   else preserve_destination_keys),
        dense_special_queries=True, special_query_original=special,
        special_query_compact=special_compact, key_log_mass=mass[..., 0].log(),
        metadata={"planner": ("kmeans_convex_4d_fused_batched_basis_v1"
                              if fused_batching and (selection_geometry_weight or support_geometry_weight) else
                              "kmeans_convex_fused_batched_basis_v1" if fused_batching else
                              "kmeans_convex_fused_basis_v1" if fused_solver else
                              "kmeans_convex_fast_basis_v1" if fast_solver else
                              "kmeans_convex_basis_v1"),
                  "threshold": float(threshold),
                  "requested_basis_ratio": float(basis_ratio), "kmeans_centers": int(k),
                  "kmeans_medoid_checksum": int(medoid_checksum),
                  "kmeans_medoid_index_sum": int(medoid_later.long().sum()),
                  "max_support": int(max_support), "merge_ratio": float(len(source_local) / total),
                  "patch_keep_ratio": float(len(keep_local) / len(features)),
                  "fit_mean": float(selected_cosine.mean()) if len(source_local) else 0.0,
                  "training_tokens": int(train_count),
                  "geometry_aware": bool(selection_geometry_weight or support_geometry_weight),
                  "selection_geometry_weight": float(selection_geometry_weight),
                  "support_geometry_weight": float(support_geometry_weight),
                  "support_geometry_guard_tolerance": support_geometry_guard_tolerance,
                  "geometry_guard_enabled": bool(fit.get("geometry_guard_enabled", False)),
                  "geometry_guard_selected": int(fit.get("geometry_guard_selected", 0)),
                  "geometry_guard_fallback": int(fit.get("geometry_guard_fallback", 0)),
                  "geometry_guard_selection_ratio": float(
                      fit.get("geometry_guard_selection_ratio", 0.0)
                  ),
                  "geometry_guard_feature_cosine_delta_mean": float(
                      fit.get("geometry_guard_feature_cosine_delta_mean", 0.0)
                  ),
                  "geometry_guard_4d_residual_gain_mean": float(
                      fit.get("geometry_guard_4d_residual_gain_mean", 0.0)
                  ),
                  "easy_enabled": bool(fit.get("easy_enabled", True)),
                  "easy_threshold": float(fit.get("easy_threshold", threshold + _EARLY_MARGIN)),
                  "easy_candidates": int(fit.get("easy_candidates", 0)),
                  "hard_candidates": int(fit.get("hard_candidates", len(candidate_local))),
                  "preserve_queries": bool(preserve_queries)},
    )


def _ttt_cell_table(height: int, width: int, cell: int, device: torch.device) -> torch.Tensor:
    rows = []
    for top in range(0, height, cell):
        for left in range(0, width, cell):
            ids = [y * width + x
                   for y in range(top, min(top + cell, height))
                   for x in range(left, min(left + cell, width))]
            rows.append(ids + [-1] * (cell * cell - len(ids)))
    return torch.tensor(rows, device=device, dtype=torch.long)


@torch.no_grad()
def _ttt_local_bases(
    features: torch.Tensor, allowed: torch.Tensor, *, frames: int,
    patch_hw: tuple[int, int], cell_size: int = 4, bases_per_cell: int = 2,
) -> torch.Tensor:
    height, width = patch_hw
    patches = height * width
    table = _ttt_cell_table(height, width, cell_size, features.device)
    cells, slots = table.shape
    ids = table[None].expand(frames, -1, -1) + (
        torch.arange(frames, device=features.device)[:, None, None] * patches
    )
    spatial = table[None].expand(frames, -1, -1) >= 0
    safe = ids.clamp_min(0)
    valid = spatial & allowed.index_select(0, safe.reshape(-1)).reshape_as(safe)
    value = features.index_select(0, safe.reshape(-1)).reshape(
        frames, cells, slots, -1,
    ).float()
    mean = (value * valid[..., None]).sum(dim=2) / valid.sum(dim=2).clamp_min(1)[..., None]
    mean = F.normalize(mean, dim=-1, eps=1e-8)
    score = (value * mean[:, :, None]).sum(dim=-1).masked_fill(~valid, -torch.inf)
    first_slot = score.argmax(dim=-1)
    first = safe.gather(2, first_slot[..., None])[..., 0]
    count = valid.sum(dim=-1)
    selected = [first[count > 0]]
    if bases_per_cell == 2:
        first_feature = features.index_select(0, first.reshape(-1)).reshape(
            frames, cells, -1,
        ).float()
        similarity = (value * first_feature[:, :, None]).sum(dim=-1)
        similarity.scatter_(2, first_slot[..., None], torch.inf)
        similarity = similarity.masked_fill(~valid, torch.inf)
        second_slot = similarity.argmin(dim=-1)
        second = safe.gather(2, second_slot[..., None])[..., 0]
        selected.append(second[count > 1])
    return torch.cat(selected).unique()


@torch.no_grad()
def _ttt_residual_bases(
    features: torch.Tensor, basis_ids: torch.Tensor, amount: int,
    allowed: torch.Tensor, *, candidate_limit: int = 32768, chunk: int = 1024,
) -> torch.Tensor:
    if amount <= 0:
        return torch.empty(0, device=features.device, dtype=torch.long)
    mask = allowed.clone()
    mask[basis_ids] = False
    candidates = mask.nonzero().flatten()
    if len(candidates) > candidate_limit:
        positions = torch.linspace(
            0, len(candidates) - 1, candidate_limit, device=features.device,
        ).round().long().unique()
        candidates = candidates.index_select(0, positions)
    basis = features.index_select(0, basis_ids).to(torch.bfloat16).T.contiguous()
    residual = []
    for begin in range(0, len(candidates), chunk):
        ids = candidates[begin:begin + chunk]
        nearest = (features.index_select(0, ids).to(torch.bfloat16) @ basis).amax(dim=-1)
        residual.append(1.0 - nearest.float())
    count = min(amount, len(candidates))
    return candidates.index_select(0, torch.cat(residual).topk(count).indices)


@torch.no_grad()
def _ttt_event_bases(
    features: torch.Tensor, geometry: torch.Tensor, labels: torch.Tensor,
    sensitivity: torch.Tensor, excluded: torch.Tensor, *, frames: int,
    patch_hw: tuple[int, int], amount: int, geometry_weight: float = .01,
) -> torch.Tensor:
    if geometry_weight < 0:
        raise ValueError("TTT event geometry weight must be nonnegative")
    patches = patch_hw[0] * patch_hw[1]
    ids = ((labels == 1) & ~excluded).nonzero().flatten()
    if amount <= 0 or not len(ids):
        return ids[:0]
    geo = geometry.index_select(0, ids).float()
    xyz_bin = torch.floor((geo[:, :3] + 4.0) / .50).long().clamp(0, 31)
    time_bin = (ids // patches) // 4
    code = (((time_bin * 37 + xyz_bin[:, 0]) * 37 + xyz_bin[:, 1]) * 37 + xyz_bin[:, 2])
    _, inverse = code.unique(return_inverse=True)
    groups = int(inverse.max()) + 1
    order = inverse.argsort(stable=True)
    counts = torch.bincount(inverse, minlength=groups)
    candidate_features = features.index_select(0, ids).float()
    candidate_geometry = geometry.index_select(0, ids).float()
    center_feature = F.normalize(torch.segment_reduce(
        candidate_features.index_select(0, order), reduce="mean", lengths=counts,
    ), dim=-1, eps=1e-8)
    center_geometry = torch.segment_reduce(
        candidate_geometry.index_select(0, order), reduce="mean", lengths=counts,
    )
    score = (
        (candidate_features * center_feature.index_select(0, inverse)).sum(dim=-1)
        - geometry_weight * (
            candidate_geometry - center_geometry.index_select(0, inverse)
        ).square().sum(dim=-1)
        + .005 * sensitivity.index_select(0, ids)
    )
    best = torch.full((groups,), -torch.inf, device=features.device)
    best.scatter_reduce_(0, inverse, score, reduce="amax", include_self=True)
    sentinel = len(labels)
    candidate = torch.where(score == best.index_select(0, inverse), ids, torch.full_like(ids, sentinel))
    representative = torch.full((groups,), sentinel, device=features.device, dtype=torch.long)
    representative.scatter_reduce_(0, inverse, candidate, reduce="amin", include_self=True)
    representative = representative[representative < sentinel]
    if len(representative) > amount:
        group_score = torch.segment_reduce(
            sensitivity.index_select(0, ids).index_select(0, order),
            reduce="mean", lengths=counts,
        ) * counts.float().sqrt()
        representative = representative.index_select(0, group_score.topk(amount).indices)
    if len(representative) < amount:
        chosen = torch.zeros(len(labels), device=features.device, dtype=torch.bool)
        chosen[representative] = True
        remaining = ids[~chosen.index_select(0, ids)]
        fill = min(amount - len(representative), len(remaining))
        if fill:
            rank = sensitivity.index_select(0, remaining).topk(fill).indices
            representative = torch.cat((representative, remaining.index_select(0, rank)))
    return representative.unique()


def _uniform_2x2_basis_tables(
    frames: int, patch_hw: tuple[int, int], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the released uniform 2x2 anchors and protected 4x4 grouping.

    The random choices intentionally match :func:`build_sparse_convex_plan`:
    a fresh CPU generator with seed zero selects one physical patch from every
    2x2 cell of every non-reference frame.  The returned group table has at
    most four such anchors in each 4x4 macro-cell.  Slot zero is the protected
    spatial skeleton used by the TTT residual planner.
    """
    height, width = patch_hw
    patches = height * width
    if frames <= 1:
        empty = torch.empty((0, 0, 0), device=device, dtype=torch.long)
        return torch.arange(patches, device=device), empty, empty, empty
    generator = torch.Generator(device="cpu").manual_seed(0)
    rows, cols = torch.arange(0, height, 2), torch.arange(0, width, 2)
    choice = torch.rand(frames - 1, len(rows), len(cols), 2, generator=generator)
    row = rows[None, :, None] + (
        choice[..., 0] * (height - rows).clamp(max=2)[None, :, None]
    ).long()
    col = cols[None, None, :] + (
        choice[..., 1] * (width - cols).clamp(max=2)[None, None, :]
    ).long()
    positions = (row * width + col).reshape(frames - 1, -1)
    macro_cols = (width + 3) // 4
    macro_rows = (height + 3) // 4
    macro_count = macro_rows * macro_cols
    macro = ((positions // width) // 4) * macro_cols + ((positions % width) // 4)
    grouped = torch.full((frames - 1, macro_count, 4), -1, dtype=torch.long)
    for frame in range(frames - 1):
        counts = torch.zeros(macro_count, dtype=torch.long)
        for position, group in zip(positions[frame].tolist(), macro[frame].tolist()):
            slot = int(counts[group])
            grouped[frame, group, slot] = frame * patches + patches + position
            counts[group] += 1
    first_table = _ttt_cell_table(height, width, 4, torch.device("cpu"))
    return (
        torch.arange(patches, device=device),
        positions.to(device), grouped.to(device), first_table.to(device),
    )


@torch.no_grad()
def _protected_uniform_skeleton(
    features: torch.Tensor, *, frames: int, patch_hw: tuple[int, int],
    top1_threshold: float = .95,
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Prune redundant uniform anchors without sacrificing spatial coverage.

    Every non-reference frame retains slot zero in every non-empty 4x4
    macro-cell.  The other uniform 2x2 anchors may be removed only when they
    match a guaranteed physical target from the same macro-cell: a reference
    patch, the current protected slot, or the preceding frame's protected
    slot.  Consequently mappings are local, one-hop, deterministic, and each
    macro-cell keeps at least one per-frame anchor.
    """
    first, _, grouped, first_table = _uniform_2x2_basis_tables(
        frames, patch_hw, features.device,
    )
    if frames <= 1:
        return first, {
            "initial_uniform_bases": int(len(first)),
            "protected_uniform_bases": int(len(first)),
            "top1_pruned_uniform_bases": 0,
            "protected_macro_cells": 0,
            "top1_prune_mean_cosine": 0.0,
        }
    patches = patch_hw[0] * patch_hw[1]
    valid = grouped >= 0
    safe = grouped.clamp_min(0)
    protected = safe[..., 0]
    protected_valid = valid[..., 0]
    current_target = protected[..., None]
    previous_target = torch.cat((
        torch.full_like(protected[:1], -1), protected[:-1],
    ), dim=0)[..., None]
    reference_target = first_table[None].expand(frames - 1, -1, -1)
    target = torch.cat((reference_target, current_target, previous_target), dim=-1)
    target_valid = target >= 0
    target_safe = target.clamp_min(0)
    source_feature = features.index_select(0, safe.reshape(-1)).reshape(
        *safe.shape, -1,
    ).float()
    target_feature = features.index_select(0, target_safe.reshape(-1)).reshape(
        *target_safe.shape, -1,
    ).float()
    score = torch.einsum("fmsd,fmtd->fmst", source_feature, target_feature)
    score = score.masked_fill(~target_valid[..., None, :], -torch.inf)
    top1 = score.amax(dim=-1)
    # Slot zero is always physical.  This is the explicit 4x4 coverage guard.
    removable = valid & (torch.arange(4, device=features.device)[None, None] > 0)
    pruned = removable & (top1 >= top1_threshold)
    kept = valid & ~pruned
    later = safe[kept]
    basis = torch.cat((first, later)).unique()
    pruned_score = top1[pruned]
    initial = int(patches + valid.sum())
    return basis, {
        "initial_uniform_bases": initial,
        "protected_uniform_bases": int(len(basis)),
        "top1_pruned_uniform_bases": int(pruned.sum()),
        "protected_macro_cells": int(protected_valid.sum()),
        "top1_prune_mean_cosine": (
            float(pruned_score.mean()) if pruned_score.numel() else 0.0
        ),
    }


@torch.no_grad()
def _build_ttt_fixed_topology_plan(
    tokens: torch.Tensor, *, frames: int, tokens_per_frame: int,
    patch_token_start: int, patch_hw: tuple[int, int], threshold: float,
    preserve_queries: bool, preserve_destination_keys: bool | None,
    fixed_dynamic_mask: torch.Tensor | None = None,
) -> WeightedBilinearTokenPlan:
    """Uniform skeleton plus one fixed TTT event set and local retrieval."""
    profile_total = _profile_begin()
    profile_setup = _profile_begin()
    height, width = patch_hw
    patches = height * width
    patch_count = frames * patches
    total, device = tokens.shape[1], tokens.device
    minimum = int(os.environ.get("VGGT_TTT_FIXED_DYNAMIC_MIN_LAYERS", "2"))
    if fixed_dynamic_mask is not None:
        if fixed_dynamic_mask.numel() != patch_count:
            raise ValueError("fixed dynamic basis mask has the wrong token count")
        dynamic = fixed_dynamic_mask.to(device=device, dtype=torch.bool).nonzero().flatten()
    else:
        fixed_path = os.environ.get("VGGT_TTT_FIXED_DYNAMIC_BASIS_PATH")
        if not fixed_path:
            raise RuntimeError("fixed-topology TTT requires a fixed dynamic basis")
        frequency = _TTT_FIXED_DYNAMIC_CACHE.get(fixed_path)
        if frequency is None:
            payload = torch.load(fixed_path, map_location="cpu", weights_only=False)
            frequency = payload["dynamic"].to(torch.int16)
            _TTT_FIXED_DYNAMIC_CACHE[fixed_path] = frequency
        if len(frequency) != patch_count:
            raise ValueError("fixed dynamic basis frequency has the wrong token count")
        dynamic = (frequency.to(device) >= minimum).nonzero().flatten()

    first, _, grouped, _ = _uniform_2x2_basis_tables(
        frames, patch_hw, device,
    )
    stable_slots = int(os.environ.get("VGGT_TTT_STABLE_SLOTS_PER_4X4", "4"))
    if not 1 <= stable_slots <= 4:
        raise ValueError("VGGT_TTT_STABLE_SLOTS_PER_4X4 must be in [1,4]")
    selected_grouped = grouped[..., :stable_slots]
    stable = torch.cat((first, selected_grouped[selected_grouped >= 0])).unique()
    basis = torch.cat((stable, dynamic)).unique()
    basis_mask = torch.zeros(patch_count, device=device, dtype=torch.bool)
    basis_mask[basis] = True
    dynamic_mask = torch.zeros(patch_count, device=device, dtype=torch.bool)
    dynamic_mask[dynamic] = True
    source = (~basis_mask).nonzero().flatten()
    patch_ids = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + patch_token_start + torch.arange(patches, device=device)[None]
    ).flatten()
    features = F.normalize(
        tokens[0].index_select(0, patch_ids).float(), dim=-1, eps=1e-6,
    )
    source_frame = torch.div(source, patches, rounding_mode="floor")
    basis_frame = torch.div(basis, patches, rounding_mode="floor")
    support = torch.empty((len(source), 4), device=device, dtype=torch.long)
    top1 = torch.empty(len(source), device=device, dtype=features.dtype)
    local_candidate_total = 0
    projection_dim = int(os.environ.get("VGGT_TTT_APPROX_GLOBAL_DIM", "0"))
    hash_bits = int(os.environ.get("VGGT_TTT_HASH_GLOBAL_BITS", "0"))
    _profile_end("fixed_topology.setup", profile_setup)
    if hash_bits:
        if not 4 <= hash_bits <= 16:
            raise ValueError("VGGT_TTT_HASH_GLOBAL_BITS must be in [4,16]")
        hash_dim = max(
            hash_bits, int(os.environ.get("VGGT_TTT_HASH_GLOBAL_DIM", "64")),
        )
        bucket_capacity = int(os.environ.get("VGGT_TTT_HASH_BUCKET_CAPACITY", "64"))
        rerank_count = int(os.environ.get("VGGT_TTT_HASH_RERANK", "32"))
        projection = _fixed_hash_projection(features, dimensions=hash_dim)
        route = F.normalize(
            features.to(torch.bfloat16) @ projection, dim=-1, eps=1e-8,
        )
        bit_weight = 1 << torch.arange(hash_bits, device=device, dtype=torch.long)
        basis_code = ((route.index_select(0, basis)[:, :hash_bits] >= 0).long()
                      * bit_weight).sum(dim=-1)
        sorted_code, order = basis_code.sort(stable=True)
        sorted_basis = basis.index_select(0, order)
        bucket_count = 1 << hash_bits
        counts = torch.bincount(sorted_code, minlength=bucket_count)
        offsets = counts.cumsum(0) - counts
        rank = torch.arange(len(basis), device=device) - offsets.index_select(0, sorted_code)
        valid = rank < bucket_capacity
        table = basis[0].expand(bucket_count, bucket_capacity).clone()
        table[sorted_code[valid], rank[valid]] = sorted_basis[valid]
        source_code = ((route.index_select(0, source)[:, :hash_bits] >= 0).long()
                       * bit_weight).sum(dim=-1)
        flips = torch.cat((
            torch.zeros(1, device=device, dtype=torch.long), bit_weight,
        ))
        candidate_width = len(flips) * bucket_capacity
        coarse_keep = min(rerank_count, candidate_width)
        for begin in range(0, len(source), 1024):
            rows = torch.arange(
                begin, min(begin + 1024, len(source)), device=device,
            )
            source_ids = source.index_select(0, rows)
            probes = source_code.index_select(0, rows)[:, None].bitwise_xor(flips)
            candidates = table.index_select(0, probes.reshape(-1)).reshape(
                len(rows), candidate_width,
            )
            source_route = route.index_select(0, source_ids)
            candidate_route = route.index_select(
                0, candidates.reshape(-1),
            ).reshape(len(rows), candidate_width, hash_dim)
            coarse_score = torch.einsum(
                "nd,nkd->nk", source_route, candidate_route,
            )
            coarse_local = coarse_score.topk(coarse_keep, dim=-1).indices
            shortlist = candidates.gather(1, coarse_local)
            source_feature = features.index_select(0, source_ids)
            candidate_feature = features.index_select(
                0, shortlist.reshape(-1),
            ).reshape(len(rows), coarse_keep, -1)
            exact = torch.einsum(
                "nd,nkd->nk", source_feature.to(torch.bfloat16),
                candidate_feature.to(torch.bfloat16),
            )
            value, exact_local = exact.topk(4, dim=-1)
            selected = shortlist.gather(1, exact_local)
            support.index_copy_(0, rows, selected)
            top1.index_copy_(0, rows, value[:, 0].float())
        local_candidate_total = int(len(source)) * candidate_width
    elif projection_dim:
        profile_search = _profile_begin()
        candidate_count = min(
            int(os.environ.get("VGGT_TTT_APPROX_GLOBAL_CANDIDATES", "64")),
            len(basis),
        )
        projection = _fixed_hash_projection(features, dimensions=projection_dim)
        route = F.normalize(
            features.to(torch.bfloat16) @ projection, dim=-1, eps=1e-8,
        )
        basis_route = route.index_select(0, basis).T.contiguous()
        # 4096 amortizes the coarse-GEMM/Top-k and exact-rerank launches on
        # long sequences while retaining the exact same candidate set and
        # four-support solve as the former 2048-row implementation.
        search_chunk = int(os.environ.get("VGGT_TTT_TOPOLOGY_SEARCH_CHUNK", "4096"))
        if search_chunk < 256:
            raise ValueError("VGGT_TTT_TOPOLOGY_SEARCH_CHUNK must be at least 256")
        for begin in range(0, len(source), search_chunk):
            rows = torch.arange(
                begin, min(begin + search_chunk, len(source)), device=device,
            )
            source_ids = source.index_select(0, rows)
            coarse = (
                route.index_select(0, source_ids) @ basis_route
            ).topk(candidate_count, dim=-1).indices
            candidates = basis.index_select(0, coarse.reshape(-1)).reshape(
                len(rows), candidate_count,
            )
            source_feature = features.index_select(0, source_ids)
            candidate_feature = features.index_select(
                0, candidates.reshape(-1),
            ).reshape(len(rows), candidate_count, -1)
            exact = torch.einsum(
                "nd,nkd->nk", source_feature.to(torch.bfloat16),
                candidate_feature.to(torch.bfloat16),
            )
            value, exact_local = exact.topk(4, dim=-1)
            selected = candidates.gather(1, exact_local)
            support.index_copy_(0, rows, selected)
            top1.index_copy_(0, rows, value[:, 0].float())
        local_candidate_total = int(len(source)) * candidate_count
        _profile_end("fixed_topology.search_and_rerank", profile_search)
    else:
        for begin in range(1, frames, _LOCAL_FRAME_GROUP):
            end = min(begin + _LOCAL_FRAME_GROUP, frames)
            rows = ((source_frame >= begin) & (source_frame < end)).nonzero().flatten()
            if not rows.numel():
                continue
            local_mask = (
                (basis_frame == 0)
                | ((basis_frame >= max(1, begin - 1))
                   & (basis_frame <= min(frames - 1, end)))
            )
            if os.environ.get("VGGT_TTT_GLOBAL_DYNAMIC_CANDIDATES", "0") == "1":
                local_mask |= dynamic_mask.index_select(0, basis)
            local_basis = basis.index_select(0, local_mask.nonzero().flatten())
            value, local = (
                features.index_select(0, source.index_select(0, rows)).to(torch.bfloat16)
                @ features.index_select(0, local_basis).to(torch.bfloat16).T
            ).topk(4, dim=-1)
            support.index_copy_(
                0, rows,
                local_basis.index_select(0, local.reshape(-1)).reshape(len(rows), 4),
            )
            top1.index_copy_(0, rows, value[:, 0].float())
            local_candidate_total += int(len(rows)) * int(len(local_basis))

    profile_fit = _profile_begin()
    fit = torch.empty(len(source), device=device, dtype=torch.float32)
    weights = torch.zeros((len(source), 4), device=device, dtype=torch.float32)
    local_top1 = os.environ.get("VGGT_TTT_LOCAL_TOP1", "0") != "0"
    easy = (
        top1 >= threshold + _EARLY_MARGIN if local_top1
        else torch.zeros_like(top1, dtype=torch.bool)
    )
    weights[easy, 0] = 1.0
    fit[easy] = top1[easy]
    hard = (~easy).nonzero().flatten()
    for begin in range(0, len(hard), 16384):
        rows = hard[begin:begin + 16384]
        solve_source = features.index_select(0, source.index_select(0, rows))
        solve_support = features.index_select(
            0, support.index_select(0, rows).reshape(-1),
        ).reshape(len(rows), 4, -1)
        fit[rows], weights[rows] = _convex_fit_dispatch(solve_source, solve_support)
    _profile_end("fixed_topology.initial_convex_fit", profile_fit)

    failed = hard[fit.index_select(0, hard) < threshold]
    if failed.numel():
        profile_fallback = _profile_begin()
        dictionary = features.index_select(0, basis).to(torch.bfloat16).T.contiguous()
        for begin in range(0, len(failed), FAST_MATCH_SOURCE_CHUNK):
            rows = failed[begin:begin + FAST_MATCH_SOURCE_CHUNK]
            solve_source = features.index_select(0, source.index_select(0, rows))
            local = (solve_source.to(torch.bfloat16) @ dictionary).topk(4, dim=-1).indices
            selected = basis.index_select(0, local.reshape(-1)).reshape(len(rows), 4)
            support.index_copy_(0, rows, selected)
            solve_support = features.index_select(0, selected.reshape(-1)).reshape(
                len(rows), 4, -1,
            )
            solved_fit, solved_weights = _convex_fit_dispatch(
                solve_source, solve_support,
            )
            fit.index_copy_(0, rows, solved_fit)
            weights.index_copy_(0, rows, solved_weights)
        _profile_end("fixed_topology.global_fallback", profile_fallback)

    profile_finalize = _profile_begin()
    accepted = fit >= threshold
    accepted_rows = accepted.nonzero().flatten()
    source_local = source.index_select(0, accepted_rows)
    support_local = support.index_select(0, accepted_rows)
    selected_weights = weights.index_select(0, accepted_rows)
    kept_mask = torch.ones(patch_count, device=device, dtype=torch.bool)
    kept_mask[source_local] = False
    keep_local = kept_mask.nonzero().flatten()
    basis_original = patch_ids.index_select(0, keep_local)
    source_original = patch_ids.index_select(0, source_local)
    support_original = patch_ids.index_select(
        0, support_local.reshape(-1),
    ).reshape_as(support_local)
    special = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + torch.arange(patch_token_start, device=device)[None]
    ).flatten()
    keep = torch.cat((special, basis_original)).sort().values.view(1, -1)
    original_to_compact = torch.full((1, total), -1, device=device, dtype=torch.long)
    original_to_compact.scatter_(
        1, keep, torch.arange(keep.shape[1], device=device).view(1, -1),
    )
    parent_compact = original_to_compact.gather(
        1, support_original.reshape(1, -1),
    ).reshape_as(support_original)
    if (parent_compact < 0).any():
        raise RuntimeError("fixed-topology TTT supports must remain physical")
    mass = deterministic_weighted_normalizer(
        parent_compact.view(1, len(source_local), 4),
        selected_weights.view(1, len(source_local), 4), keep.shape[1],
    )
    special_compact = original_to_compact[0, special]
    plan = WeightedBilinearTokenPlan(
        keep=keep, source_original=source_original.view(1, -1),
        source_parent_compact=parent_compact.view(1, len(source_local), 4),
        source_pool_weights=selected_weights.view(1, len(source_local), 4),
        source_restore_weights=selected_weights.view(1, len(source_local), 4),
        original_tokens=total, normalizer=mass, preserve_queries=preserve_queries,
        preserve_destination_keys=(preserve_queries if preserve_destination_keys is None
                                   else preserve_destination_keys),
        dense_special_queries=True, special_query_original=special,
        special_query_compact=special_compact, key_log_mass=mass[..., 0].log(),
        metadata={
            "planner": "ttt_fixed_dynamic_local_topology_v1",
            "threshold": float(threshold),
            "fixed_dynamic_bases": int(len(dynamic)),
            "uniform_bases": int(len(stable)),
            "stable_slots_per_4x4": stable_slots,
            "local_top1": local_top1,
            "approx_global_dim": projection_dim,
            "hash_global_bits": hash_bits,
            "topology_search_chunk": (
                search_chunk if projection_dim and not hash_bits else None
            ),
            "fit_failure_promotions": int((~accepted).sum()),
            "local_candidate_mean": (
                local_candidate_total / len(source) if len(source) else 0.0
            ),
            "global_fallback_candidates": int(len(failed)),
            "merge_ratio": float(len(source_local) / total),
            "patch_keep_ratio": float(len(keep_local) / patch_count),
            "fit_mean": float(fit[accepted].mean()) if accepted.any() else 0.0,
            "preserve_queries": bool(preserve_queries),
        },
    )
    _profile_end("fixed_topology.finalize_plan", profile_finalize)
    _profile_end("fixed_topology.total", profile_total)
    return plan


@torch.no_grad()
def build_ttt_protected_uniform_convex_plan(
    tokens: torch.Tensor, *, frames: int, tokens_per_frame: int,
    patch_token_start: int, patch_hw: tuple[int, int], threshold: float,
    labels: torch.Tensor, sensitivity: torch.Tensor, geometry: torch.Tensor,
    max_support: int = 4, preserve_queries: bool = False,
    preserve_destination_keys: bool | None = None,
    top1_prune_threshold: float = .95, sensitive_event_ratio: float = .02,
    selection_geometry_weight: float = .01,
    support_geometry_weight: float = 0.0,
    support_geometry_guard_tolerance: float | None = None,
    fixed_dynamic_mask: torch.Tensor | None = None,
) -> WeightedBilinearTokenPlan:
    """Protected uniform skeleton plus TTT-proposed residual event bases.

    TTT is deliberately only a proposal mechanism.  A sensitive event token
    becomes physical only when its current-layer feature cannot be certified
    by the protected uniform dictionary.  All remaining patches then receive
    the same four-support convex certificate and quality-aware attention used
    by the other sparse-convex planners.
    """
    if tokens.shape[0] != 1 or max_support != 4:
        raise ValueError("protected TTT planner requires batch one and four supports")
    if os.environ.get("VGGT_TTT_FIXED_TOPOLOGY", "0") == "1":
        return _build_ttt_fixed_topology_plan(
            tokens, frames=frames, tokens_per_frame=tokens_per_frame,
            patch_token_start=patch_token_start, patch_hw=patch_hw,
            threshold=threshold, preserve_queries=preserve_queries,
            preserve_destination_keys=preserve_destination_keys,
            fixed_dynamic_mask=fixed_dynamic_mask,
        )
    height, width = patch_hw
    patches = height * width
    patch_count = frames * patches
    total, device = tokens.shape[1], tokens.device
    if total != frames * tokens_per_frame or patch_token_start + patches != tokens_per_frame:
        raise ValueError("invalid protected TTT token layout")
    if labels.numel() != patch_count or sensitivity.numel() != patch_count:
        raise ValueError("TTT labels/sensitivity must contain one value per patch token")
    if geometry.shape != (patch_count, 4):
        raise ValueError("TTT geometry must be [frames*patches,4]")
    if selection_geometry_weight < 0 or support_geometry_weight < 0:
        raise ValueError("TTT geometry weights must be nonnegative")
    if (support_geometry_guard_tolerance is not None
            and support_geometry_guard_tolerance < 0):
        raise ValueError("TTT geometry guard tolerance must be nonnegative")
    labels = labels.to(device=device, dtype=torch.long)
    sensitivity = sensitivity.to(device=device, dtype=torch.float32)
    geometry = geometry.to(device=device, dtype=torch.float32)
    patch_ids = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + patch_token_start + torch.arange(patches, device=device)[None]
    ).flatten()
    features = F.normalize(
        tokens[0].index_select(0, patch_ids).float(), dim=-1, eps=1e-8,
    ).to(torch.bfloat16)

    stable_basis, skeleton_stats = _protected_uniform_skeleton(
        features, frames=frames, patch_hw=patch_hw,
        top1_threshold=top1_prune_threshold,
    )
    stable_mask = torch.zeros(patch_count, device=device, dtype=torch.bool)
    stable_mask[stable_basis] = True
    freed_budget = int(skeleton_stats["top1_pruned_uniform_bases"])
    pruning_enabled = top1_prune_threshold <= 1.0
    event_budget = (
        freed_budget if pruning_enabled
        else int(round(sensitive_event_ratio * patch_count))
    )

    fixed_dynamic_path = os.environ.get("VGGT_TTT_FIXED_DYNAMIC_BASIS_PATH")
    fixed_dynamic_minimum = int(os.environ.get("VGGT_TTT_FIXED_DYNAMIC_MIN_LAYERS", "2"))
    fixed_dynamic = None
    predicted_fixed_dynamic = fixed_dynamic_mask is not None
    if predicted_fixed_dynamic:
        if fixed_dynamic_mask.numel() != patch_count:
            raise ValueError("predicted fixed dynamic basis mask has the wrong token count")
        fixed_dynamic = fixed_dynamic_mask.to(
            device=device, dtype=torch.bool,
        ).nonzero().flatten()
        fixed_dynamic = fixed_dynamic[~stable_mask.index_select(0, fixed_dynamic)]
    elif fixed_dynamic_path:
        frequency = _TTT_FIXED_DYNAMIC_CACHE.get(fixed_dynamic_path)
        if frequency is None:
            payload = torch.load(fixed_dynamic_path, map_location="cpu", weights_only=False)
            frequency = payload["dynamic"].to(torch.int16)
            _TTT_FIXED_DYNAMIC_CACHE[fixed_dynamic_path] = frequency
        if len(frequency) != patch_count:
            raise ValueError("fixed dynamic basis frequency has the wrong token count")
        fixed_dynamic = (frequency.to(device) >= fixed_dynamic_minimum).nonzero().flatten()
        fixed_dynamic = fixed_dynamic[~stable_mask.index_select(0, fixed_dynamic)]

    # Ask for at most twice the available reserve so the feature certificate,
    # rather than TTT alone, decides which event representatives are retained.
    event_proposal = (
        fixed_dynamic if fixed_dynamic is not None else
        _ttt_event_bases(
            features, geometry, labels, sensitivity, stable_mask,
            frames=frames, patch_hw=patch_hw, amount=2 * event_budget,
            geometry_weight=selection_geometry_weight,
        )
    )
    event = event_proposal if fixed_dynamic is not None else event_proposal[:0]
    event_rejected_as_representable = 0
    temporal_local_search = os.environ.get(
        "VGGT_TTT_LOCAL_SUPPORT_SEARCH", "1",
    ) != "0"
    if fixed_dynamic is None and event_budget and event_proposal.numel():
        fit_kwargs = dict(
            geometry=geometry if support_geometry_weight else None,
            geometry_weight=support_geometry_weight,
            geometry_guard_tolerance=support_geometry_guard_tolerance,
        )
        event_fit = (
            _adaptive_fit_temporal_local(
                features, stable_basis, frames=frames, patches=patches,
                threshold=threshold, source_ids=event_proposal, **fit_kwargs,
            )
            if temporal_local_search else
            _adaptive_fit_fused(
                features, stable_basis, max_support=4, threshold=threshold,
                source_ids=event_proposal,
                chunk=max(256, int(os.environ.get("VGGT_TTT_FIT_CHUNK", "4096"))),
            )
        )
        residual_limit = (2.0 * (1.0 - threshold)) ** .5
        represented = (
            (event_fit["cosine"] >= threshold)
            & (event_fit["residual"] <= residual_limit)
            & (event_fit["reconstruction_norm"] >= .80)
        )
        candidate = (~represented).nonzero().flatten()
        event_rejected_as_representable = int(represented.sum())
        if candidate.numel() > event_budget:
            candidate_ids = event_proposal.index_select(0, candidate)
            priority = (
                sensitivity.index_select(0, candidate_ids)
                * event_fit["residual"].index_select(0, candidate)
            )
            candidate = candidate.index_select(0, priority.topk(event_budget).indices)
        event = event_proposal.index_select(0, candidate)

    basis = torch.cat((stable_basis, event)).unique()
    basis_mask = torch.zeros(patch_count, device=device, dtype=torch.bool)
    basis_mask[basis] = True
    source = (~basis_mask).nonzero().flatten()
    fit_kwargs = dict(
        geometry=geometry if support_geometry_weight else None,
        geometry_weight=support_geometry_weight,
        geometry_guard_tolerance=support_geometry_guard_tolerance,
    )
    fit = (
        _adaptive_fit_temporal_local(
            features, basis, frames=frames, patches=patches,
            threshold=threshold, source_ids=source, **fit_kwargs,
        )
        if temporal_local_search else
        _adaptive_fit_fused(
            features, basis, max_support=4, threshold=threshold,
            source_ids=source,
            chunk=max(256, int(os.environ.get("VGGT_TTT_FIT_CHUNK", "4096"))),
        )
    )
    residual_limit = (2.0 * (1.0 - threshold)) ** .5
    accepted = (
        (fit["cosine"] >= threshold)
        & (fit["residual"] <= residual_limit)
        & (fit["reconstruction_norm"] >= .80)
    )
    if _TTT_TRACE_PATH:
        trace: dict[str, torch.Tensor | int | float] = {
            "layer": len(_TTT_TRACE_RECORDS),
            "patches": patches,
            "frames": frames,
            "stable_basis": stable_basis.detach().to("cpu", torch.int32),
            "event_proposal": event_proposal.detach().to("cpu", torch.int32),
            "event_basis": event.detach().to("cpu", torch.int32),
            "source_ids": source.detach().to("cpu", torch.int32),
            "support_ids": fit["support_ids"].detach().to("cpu", torch.int32),
            "weights": fit["weights"].detach().to("cpu", torch.float16),
            "cosine": fit["cosine"].detach().to("cpu", torch.float16),
            "residual": fit["residual"].detach().to("cpu", torch.float16),
            "reconstruction_norm": fit["reconstruction_norm"].detach().to(
                "cpu", torch.float16,
            ),
            "accepted": accepted.detach().to("cpu"),
        }
        if not _TTT_TRACE_RECORDS:
            trace.update({
                "labels": labels.detach().to("cpu", torch.uint8),
                "sensitivity": sensitivity.detach().to("cpu", torch.float16),
                "geometry": geometry.detach().to("cpu", torch.float16),
            })
        _TTT_TRACE_RECORDS.append(trace)
    rows = accepted.nonzero().flatten()
    source_local = source.index_select(0, rows)
    support_local = fit["support_ids"].index_select(0, rows)
    weights = fit["weights"].index_select(0, rows).float()
    selected_cosine = fit["cosine"].index_select(0, rows)

    kept_mask = torch.ones(patch_count, device=device, dtype=torch.bool)
    kept_mask[source_local] = False
    keep_local = kept_mask.nonzero().flatten()
    promoted = (~accepted).sum()
    basis_original = patch_ids.index_select(0, keep_local)
    source_original = patch_ids.index_select(0, source_local)
    support_original = patch_ids.index_select(
        0, support_local.reshape(-1),
    ).reshape_as(support_local)
    special = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + torch.arange(patch_token_start, device=device)[None]
    ).flatten()
    keep = torch.cat((special, basis_original)).sort().values.view(1, -1)
    original_to_compact = torch.full((1, total), -1, device=device, dtype=torch.long)
    original_to_compact.scatter_(
        1, keep, torch.arange(keep.shape[1], device=device).view(1, -1),
    )
    parent_compact = original_to_compact.gather(
        1, support_original.reshape(1, -1),
    ).reshape_as(support_original)
    if (parent_compact < 0).any():
        raise RuntimeError("protected TTT convex support must remain physical")
    mass = deterministic_weighted_normalizer(
        parent_compact.view(1, len(source_local), 4),
        weights.view(1, len(source_local), 4), keep.shape[1],
    )
    special_compact = original_to_compact[0, special]
    initial_budget = int(skeleton_stats["initial_uniform_bases"])
    return WeightedBilinearTokenPlan(
        keep=keep, source_original=source_original.view(1, -1),
        source_parent_compact=parent_compact.view(1, len(source_local), 4),
        source_pool_weights=weights.view(1, len(source_local), 4),
        source_restore_weights=weights.view(1, len(source_local), 4),
        original_tokens=total, normalizer=mass, preserve_queries=preserve_queries,
        preserve_destination_keys=(preserve_queries if preserve_destination_keys is None
                                   else preserve_destination_keys),
        dense_special_queries=True, special_query_original=special,
        special_query_compact=special_compact, key_log_mass=mass[..., 0].log(),
        metadata={
            "planner": "ttt_protected_uniform_residual_convex_v1",
            "threshold": float(threshold), "max_support": 4,
            "top1_prune_threshold": float(top1_prune_threshold),
            "top1_pruning_enabled": bool(pruning_enabled),
            **skeleton_stats,
            "freed_basis_budget": freed_budget,
            "sensitive_event_budget": event_budget,
            "sensitive_event_ratio_cap": float(sensitive_event_ratio),
            "fixed_dynamic_oracle": fixed_dynamic_path is not None,
            "predicted_fixed_dynamic": bool(predicted_fixed_dynamic),
            "fixed_dynamic_minimum_layers": fixed_dynamic_minimum,
            "selection_geometry_weight": float(selection_geometry_weight),
            "support_geometry_weight": float(support_geometry_weight),
            "support_geometry_guard_tolerance": support_geometry_guard_tolerance,
            "geometry_guard_enabled": bool(fit.get("geometry_guard_enabled", False)),
            "geometry_guard_selected": int(fit.get("geometry_guard_selected", 0)),
            "temporal_local_search": bool(fit.get("local_search", False)),
            "local_candidate_mean": float(fit.get("local_candidate_mean", 0.0)),
            "global_fallback_candidates": int(
                fit.get("global_fallback_candidates", 0)
            ),
            "streaming_top4_kernel": bool(fit.get("streaming_top4_kernel", False)),
            "active_set_kernel": bool(fit.get("active_set_kernel", False)),
            "sensitive_event_proposals": int(len(event_proposal)),
            "sensitive_event_representable": event_rejected_as_representable,
            "sensitive_event_bases": int(len(event)),
            "fit_failure_promotions": int(promoted),
            "physical_patch_tokens": int(len(keep_local)),
            "initial_budget_overflow": int(len(keep_local) - initial_budget),
            "patch_keep_ratio": float(len(keep_local) / patch_count),
            "merge_ratio": float(len(source_local) / total),
            "fit_mean": float(selected_cosine.mean()) if len(source_local) else 0.0,
            "max_attention_mass": float(mass.max()),
            "stable_ratio": float((labels == 0).float().mean()),
            "sensitive_ratio": float((labels == 1).float().mean()),
            "uncertain_ratio": float((labels == 2).float().mean()),
            "preserve_queries": bool(preserve_queries),
        },
    )


@torch.no_grad()
def build_ttt_partitioned_convex_plan(
    tokens: torch.Tensor, *, frames: int, tokens_per_frame: int,
    patch_token_start: int, patch_hw: tuple[int, int], threshold: float,
    labels: torch.Tensor, sensitivity: torch.Tensor, geometry: torch.Tensor,
    max_support: int = 4, preserve_queries: bool = False,
    preserve_destination_keys: bool | None = None, cell_size: int = 4,
    bases_per_cell: int = 2, stable_global_ratio: float = .01,
    sensitive_event_ratio: float = .02,
    selection_geometry_weight: float = .01,
    support_geometry_weight: float = .0025,
    support_geometry_guard_tolerance: float | None = 0.0,
) -> WeightedBilinearTokenPlan:
    """Real sparse forward for the validated TTT stable/event dictionary."""
    if tokens.shape[0] != 1 or max_support != 4:
        raise ValueError("TTT partitioned convex planner requires batch one and four supports")
    height, width = patch_hw
    patches = height * width
    total, device = tokens.shape[1], tokens.device
    patch_count = frames * patches
    if total != frames * tokens_per_frame or patch_token_start + patches != tokens_per_frame:
        raise ValueError("invalid TTT partitioned token layout")
    if labels.numel() != patch_count or sensitivity.numel() != patch_count:
        raise ValueError("TTT labels/sensitivity must contain one value per patch token")
    if geometry.shape != (patch_count, 4):
        raise ValueError("TTT geometry must be [frames*patches,4]")
    if selection_geometry_weight < 0 or support_geometry_weight < 0:
        raise ValueError("TTT geometry weights must be nonnegative")
    if (support_geometry_guard_tolerance is not None
            and support_geometry_guard_tolerance < 0):
        raise ValueError("TTT geometry guard tolerance must be nonnegative")
    labels = labels.to(device=device, dtype=torch.long)
    sensitivity = sensitivity.to(device=device, dtype=torch.float32)
    geometry = geometry.to(device=device, dtype=torch.float32)
    patch_ids = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + patch_token_start + torch.arange(patches, device=device)[None]
    ).flatten()
    features = F.normalize(
        tokens[0].index_select(0, patch_ids).float(), dim=-1, eps=1e-8,
    ).to(torch.bfloat16)
    first = torch.arange(patches, device=device)
    later = torch.arange(patch_count, device=device) >= patches
    stable_allowed = (labels == 0) & later
    local = _ttt_local_bases(
        features, stable_allowed, frames=frames, patch_hw=patch_hw,
        cell_size=cell_size, bases_per_cell=bases_per_cell,
    )
    stable_seed = torch.cat((first, local)).unique()
    stable_global = _ttt_residual_bases(
        features, stable_seed, int(round(stable_global_ratio * patch_count)), labels == 0,
    )
    uncertain = ((labels == 2) & later).nonzero().flatten()
    initial = torch.cat((stable_seed, stable_global, uncertain)).unique()
    initial_mask = torch.zeros(patch_count, device=device, dtype=torch.bool)
    initial_mask[initial] = True
    event = _ttt_event_bases(
        features, geometry, labels, sensitivity, initial_mask,
        frames=frames, patch_hw=patch_hw,
        amount=int(round(sensitive_event_ratio * patch_count)),
        geometry_weight=selection_geometry_weight,
    )
    basis = torch.cat((initial, event)).unique()
    basis_mask = torch.zeros(patch_count, device=device, dtype=torch.bool)
    basis_mask[basis] = True
    stable_basis = basis_mask & (labels == 0)
    reliable_basis = basis_mask & (labels != 2)

    accepted_ids, accepted_support, accepted_weights, accepted_cosine = [], [], [], []
    for source_mask, support_mask, use_geometry in (
        ((labels == 0) & ~basis_mask, stable_basis, False),
        ((labels == 1) & ~basis_mask, reliable_basis, True),
    ):
        source = source_mask.nonzero().flatten()
        if not len(source):
            continue
        fit = _adaptive_fit_fused(
            features, support_mask.nonzero().flatten(), max_support=4,
            threshold=threshold, source_ids=source, chunk=8192,
            geometry=geometry if use_geometry and support_geometry_weight else None,
            geometry_weight=support_geometry_weight if use_geometry else 0.0,
            geometry_guard_tolerance=(support_geometry_guard_tolerance
                                      if use_geometry else None),
        )
        limit = (2.0 * (1.0 - threshold)) ** .5
        accepted = (
            (fit["cosine"] >= threshold) & (fit["residual"] <= limit)
            & (fit["reconstruction_norm"] >= .80)
        )
        rows = accepted.nonzero().flatten()
        accepted_ids.append(source.index_select(0, rows))
        accepted_support.append(fit["support_ids"].index_select(0, rows))
        accepted_weights.append(fit["weights"].index_select(0, rows).float())
        accepted_cosine.append(fit["cosine"].index_select(0, rows))
    source_local = torch.cat(accepted_ids) if accepted_ids else first[:0]
    support_local = torch.cat(accepted_support) if accepted_support else first[:0].reshape(0, 4)
    weights = torch.cat(accepted_weights) if accepted_weights else features.new_empty((0, 4)).float()
    selected_cosine = torch.cat(accepted_cosine) if accepted_cosine else features.new_empty(0).float()
    kept_mask = torch.ones(patch_count, device=device, dtype=torch.bool)
    kept_mask[source_local] = False
    keep_local = kept_mask.nonzero().flatten()
    basis_original = patch_ids.index_select(0, keep_local)
    source_original = patch_ids.index_select(0, source_local)
    support_original = patch_ids.index_select(0, support_local.reshape(-1)).reshape_as(support_local)
    special = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + torch.arange(patch_token_start, device=device)[None]
    ).flatten()
    keep = torch.cat((special, basis_original)).sort().values.view(1, -1)
    original_to_compact = torch.full((1, total), -1, device=device, dtype=torch.long)
    original_to_compact.scatter_(
        1, keep, torch.arange(keep.shape[1], device=device).view(1, -1),
    )
    parent_compact = original_to_compact.gather(
        1, support_original.reshape(1, -1),
    ).reshape_as(support_original)
    if (parent_compact < 0).any():
        raise RuntimeError("TTT convex support must remain physical")
    mass = deterministic_weighted_normalizer(
        parent_compact.view(1, len(source_local), 4),
        weights.view(1, len(source_local), 4), keep.shape[1],
    )
    special_compact = original_to_compact[0, special]
    return WeightedBilinearTokenPlan(
        keep=keep, source_original=source_original.view(1, -1),
        source_parent_compact=parent_compact.view(1, len(source_local), 4),
        source_pool_weights=weights.view(1, len(source_local), 4),
        source_restore_weights=weights.view(1, len(source_local), 4),
        original_tokens=total, normalizer=mass, preserve_queries=preserve_queries,
        preserve_destination_keys=(preserve_queries if preserve_destination_keys is None
                                   else preserve_destination_keys),
        dense_special_queries=True, special_query_original=special,
        special_query_compact=special_compact, key_log_mass=mass[..., 0].log(),
        metadata={
            "planner": "ttt_partitioned_stable_event_convex_v1",
            "threshold": float(threshold), "max_support": 4,
            "stable_local_bases": int(len(local)),
            "stable_global_residual_bases": int(len(stable_global)),
            "sensitive_event_bases": int(len(event)),
            "selection_geometry_weight": float(selection_geometry_weight),
            "support_geometry_weight": float(support_geometry_weight),
            "support_geometry_guard_tolerance": support_geometry_guard_tolerance,
            "protected_uncertain_tokens": int(len(uncertain)),
            "physical_patch_tokens": int(len(keep_local)),
            "patch_keep_ratio": float(len(keep_local) / patch_count),
            "merge_ratio": float(len(source_local) / total),
            "fit_mean": float(selected_cosine.mean()) if len(source_local) else 0.0,
            "stable_ratio": float((labels == 0).float().mean()),
            "sensitive_ratio": float((labels == 1).float().mean()),
            "uncertain_ratio": float((labels == 2).float().mean()),
            "preserve_queries": bool(preserve_queries),
        },
    )


@torch.no_grad()
def build_hash_convex_plan(
    tokens: torch.Tensor, *, frames: int, tokens_per_frame: int, patch_token_start: int,
    patch_hw: tuple[int, int], threshold: float, hash_bits: int = 12,
    preserve_queries: bool = False, preserve_destination_keys: bool | None = None,
) -> WeightedBilinearTokenPlan:
    """Training-free one-pass semantic hash medoids with four-way convex certificates."""
    if tokens.shape[0] != 1 or frames < 2 or not 4 <= hash_bits <= 16:
        raise ValueError("hash-convex plan requires batch one, multiple frames, and 4..16 hash bits")
    height, width = patch_hw
    patches = height * width
    total, device = tokens.shape[1], tokens.device
    if total != frames * tokens_per_frame or patch_token_start + patches != tokens_per_frame:
        raise ValueError("invalid hash-convex token layout")
    patch_ids = (torch.arange(frames, device=device)[:, None] * tokens_per_frame
                 + patch_token_start + torch.arange(patches, device=device)[None]).flatten()
    features = F.normalize(tokens[0].index_select(0, patch_ids).float(), dim=-1, eps=1e-8)
    projection = _fixed_hash_projection(features)
    route = F.normalize(features.to(torch.bfloat16) @ projection, dim=-1, eps=1e-8).float()
    bit_weight = (1 << torch.arange(hash_bits, device=device, dtype=torch.long))
    codes = ((route[:, :hash_bits] >= 0).long() * bit_weight).sum(dim=-1)
    buckets = 1 << hash_bits
    sums = torch.zeros((buckets, route.shape[1]), device=device, dtype=torch.float32)
    counts = torch.zeros(buckets, device=device, dtype=torch.float32)
    sums.index_add_(0, codes, route)
    counts.index_add_(0, codes, torch.ones(len(route), device=device))
    centers = F.normalize(sums / counts.clamp_min(1)[:, None], dim=-1, eps=1e-8)
    center_score = (route * centers.index_select(0, codes)).sum(dim=-1)
    best_score = torch.full((buckets,), -torch.inf, device=device)
    best_score.scatter_reduce_(0, codes, center_score, reduce="amax", include_self=True)
    token_index = torch.arange(len(features), device=device)
    eligible = center_score >= best_score.index_select(0, codes) - 1e-7
    proposed = torch.where(eligible, token_index, torch.full_like(token_index, len(features)))
    representative = torch.full((buckets,), len(features), device=device, dtype=torch.long)
    representative.scatter_reduce_(0, codes, proposed, reduce="amin", include_self=True)
    representative[representative == len(features)] = -1
    medoids = representative[representative >= 0]
    first_frame = torch.arange(patches, device=device)
    basis_local = torch.cat((first_frame, medoids)).unique()
    fit = _hash_four_convex_fit(features, route, codes, representative, hash_bits=hash_bits)
    residual_limit = (2.0 * (1.0 - threshold)) ** .5
    accepted = ((fit["cosine"] >= threshold) & (fit["residual"] <= residual_limit)
                & (fit["reconstruction_norm"] >= .80))
    accepted[:patches] = False
    accepted[basis_local] = False
    source_local = accepted.nonzero().flatten()
    keep_local = (~accepted).nonzero().flatten()
    basis_original = patch_ids.index_select(0, keep_local)
    source_original = patch_ids.index_select(0, source_local)
    support_local = fit["support_ids"].index_select(0, source_local)
    support_original = patch_ids.index_select(0, support_local.reshape(-1)).reshape_as(support_local)
    weights = fit["weights"].index_select(0, source_local).float()
    special = (torch.arange(frames, device=device)[:, None] * tokens_per_frame
               + torch.arange(patch_token_start, device=device)[None]).flatten()
    keep = torch.cat((special, basis_original)).sort().values.view(1, -1)
    identity = torch.arange(total, device=device).view(1, -1)
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=device).view(1, -1))
    parent_compact = original_to_compact.gather(1, support_original.reshape(1, -1)).reshape_as(support_original)
    if (parent_compact < 0).any():
        raise RuntimeError("hash-convex support must remain physical")
    mass = torch.ones((1, keep.shape[1], 1), device=device, dtype=torch.float32)
    for support_index in range(4):
        mass.scatter_add_(1, parent_compact[None, :, support_index, None],
                          weights[None, :, support_index, None])
    special_compact = original_to_compact[0, special]
    selected_cosine = fit["cosine"].index_select(0, source_local)
    return WeightedBilinearTokenPlan(
        keep=keep, source_original=source_original.view(1, -1),
        source_parent_compact=parent_compact.view(1, len(source_local), 4),
        source_pool_weights=weights.view(1, len(source_local), 4),
        source_restore_weights=weights.view(1, len(source_local), 4),
        original_tokens=total, normalizer=mass, preserve_queries=preserve_queries,
        preserve_destination_keys=(preserve_queries if preserve_destination_keys is None
                                   else preserve_destination_keys),
        dense_special_queries=True, special_query_original=special,
        special_query_compact=special_compact, key_log_mass=mass[..., 0].log(),
        metadata={"planner": "training_free_hash_convex_v1", "threshold": float(threshold),
                  "hash_bits": int(hash_bits), "occupied_buckets": int(len(medoids)),
                  "merge_ratio": float(len(source_local) / total),
                  "patch_keep_ratio": float(len(keep_local) / len(features)),
                  "fit_mean": float(selected_cosine.mean()) if len(source_local) else 0.0,
                  "preserve_queries": bool(preserve_queries)},
    )


def _convex_fit(source: torch.Tensor, supports: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact four-variable simplex least squares via active-set enumeration.

    The problem is min ||S a - x||^2 over the probability simplex.  With four
    variables only 15 active sets exist, so every KKT system can be solved
    simultaneously as one batched 5x5 linear solve.  This replaces the previous
    24-step projected-gradient loop and its ~18 kernel launches per iteration.
    """
    source, supports = source.float(), supports.float()
    count, device = source.shape[0], source.device
    gram = supports @ supports.transpose(1, 2)
    rhs = torch.bmm(supports, source.unsqueeze(-1)).squeeze(-1)
    active = _ACTIVE_SETS.to(device)  # [15,4]
    pair = active[:, :, None] * active[:, None, :]
    inactive_diag = torch.diag_embed(1.0 - active)
    block = 2.0 * gram[:, None] * pair[None]
    # Inactive variables are pinned with an identity row; the extra ridge keeps
    # the near-duplicate support Gram invertible without changing the solution
    # materially (entries are O(1), the ridge is O(1e-4)).
    block = block * (1.0 - inactive_diag) + inactive_diag + 2.0 * _RIDGE * torch.diag_embed(active)
    matrix = torch.zeros(count, 15, 5, 5, device=device, dtype=torch.float32)
    matrix[..., :4, :4] = block
    matrix[..., :4, 4] = active
    matrix[..., 4, :4] = 1.0
    target = torch.zeros(count, 15, 5, device=device, dtype=torch.float32)
    target[..., :4] = 2.0 * rhs[:, None] * active[None]
    target[..., 4] = 1.0
    solution, info = torch.linalg.solve_ex(matrix, target)
    alpha = solution[..., :4].clamp_min(0.0)
    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    quadratic = torch.matmul(gram[:, None], alpha.unsqueeze(-1)).squeeze(-1)
    objective = (alpha * quadratic).sum(dim=-1) - 2.0 * (alpha * rhs[:, None]).sum(dim=-1)
    feasible = (info == 0) & (solution[..., :4] >= -1e-5).all(dim=-1)
    objective = torch.where(feasible, objective, torch.inf)
    best = objective.argmin(dim=-1)
    alpha = alpha.gather(1, best[:, None, None].expand(-1, 1, 4)).squeeze(1)
    numerator = (alpha * rhs).sum(dim=-1)
    squared_norm = (alpha * torch.bmm(gram, alpha.unsqueeze(-1)).squeeze(-1)).sum(dim=-1)
    cosine = numerator / squared_norm.clamp_min(1e-12).sqrt()
    single = rhs[:, 0]
    one_hot = single > cosine
    if one_hot.any():
        alpha[one_hot] = 0
        alpha[one_hot, 0] = 1
    return torch.maximum(cosine, single), alpha


def _convex_fit_dispatch(
    source: torch.Tensor, supports: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the fused exact active-set kernel when explicitly enabled."""
    solver = os.environ.get("VGGT_TTT_LOCAL_SOLVER", "projected")
    if solver == "projected":
        source, supports = source.float(), supports.float()
        gram = supports @ supports.transpose(1, 2)
        rhs = (supports @ source.unsqueeze(-1)).squeeze(-1)
        alpha = four_support_projected_gradient(gram, rhs, steps=32)
        reconstruction = (supports * alpha.unsqueeze(-1)).sum(dim=1)
        return F.cosine_similarity(
            source, reconstruction, dim=-1, eps=1e-8,
        ), alpha
    if solver != "exact":
        raise ValueError("VGGT_TTT_LOCAL_SOLVER must be exact or projected")
    use_kernel = (
        source.is_cuda and triton_acceleration_available()
        and os.environ.get("VGGT_TRITON_ACTIVE_SET", "0") == "1"
    )
    if not use_kernel:
        return _convex_fit(source, supports)
    source, supports = source.float(), supports.float()
    gram = supports @ supports.transpose(1, 2)
    rhs = (supports @ source.unsqueeze(-1)).squeeze(-1)
    alpha = four_support_active_set(gram, rhs, ridge=_RIDGE)
    reconstruction = (supports * alpha.unsqueeze(-1)).sum(dim=1)
    cosine = F.cosine_similarity(source, reconstruction, dim=-1, eps=1e-8)
    single = rhs[:, 0]
    one_hot = single > cosine
    if one_hot.any():
        alpha[one_hot] = 0
        alpha[one_hot, 0] = 1
    return torch.maximum(cosine, single), alpha


@torch.no_grad()
def build_sparse_convex_plan(
    tokens: torch.Tensor, *, frames: int, tokens_per_frame: int,
    patch_token_start: int, patch_hw: tuple[int, int], threshold: float = .90,
    preserve_queries: bool = False, match_chunk_size: int = FAST_MATCH_SOURCE_CHUNK,
    fit_chunk_size: int = 16384, preserve_destination_keys: bool | None = None,
) -> WeightedBilinearTokenPlan:
    if tokens.shape[0] != 1:
        raise ValueError("sparse-convex compaction currently supports batch size one")
    _, total, _ = tokens.shape
    height, width = patch_hw
    patches = height * width
    if total != frames * tokens_per_frame or patch_token_start + patches != tokens_per_frame:
        raise ValueError("invalid frame/token layout for sparse-convex compaction")
    device = tokens.device
    identity = torch.arange(total, device=device).view(1, -1)
    patch_ids = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + patch_token_start + torch.arange(patches, device=device)[None, :]
    )

    frame0_ids = patch_ids[0]
    if frames > 1:
        generator = torch.Generator(device="cpu").manual_seed(0)
        rows, cols = torch.arange(0, height, 2), torch.arange(0, width, 2)
        choice = torch.rand(frames - 1, len(rows), len(cols), 2, generator=generator)
        row = rows[None, :, None] + (choice[..., 0] * (height - rows).clamp(max=2)[None, :, None]).long()
        col = cols[None, None, :] + (choice[..., 1] * (width - cols).clamp(max=2)[None, None, :]).long()
        anchor_position = (row * width + col).reshape(frames - 1, -1).to(device)
        anchor_ids = patch_ids[1:].gather(1, anchor_position)
        candidate_mask = torch.ones((frames - 1, patches), device=device, dtype=torch.bool)
        candidate_mask.scatter_(1, anchor_position, False)
        candidate_position = torch.arange(patches, device=device).expand(frames - 1, patches)[
            candidate_mask
        ].reshape(frames - 1, -1)
        candidate = patch_ids[1:].gather(1, candidate_position).flatten()
        destination = torch.cat((frame0_ids, anchor_ids.flatten()))
    else:
        candidate = torch.empty(0, device=device, dtype=torch.long)
        destination = frame0_ids

    normalized = F.normalize(tokens[0], dim=-1, eps=1e-6)

    count = candidate.numel()
    support = torch.empty((count, 4), device=device, dtype=torch.long)
    top1 = torch.empty(count, device=device, dtype=normalized.dtype)
    if frames > 1:
        per_frame = candidate_position.shape[1]
        # A source is matched against the reference frame plus the anchor banks
        # of its temporal neighbourhood instead of the whole dictionary.  Local
        # acceptance can only miss merges, never invent them: every source that
        # fails locally is re-evaluated against the full dictionary below.
        for begin in range(1, frames, _LOCAL_FRAME_GROUP):
            end = min(begin + _LOCAL_FRAME_GROUP, frames)
            anchor_begin = max(begin - 2, 0)
            anchor_end = min(end - 1, frames - 2) + 1
            local = torch.cat((frame0_ids, anchor_ids[anchor_begin:anchor_end].flatten()))
            positions = candidate_position[begin - 1:end - 1]
            source_ids = patch_ids[begin:end].gather(1, positions).flatten()
            value, local_index = (
                normalized[source_ids].to(torch.bfloat16) @ normalized[local].to(torch.bfloat16).T
            ).topk(4, dim=-1)
            offset = (begin - 1) * per_frame
            support[offset:offset + source_ids.numel()] = local[local_index]
            top1[offset:offset + source_ids.numel()] = value[:, 0]

    fit = torch.empty(count, device=device, dtype=torch.float32)
    alpha = torch.zeros((count, 4), device=device, dtype=torch.float32)
    # The one-hot top-1 solution is feasible for every candidate, so a source
    # whose best local match clears the threshold is accepted without solving
    # the four-variable problem.  A small margin keeps BF16 score quantisation
    # from moving decisions across the threshold.
    easy = top1 >= threshold + _EARLY_MARGIN
    hard = (~easy).nonzero().flatten()
    alpha[easy, 0] = 1.0
    fit[easy] = top1[easy].float()
    for begin in range(0, hard.numel(), fit_chunk_size):
        index = hard[begin:begin + fit_chunk_size]
        source = F.normalize(normalized[candidate[index]].float(), dim=-1, eps=1e-6)
        parent_feature = F.normalize(normalized[support[index]].float(), dim=-1, eps=1e-6)
        fit[index], alpha[index] = _convex_fit(source, parent_feature)
    failed = hard[fit[hard] < threshold]
    if failed.numel():
        dictionary = normalized[destination].to(torch.bfloat16)
        for begin in range(0, failed.numel(), match_chunk_size):
            index = failed[begin:begin + match_chunk_size]
            source = normalized[candidate[index]]
            _, local = (source.to(torch.bfloat16) @ dictionary.T).topk(4, dim=-1)
            support[index] = destination[local]
            source = F.normalize(source.float(), dim=-1, eps=1e-6)
            parent_feature = F.normalize(normalized[support[index]].float(), dim=-1, eps=1e-6)
            fit[index], alpha[index] = _convex_fit(source, parent_feature)
    selected = fit >= threshold
    source_original = candidate[selected].view(1, -1)
    source_support = support[selected].view(1, -1, 4)
    weights = alpha[selected].view(1, -1, 4)

    kept_mask = torch.ones(total, device=device, dtype=torch.bool)
    kept_mask[source_original[0]] = False
    keep = identity[:, kept_mask]
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=device).view(1, -1))
    parent_compact = original_to_compact.gather(1, source_support.reshape(1, -1)).reshape_as(source_support)
    if (parent_compact < 0).any():
        raise RuntimeError("sparse-convex support must remain physical")
    mass = torch.ones((1, keep.shape[1], 1), device=device, dtype=torch.float32)
    for corner in range(4):
        mass.scatter_add_(1, parent_compact[:, :, corner, None], weights[:, :, corner, None])
    special_original = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + torch.arange(patch_token_start, device=device)[None, :]
    ).flatten()
    special_compact = original_to_compact[0, special_original]
    return WeightedBilinearTokenPlan(
        keep=keep, source_original=source_original,
        source_parent_compact=parent_compact,
        source_pool_weights=weights, source_restore_weights=weights,
        original_tokens=total, normalizer=mass,
        preserve_queries=preserve_queries,
        preserve_destination_keys=(
            preserve_queries if preserve_destination_keys is None else preserve_destination_keys
        ),
        dense_special_queries=True,
        special_query_original=special_original,
        special_query_compact=special_compact,
        key_log_mass=mass[..., 0].log(),
        metadata={
            "planner": "fixed_topology_sparse_convex_v1",
            "threshold": float(threshold),
            "merge_ratio": float(source_original.numel() / total),
            "fit_mean": float(fit[selected].mean()) if selected.any() else 0.0,
            "preserve_queries": bool(preserve_queries),
            "search": "local_frame_group_exact_fallback_v1",
            "solver": "simplex_active_set_exact_v1",
            "early_accept_margin": float(_EARLY_MARGIN),
            "hard_candidates": int(hard.numel()),
            "fallback_candidates": int(failed.numel()),
        },
    )
