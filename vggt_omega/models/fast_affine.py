"""Fast bounded affine-star planner and unified Q/K/V attention quotient.

The planner estimates adjacent-frame affine transport, composes it only inside
short anchor windows, and certifies a patch against the corresponding bilinear
blend in the window's physical anchor frame.  Certified patches use one shared
quotient for Q, K and V and are restored with the same four weights.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from vggt_omega.models.token_merging import WeightedBilinearTokenPlan


# ---- optional profiler (FAST_AFFINE_PROFILE=1) -------------------------------
import os as _os
import atexit as _atexit

_PROFILE = _os.environ.get("FAST_AFFINE_PROFILE") == "1"
_PROFILE_STATE = {"planner_ms": 0.0, "calls": 0}

# Optional reduced-precision attention operands.  ``scaled_dot_product_attention``
# can only reach the flash kernel with fp16/bf16 inputs; the checkpoint is fp32,
# so every attention call currently lands on the (much slower) memory-efficient
# fp32 kernel.  Casting Q/K/V just around SDPA keeps the rest of the network in
# fp32.  Set FAST_AFFINE_ATTN_DTYPE=bf16 to enable.
_ATTN_DTYPE = {
    "": None, "none": None, "fp32": None,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16,
}.get(_os.environ.get("FAST_AFFINE_ATTN_DTYPE", "").lower(), None)

# RoPE runs in the activation dtype by default (the reference promotes to fp32).
# FAST_AFFINE_ROPE_FP32=1 restores the reference precision with the same
# allocation pattern, for an accuracy/speed A/B.
_ROPE_DTYPE = torch.float32 if _os.environ.get("FAST_AFFINE_ROPE_FP32") == "1" else None

# Planner diagnostics (merge ratios, validity fractions) are reported through
# ``.item()`` and therefore drain the device pipeline.  FAST_AFFINE_STATS=0
# skips those reads for timing runs; merge decisions are identical.
_STATS = _os.environ.get("FAST_AFFINE_STATS", "1") != "0"

# Dual-motion affine fitting: a second anchor-displacement mode captures
# independently moving objects that the single global affine cannot explain.
# The merge object (four bilinear neighbours) and the restoration weights are
# unchanged; only the per-token choice of affine improves.
_MULTIMODAL = _os.environ.get("FAST_AFFINE_MULTIMODAL", "0") == "1"
_MODE_PROBE_STRIDE = int(_os.environ.get("FAST_AFFINE_MODE_PROBE_STRIDE", "8"))

# Local correspondence search around the affine prediction (nine sub-patch
# offsets).  The four-neighbour bilinear merge and restoration are unchanged.
_LOCAL_SEARCH = _os.environ.get("FAST_AFFINE_LOCAL_SEARCH", "0") == "1"



def _profile_flush() -> None:
    if not _PROFILE or not _PROFILE_STATE["calls"]:
        return
    print(
        f"[fast-affine] planner_ms_total={_PROFILE_STATE['planner_ms']:.1f} "
        f"calls={_PROFILE_STATE['calls']} "
        f"per_call_ms={_PROFILE_STATE['planner_ms'] / _PROFILE_STATE['calls']:.2f}",
        flush=True,
    )


if _PROFILE:
    _atexit.register(_profile_flush)

__all__ = ["build_affine_warp_plan_fast", "install"]

# Bounds the [chunk, P, 4, C] neighbour gather.
_FRAME_CHUNK = 24
# Patch-grid displacements live in [-(size-1), size-1]; 33 >= 31 covers it.
_OFFSET = 36


def build_affine_warp_plan_fast(
    tokens: torch.Tensor,
    *,
    frames: int,
    tokens_per_frame: int,
    patch_token_start: int,
    patch_hw: tuple[int, int],
    similarity_threshold: float = .95,
    anchor_grid: int = 6,
    anchor_match_threshold: float = .70,
    assignment_mode: str = "soft_soft",
    dense_special_kv: bool = False,
    pool_source_content: bool = True,
    qkv_features: torch.Tensor | None = None,
    qkv_similarity_threshold: float | None = None,
    star_window: int = 2,
    target_merge_ratio: float | None = None,
    token_importance: torch.Tensor | None = None,
) -> WeightedBilinearTokenPlan:
    """Build one unified affine-star quotient for Q, K and V."""
    if tokens.shape[0] != 1:
        raise ValueError("affine-star compaction currently supports batch size one")
    if star_window < 2:
        raise ValueError("star_window must be at least two")
    if assignment_mode not in ("soft_soft", "hard_hard"):
        raise ValueError("affine-star assignment must be soft_soft or hard_hard")

    if _PROFILE:
        _ev0 = torch.cuda.Event(enable_timing=True)
        _ev1 = torch.cuda.Event(enable_timing=True)
        _ev0.record()

    height, width = patch_hw
    patches = height * width
    total = tokens.shape[1]
    device = tokens.device
    identity = torch.arange(total, device=device).view(1, -1)
    patch_index = torch.arange(patches, device=device)

    patch_rows = torch.arange(height, device=device)
    patch_cols = torch.arange(width, device=device)
    full_row, full_col = torch.meshgrid(patch_rows, patch_cols, indexing="ij")
    full_xy = torch.stack((full_col.reshape(-1), full_row.reshape(-1)), dim=-1).float()
    full_h = torch.cat((full_xy, torch.ones((patches, 1), device=device)), dim=-1)

    anchor_rows = torch.linspace(0, height - 1, min(anchor_grid, height), device=device).round().long().unique()
    anchor_cols = torch.linspace(0, width - 1, min(anchor_grid, width), device=device).round().long().unique()
    anchor_row, anchor_col = torch.meshgrid(anchor_rows, anchor_cols, indexing="ij")
    anchor_local = (anchor_row * width + anchor_col).reshape(-1)
    anchor_xy = full_xy[anchor_local]
    anchor_h = torch.cat((anchor_xy, torch.ones((anchor_xy.shape[0], 1), device=device)), dim=-1)

    reshaped = tokens.view(1, frames, tokens_per_frame, tokens.shape[-1])
    # One normalisation for the whole clip instead of 2 per frame.
    unit = F.normalize(reshaped[0, :, patch_token_start:].float(), dim=-1, eps=1e-6)

    merged = torch.zeros((1, total), dtype=torch.bool, device=device)
    parents = torch.zeros((1, total, 4), dtype=torch.long, device=device)
    weights_all = torch.zeros((1, total, 4), dtype=torch.float32, device=device)
    weights_all[..., 0] = 1.0
    pair_affine = torch.zeros((max(frames - 1, 0), 3, 2), dtype=torch.float32, device=device)

    valid_acc = torch.zeros((), device=device)
    inlier_acc = torch.zeros((), device=device)
    anchor_total = 0

    bins = 2 * _OFFSET + 1
    pair_count = frames - 1
    for start in range(0, pair_count, _FRAME_CHUNK):
        stop = min(start + _FRAME_CHUNK, pair_count)
        count = stop - start
        # Pair i compares observation frame i+1 against frame i.
        source = unit[start + 1: stop + 1]
        previous = unit[start: stop]

        anchor_source = source[:, anchor_local]
        anchor_score, anchor_match = torch.einsum(
            "nad,npd->nap", anchor_source, previous,
        ).max(dim=-1)
        match_xy = full_xy[anchor_match]
        displacement = (match_xy - anchor_xy[None]).round().long()
        candidate = anchor_score >= anchor_match_threshold

        # Exact modal displacement from a small 2-D histogram (no sort/unique).
        code = (
            (displacement[..., 0] + _OFFSET).clamp_(0, bins - 1) * bins
            + (displacement[..., 1] + _OFFSET).clamp_(0, bins - 1)
        )
        histogram = torch.zeros((count, bins * bins), dtype=torch.float32, device=device)
        histogram.scatter_add_(1, code, candidate.to(torch.float32))
        modal_code = histogram.argmax(dim=1)
        modal = torch.stack(
            (modal_code // bins - _OFFSET, modal_code % bins - _OFFSET), dim=-1,
        )
        has_candidate = candidate.any(dim=-1)
        modal = torch.where(has_candidate[:, None], modal, torch.zeros_like(modal))

        inlier = candidate & ((displacement - modal[:, None]).abs().amax(dim=-1) <= 1)

        def fit(rows: torch.Tensor) -> torch.Tensor:
            """Masked 3-parameter affine fit by weighted normal equations."""
            weight = rows.to(torch.float32).unsqueeze(-1)
            design = anchor_h[None] * weight
            lhs = torch.einsum("naj,nak->njk", design, anchor_h[None])
            rhs = torch.einsum("naj,nak->njk", design, match_xy * weight)
            ridge = torch.eye(3, device=device).unsqueeze(0) * 1e-6
            solution = torch.linalg.solve(lhs + ridge, rhs + ridge[..., :2])
            finite = torch.isfinite(solution).all(dim=(1, 2))
            unit_affine = torch.zeros((count, 3, 2), device=device)
            unit_affine[:, 0, 0] = 1.0
            unit_affine[:, 1, 1] = 1.0
            return torch.where(finite[:, None, None], solution, unit_affine + 0.0)

        translation = torch.zeros((count, 3, 2), device=device)
        translation[:, 0, 0] = 1.0
        translation[:, 1, 1] = 1.0
        translation[:, 2] = modal.float()

        enough = inlier.sum(dim=-1) >= 3
        affine = torch.where(enough[:, None, None], fit(inlier), translation)
        predicted = torch.einsum("naj,njk->nak", anchor_h[None], affine)
        reprojection = (predicted - match_xy).norm(dim=-1)
        refined = inlier & (reprojection <= 1.5)
        again = refined.sum(dim=-1) >= 3
        affine = torch.where(again[:, None, None], fit(refined), affine)
        inlier = torch.where(again[:, None], refined, inlier)
        pair_affine[start:stop] = affine

        warped = torch.einsum("pj,njk->npk", full_h, affine)
        x, y = warped.unbind(dim=-1)
        valid = (x >= 0.0) & (x <= float(width - 1)) & (y >= 0.0) & (y <= float(height - 1))
        x0 = x.floor().long().clamp(0, width - 2)
        y0 = y.floor().long().clamp(0, height - 2)
        fx = (x - x0.float()).clamp(0.0, 1.0)
        fy = (y - y0.float()).clamp(0.0, 1.0)
        local_parents = torch.stack((
            y0 * width + x0,
            y0 * width + (x0 + 1),
            (y0 + 1) * width + x0,
            (y0 + 1) * width + (x0 + 1),
        ), dim=-1)
        local_weights = torch.stack((
            (1.0 - fx) * (1.0 - fy), fx * (1.0 - fy),
            (1.0 - fx) * fy, fx * fy,
        ), dim=-1).float()

        if _MULTIMODAL and bool(candidate.any()):
            # Second motion mode: the strongest anchor-displacement peak that is
            # far from the dominant one.  Independently moving objects otherwise
            # inherit the camera affine and their four neighbours are wrong.
            grid_x = torch.arange(bins, device=device)[None, :].expand(bins, bins).reshape(-1)
            grid_y = torch.arange(bins, device=device)[:, None].expand(bins, bins).reshape(-1)
            modal_x = (modal_code // bins)[:, None]
            modal_y = (modal_code % bins)[:, None]
            far = ((grid_x[None] - modal_x).abs() > 4) | ((grid_y[None] - modal_y).abs() > 4)
            histogram_second = histogram.masked_fill(~far, 0.0)
            second_code = histogram_second.argmax(dim=1)
            has_second = histogram_second.max(dim=1).values >= 3.0
            second = torch.stack(
                (second_code // bins - _OFFSET, second_code % bins - _OFFSET), dim=-1,
            )
            second = torch.where(has_second[:, None], second, modal)

            inlier_second = candidate & ((displacement - second[:, None]).abs().amax(dim=-1) <= 1)
            translation_second = torch.zeros((count, 3, 2), device=device)
            translation_second[:, 0, 0] = 1.0
            translation_second[:, 1, 1] = 1.0
            translation_second[:, 2] = second.float()
            enough_second = inlier_second.sum(dim=-1) >= 3
            affine_second = torch.where(
                enough_second[:, None, None], fit(inlier_second), translation_second,
            )
            predicted_second = torch.einsum("naj,njk->nak", anchor_h[None], affine_second)
            refined_second = inlier_second & (
                (predicted_second - match_xy).norm(dim=-1) <= 1.5
            )
            affine_second = torch.where(
                (refined_second.sum(dim=-1) >= 3)[:, None, None],
                fit(refined_second), affine_second,
            )

            warped_second = torch.einsum("pj,njk->npk", full_h, affine_second)
            sx, sy = warped_second.unbind(dim=-1)
            valid_second = (
                (sx >= 0.0) & (sx <= float(width - 1))
                & (sy >= 0.0) & (sy <= float(height - 1))
            )
            sx0 = sx.floor().long().clamp(0, width - 2)
            sy0 = sy.floor().long().clamp(0, height - 2)
            sfx = (sx - sx0.float()).clamp(0.0, 1.0)
            sfy = (sy - sy0.float()).clamp(0.0, 1.0)
            second_parents = torch.stack((
                sy0 * width + sx0,
                sy0 * width + (sx0 + 1),
                (sy0 + 1) * width + sx0,
                (sy0 + 1) * width + (sx0 + 1),
            ), dim=-1)
            second_weights = torch.stack((
                (1.0 - sfx) * (1.0 - sfy), sfx * (1.0 - sfy),
                (1.0 - sfx) * sfy, sfx * sfy,
            ), dim=-1).float()

            # Mode choice uses a channel-strided probe of the same bilinear
            # blend; the chosen mode then supplies the full-width weights.
            stride = _MODE_PROBE_STRIDE
            probe_source = source[..., ::stride]
            probe_previous = previous[..., ::stride]

            def probe_score(par: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
                index = par.reshape(count, patches * 4)
                blend_neighbour = probe_previous.gather(
                    1, index[..., None].expand(-1, -1, probe_previous.shape[-1]),
                ).reshape(count, patches, 4, probe_previous.shape[-1])
                blend = (blend_neighbour * weight[..., None]).sum(dim=2)
                return F.cosine_similarity(probe_source, blend, dim=-1, eps=1e-6)

            take_second = probe_score(second_parents, second_weights) > probe_score(
                local_parents, local_weights,
            )
            take_second = take_second & has_second[:, None] & valid_second
            local_parents = torch.where(take_second[..., None], second_parents, local_parents)
            local_weights = torch.where(take_second[..., None], second_weights, local_weights)
            valid = torch.where(take_second, valid_second, valid)

        if _LOCAL_SEARCH:
            # Local correspondence search: keep the affine prediction as the
            # prior centre, then pick the best sub-patch offset by the same
            # bilinear blend.  The merge object (four neighbours) and the
            # restoration weights are unchanged; only the neighbourhood choice
            # is optimised, which recovers candidates the fixed warp misses.
            stride = _MODE_PROBE_STRIDE
            probe_source = source[..., ::stride]
            probe_previous = previous[..., ::stride]
            offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (1, -1), (-1, 1), (1, 1))
            parents_stack = []
            weights_stack = []
            scores_stack = []
            for oy, ox in offsets:
                sx = x + float(ox)
                sy = y + float(oy)
                inside = (
                    (sx >= 0.0) & (sx <= float(width - 1))
                    & (sy >= 0.0) & (sy <= float(height - 1))
                )
                cx0 = sx.floor().long().clamp(0, width - 2)
                cy0 = sy.floor().long().clamp(0, height - 2)
                cfx = (sx - cx0.float()).clamp(0.0, 1.0)
                cfy = (sy - cy0.float()).clamp(0.0, 1.0)
                parents_stack.append(torch.stack((
                    cy0 * width + cx0,
                    cy0 * width + cx0 + 1,
                    (cy0 + 1) * width + cx0,
                    (cy0 + 1) * width + cx0 + 1,
                ), dim=-1))
                weights_stack.append(torch.stack((
                    (1.0 - cfx) * (1.0 - cfy), cfx * (1.0 - cfy),
                    (1.0 - cfx) * cfy, cfx * cfy,
                ), dim=-1).float())
                index = parents_stack[-1].reshape(count, patches * 4)
                neighbour = probe_previous.gather(
                    1, index[..., None].expand(-1, -1, probe_previous.shape[-1]),
                ).reshape(count, patches, 4, probe_previous.shape[-1])
                blend = (neighbour * weights_stack[-1][..., None]).sum(dim=2)
                scores_stack.append(
                    F.cosine_similarity(probe_source, blend, dim=-1, eps=1e-6)
                    .masked_fill(~inside, -2.0)
                )
            pick = torch.stack(scores_stack, dim=0).argmax(dim=0)
            parent_all = torch.stack(parents_stack, dim=0)
            weight_all = torch.stack(weights_stack, dim=0)
            gather_index = pick[None, :, :, None].expand(1, count, patches, 4)
            local_parents = parent_all.gather(0, gather_index).squeeze(0)
            local_weights = weight_all.gather(0, gather_index).squeeze(0)
            valid = valid | (
                torch.stack(scores_stack, dim=0).amax(dim=0) > -1.5
            )

        gather_index = local_parents.reshape(count, patches * 4)
        neighbours = previous.gather(
            1, gather_index[..., None].expand(-1, -1, previous.shape[-1]),
        ).reshape(count, patches, 4, previous.shape[-1])
        virtual = (neighbours * local_weights[..., None]).sum(dim=2)
        agreement = F.cosine_similarity(source, virtual, dim=-1, eps=1e-6)

        selected = valid & (agreement >= similarity_threshold)

        frame_ids = torch.arange(start + 1, stop + 1, device=device)
        rows = (
            frame_ids[:, None] * tokens_per_frame + patch_token_start + patch_index[None]
        ).reshape(-1)
        offset = (frame_ids - 1) * tokens_per_frame + patch_token_start
        merged[0, rows] = selected.reshape(-1)
        parents[0, rows] = (local_parents + offset[:, None, None]).reshape(-1, 4)
        weights_all[0, rows] = local_weights.reshape(-1, 4)

        valid_acc = valid_acc + valid.sum()
        inlier_acc = inlier_acc + inlier.sum()
        anchor_total += anchor_local.numel() * count

    # Diagnostic ratios are the only reason this planner ever synchronises the
    # device.  Each ``.item()`` drains the pipeline, so the plain benchmark path
    # (FAST_AFFINE_STATS=0) skips them; the merge *decisions* are unaffected.
    def stat(value: torch.Tensor) -> float:
        return float(value.item()) if _STATS else 0.0

    valid_total = int(stat(valid_acc))
    anchor_inlier_total = int(stat(inlier_acc))

    candidate_merge_ratio = 0.0
    candidate_importance_fraction = 0.0
    removed_importance_fraction = 0.0

    if star_window:
        # Build bounded affine trajectory stars. Every window starts from a
        # full physical anchor frame; later patches follow the dominant corner
        # of the adjacent-frame bilinear map back to that anchor. Unlike the
        # recursive soft quotient, each attention token has one discrete Q/K/V
        # identity, while the geometry (and thus cluster membership) still
        # comes entirely from the fitted affine transport.
        root = torch.arange(total, device=device)
        star_merged = merged.clone()
        all_unit = F.normalize(tokens[0].float(), dim=-1, eps=1e-6)
        qkv_unit = None
        if qkv_similarity_threshold is not None:
            if qkv_features is None or qkv_features.shape[-1] % 3:
                raise ValueError("affine-star Q/K/V gating requires projected Q/K/V features")
            qkv_dim = qkv_features.shape[-1] // 3
            qkv_unit = F.normalize(
                qkv_features[0].float().reshape(total, 3, qkv_dim), dim=-1, eps=1e-6,
            )
        composed = torch.eye(3, device=device, dtype=torch.float32)
        current_anchor = 0
        for frame in range(1, frames):
            begin = frame * tokens_per_frame + patch_token_start
            end = begin + patches
            rows = torch.arange(begin, end, device=device)
            if frame % star_window == 0:
                star_merged[0, begin:end] = False
                composed = torch.eye(3, device=device, dtype=torch.float32)
                current_anchor = frame
                continue
            step = torch.eye(3, device=device, dtype=torch.float32)
            step[:, :2] = pair_affine[frame - 1]
            composed = step @ composed
            anchor_frame = current_anchor
            warped_anchor = full_h @ composed[:, :2]
            ax, ay = warped_anchor.unbind(dim=-1)
            valid_anchor = (
                (ax >= 0.0) & (ax <= float(width - 1))
                & (ay >= 0.0) & (ay <= float(height - 1))
            )
            ax0 = ax.floor().long().clamp(0, width - 2)
            ay0 = ay.floor().long().clamp(0, height - 2)
            anchor_local = torch.stack((
                ay0 * width + ax0,
                ay0 * width + ax0 + 1,
                (ay0 + 1) * width + ax0,
                (ay0 + 1) * width + ax0 + 1,
            ), dim=-1)
            anchor_original = (
                anchor_frame * tokens_per_frame + patch_token_start + anchor_local
            )
            fx = (ax - ax0.float()).clamp(0.0, 1.0)
            fy = (ay - ay0.float()).clamp(0.0, 1.0)
            local_weights = torch.stack((
                (1.0 - fx) * (1.0 - fy), fx * (1.0 - fy),
                (1.0 - fx) * fy, fx * fy,
            ), dim=-1).float()
            # Certification and restoration share the same four bilinear
            # weights: the source must be predictable from the anchor-frame
            # blend, and that same blend restores its output.
            blend = (all_unit[anchor_original] * local_weights[..., None]).sum(dim=1)
            score = (all_unit[begin:end] * blend).sum(dim=-1)
            parents[0, begin:end] = anchor_original
            weights_all[0, begin:end] = local_weights
            certified = valid_anchor & (score >= similarity_threshold)
            if qkv_unit is not None:
                virtual_qkv = (
                    qkv_unit[anchor_original] * local_weights[:, :, None, None]
                ).sum(dim=1)
                qkv_agreement = F.cosine_similarity(
                    qkv_unit[begin:end], virtual_qkv, dim=-1, eps=1e-6,
                ).amin(dim=-1)
                certified &= qkv_agreement >= qkv_similarity_threshold
            star_merged[0, begin:end] &= certified

        candidate_merge_ratio = stat(star_merged.float().mean())
        if target_merge_ratio is not None:
            if token_importance is None or token_importance.numel() != total:
                raise ValueError("token_importance must contain one score per original token")
            candidates = torch.where(star_merged[0])[0]
            total_importance = token_importance.float().clamp_min(0.0).sum().clamp_min(1e-12)
            candidate_importance_fraction = stat(
                token_importance[candidates].float().clamp_min(0.0).sum().div(total_importance)
            )
            target_count = min(candidates.numel(), int(round(target_merge_ratio * total)))
            if target_count < candidates.numel():
                chosen = torch.topk(
                    token_importance[candidates], target_count,
                    largest=False, sorted=False,
                ).indices
                selected = candidates[chosen]
                star_merged.zero_()
                star_merged[0, selected] = True
            removed = torch.where(star_merged[0])[0]
            removed_importance_fraction = stat(
                token_importance[removed].float().clamp_min(0.0).sum().div(total_importance)
            )

        # Every parent is an immutable physical anchor token, so the shared soft
        # plan construction below applies unchanged: the same bilinear weights
        # certify redundancy and restore the block output in a single step.
        merged = star_merged

    keep = identity[~merged].view(1, -1)
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=device).view(1, -1))
    source_original = identity[merged].view(1, -1)
    bilinear_weights = weights_all[:, source_original[0]]
    source_parents = parents[:, source_original[0]]
    if assignment_mode == "hard_hard" and source_original.shape[1]:
        # FastVGGT-equivalent hard assignment: every source is attached to the
        # single nearest (largest bilinear coefficient) physical ancestor, and
        # both Q/K/V pooling and the output restoration use that 0/1 map.
        # Pointer jumping resolves a T-frame chain in ceil(log2 T) gathers.
        dominant_parent = parents[0].gather(
            1, weights_all[0].argmax(dim=-1, keepdim=True),
        ).squeeze(1)
        root = torch.where(
            merged[0], dominant_parent, torch.arange(total, device=device),
        )
        for _ in range(max(1, (max(frames, 2) - 1).bit_length())):
            root = root[root]
        hard_parent = root[source_original[0]]
        source_parent_compact = (
            original_to_compact[0, hard_parent].view(1, -1, 1).expand(-1, -1, 4)
        )
        if bool((source_parent_compact < 0).any()):
            raise RuntimeError("affine-warp hard source did not resolve to a physical representative")
        hard_weights = torch.zeros(
            (1, source_original.shape[1], 4), device=device, dtype=weights_all.dtype,
        )
        hard_weights[..., 0] = 1.0
        source_pool_weights = hard_weights
        source_restore_weights = hard_weights
    else:
        source_parent_compact = original_to_compact.gather(
            1, source_parents.reshape(1, -1),
        ).reshape_as(source_parents)
        if bool((source_parent_compact < 0).any()):
            raise RuntimeError("affine-warp source has a non-physical parent")
        source_pool_weights = bilinear_weights
        source_restore_weights = bilinear_weights

    normalizer = torch.ones((1, keep.shape[1], 1), device=device, dtype=torch.float32)
    if pool_source_content and source_original.shape[1]:
        # Mean-pooling into representatives reproduces FastVGGT's mass-preserving
        # K/V merge; the divisor is the number of observations per key.
        for corner in range(4):
            normalizer.scatter_add_(
                1,
                source_parent_compact[:, :, corner, None],
                source_pool_weights[:, :, corner, None].float(),
            )

    special_query_original = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + torch.arange(patch_token_start, device=device)[None, :]
    ).reshape(-1)
    special_query_compact = original_to_compact[0, special_query_original]
    # Single device read for the remaining structural assertion.
    if bool((special_query_compact < 0).any()):
        raise RuntimeError("affine-warp plan unexpectedly removed a camera/register token")

    if _PROFILE:
        _ev1.record()
        _ev1.synchronize()
        _PROFILE_STATE["planner_ms"] += _ev0.elapsed_time(_ev1)
        _PROFILE_STATE["calls"] += 1

    return WeightedBilinearTokenPlan(
        keep=keep,
        source_original=source_original,
        source_parent_compact=source_parent_compact,
        source_pool_weights=source_pool_weights,
        source_restore_weights=source_restore_weights,
        original_tokens=total,
        normalizer=normalizer,
        preserve_queries=False,
        preserve_destination_keys=False,
        dense_special_queries=bool(dense_special_kv),
        special_query_original=special_query_original,
        special_query_compact=special_query_compact,
        pool_source_content=bool(pool_source_content),
        precomputed_qkv=qkv_features,
        metadata={
            "merge_ratio": stat(merged.float().mean()),
            "similarity_threshold": float(similarity_threshold),
            "anchor_grid": int(anchor_grid),
            "anchor_match_threshold": float(anchor_match_threshold),
            "assignment_mode": assignment_mode,
            "kv_only": False,
            "dense_special_kv": bool(dense_special_kv),
            "pool_source_content": bool(pool_source_content),
            "qkv_similarity_threshold": qkv_similarity_threshold,
            "star_window": int(star_window),
            "unified_qkv": True,
            "candidate_merge_ratio": candidate_merge_ratio,
            "target_merge_ratio": target_merge_ratio,
            "importance_policy": "external_token_importance" if target_merge_ratio is not None else "none",
            "candidate_importance_fraction": candidate_importance_fraction,
            "removed_importance_fraction": removed_importance_fraction,
            "valid_patch_ratio": valid_total / max(pair_count * patches, 1),
            "anchor_inlier_ratio": anchor_inlier_total / max(anchor_total, 1),
            "planner": "bounded_affine_star_bilinear_qkv_v2",
        },
    )


def _rope_halves(out: torch.Tensor, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> None:
    """``out = x*cos + rotate_half(x)*sin`` written in place, no temporaries.

    ``rotate_half`` is a cyclic shift by half a head, so the two halves are
    independent and can be fused into two multiply-adds.  The reference builds
    ``cat([-x2, x1])`` and keeps four full-size intermediates alive.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    o1, o2 = out[..., :half], out[..., half:]
    torch.mul(x1, cos[..., :half], out=o1)
    o1.addcmul_(x2, sin[..., :half], value=-1.0)
    torch.mul(x2, cos[..., half:], out=o2)
    o2.addcmul_(x1, sin[..., half:], value=1.0)


def _apply_rope_fast(self, q: torch.Tensor, k: torch.Tensor, rope):
    """Same rotation as ``SelfAttention.apply_rope`` with one allocation each."""
    sin, cos = rope
    dtype = q.dtype if _ROPE_DTYPE is None else _ROPE_DTYPE
    if sin.dtype != dtype:
        sin = sin.to(dtype)
        cos = cos.to(dtype)
    suffix = sin.shape[-2]
    prefix = q.shape[-2] - suffix
    q_out = torch.empty(q.shape, dtype=dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=dtype, device=k.device)
    if prefix:
        q_out[:, :, :prefix] = q[:, :, :prefix]
        k_out[:, :, :prefix] = k[:, :, :prefix]
    _rope_halves(q_out[:, :, prefix:], q[:, :, prefix:], sin, cos)
    _rope_halves(k_out[:, :, prefix:], k[:, :, prefix:], sin, cos)
    if dtype != q.dtype:
        q_out = q_out.to(q.dtype)
        k_out = k_out.to(k.dtype)
    return q_out, k_out


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attn_mask=None):
    """SDPA with an optional reduced-precision compute window."""
    if _ATTN_DTYPE is None or q.dtype == _ATTN_DTYPE:
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    out = torch.nn.functional.scaled_dot_product_attention(
        q.to(_ATTN_DTYPE), k.to(_ATTN_DTYPE), v.to(_ATTN_DTYPE),
        attn_mask=None if attn_mask is None else attn_mask.to(_ATTN_DTYPE),
    )
    return out.to(q.dtype)


def _compute_attention_fast(self, qkv, attn_bias=None, rope=None, token_plan=None):
    """Attention with one shared compact token axis for Q, K and V."""
    assert attn_bias is None
    B, N, _ = qkv.shape
    C = self.qkv.in_features

    qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
    q, k, v = torch.unbind(qkv, 2)
    q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
    if self.use_qk_norm:
        q = self.q_norm(q)
        k = self.k_norm(k)
    if rope is not None:
        q, k = self.apply_rope(q, k, rope)

    if token_plan is not None:
        dense_q, dense_k, dense_v = q, k, v

        def pack(value: torch.Tensor) -> torch.Tensor:
            value = token_plan.compact(
                value.transpose(1, 2).reshape(B, N, C), merge=token_plan.pool_sources,
            )
            return value.reshape(
                B, token_plan.kept_tokens, self.num_heads, C // self.num_heads,
            ).transpose(1, 2)

        if getattr(token_plan, "preserve_queries", False):
            key = k.transpose(1, 2).reshape(B, N, C)
            if hasattr(token_plan, "compact_key"):
                key = token_plan.compact_key(key)
            else:
                key = token_plan.compact(key, merge=token_plan.pool_sources)
            k = key.reshape(
                B, token_plan.kept_tokens, self.num_heads, C // self.num_heads,
            ).transpose(1, 2)
            v = pack(v)
        elif getattr(token_plan, "query_keep", None) is not None:
            # Asymmetric quotient: K/V recurse through long chains, but a
            # dropped query is rebuilt from four physical predecessor queries.
            query = token_plan.compact_query(q.transpose(1, 2).reshape(B, N, C))
            q = query.reshape(
                B, token_plan.query_kept_tokens, self.num_heads, C // self.num_heads,
            ).transpose(1, 2)
            k, v = pack(k), pack(v)
        else:
            # Q/K/V share exactly the same quotient map.  Compacting their
            # concatenated channels performs the four weighted scatter-adds
            # once instead of three times, without changing any arithmetic.
            flat_qkv = torch.cat(
                tuple(t.transpose(1, 2).reshape(B, N, C) for t in (q, k, v)),
                dim=-1,
            )
            flat_qkv = token_plan.compact(
                flat_qkv, merge=token_plan.pool_sources,
            )
            q_flat, k_flat, v_flat = flat_qkv.split(C, dim=-1)

            def unflatten(value: torch.Tensor) -> torch.Tensor:
                return value.reshape(
                    B, token_plan.kept_tokens, self.num_heads,
                    C // self.num_heads,
                ).transpose(1, 2)

            q, k, v = map(unflatten, (q_flat, k_flat, v_flat))

    compact_bias = None
    if token_plan is not None and getattr(token_plan, "key_log_mass", None) is not None:
        compact_bias = token_plan.key_log_mass[:, None, None, :].to(device=q.device, dtype=q.dtype)
    x = _sdpa(q, k, v, attn_mask=compact_bias)

    if token_plan is not None and getattr(token_plan, "dense_special_queries", False):
        original = token_plan.special_query_original
        output_indices = (
            original if getattr(token_plan, "preserve_queries", False)
            else token_plan.special_query_compact
        )
        exact_special = _sdpa(
            dense_q.index_select(2, original), dense_k, dense_v,
        )
        x.index_copy_(2, output_indices, exact_special)

    x = x.transpose(1, 2)
    output_tokens = (
        N if token_plan is not None and getattr(token_plan, "preserve_queries", False)
        else token_plan.query_kept_tokens if token_plan is not None and hasattr(token_plan, "query_kept_tokens")
        else token_plan.kept_tokens if token_plan is not None else N
    )
    return x.reshape([B, output_tokens, C])


def install() -> None:
    """Activate the fast branch for this process only."""
    import vggt_omega.models.aggregator as aggregator
    from vggt_omega.models.layers.attention import SelfAttention

    aggregator.build_affine_warp_plan = build_affine_warp_plan_fast
    SelfAttention.compute_attention = _compute_attention_fast
    SelfAttention.apply_rope = _apply_rope_fast
