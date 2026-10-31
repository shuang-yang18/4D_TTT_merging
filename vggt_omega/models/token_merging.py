"""Token merging primitives for VGGT-Omega.

The two encoders have different token layouts: Omega stores camera/register
tokens at the beginning of *each* frame, whereas Any4D stores image tokens
contiguously. FastVGGT and class-local Q/K/V compaction originate in Any4D.
The trajectory/visibility labeler is Omega-specific, not Any4D-equivalent.
Tracks determine classes; they do NOT define the similarity-merge groups.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import os

import torch
import torch.nn.functional as F

# The dedicated environment ships Triton in-tree CUDA backends but no package
# entry-point metadata.  Ask Triton to discover those bundled backends before
# importing it; installations with normal entry points are unaffected.
os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
try:  # Optional legacy similarity kernel, not the active trajectory labeler.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on the target environment
    triton = None
    tl = None


FAST_MATCH_SOURCE_CHUNK = 5000
TEMPORAL_MATCH_SOURCE_CHUNK = 4096
TEMPORAL_MATCH_DESTINATION_CHUNK = 16384
LABEL_FRAME_CHUNK = 4
TEMPORAL_LABEL_ALGORITHM = "native_patch_trajectory_v2"
ADAPTIVE_TEMPORAL_LABEL_ALGORITHM = "native_adaptive_boundary_v4"
HYBRID_TEMPORAL_LABEL_ALGORITHM = "native_v4_rgb_local_consensus_v2"
PREDICTABILITY_TOPOLOGY_ALGORITHM = "postglobal_predictability_topology_v1"


def deterministic_weighted_normalizer(
    parent_compact: torch.Tensor,
    weights: torch.Tensor,
    kept_tokens: int,
) -> torch.Tensor:
    """Return physical-plus-source mass without CUDA atomic additions."""
    if parent_compact.shape != weights.shape or parent_compact.shape[0] != 1:
        raise ValueError("parent ids and weights must have matching [1,M,S] shapes")
    mass = torch.ones(
        (1, kept_tokens, 1), device=weights.device, dtype=torch.float32,
    )
    for corner in range(parent_compact.shape[-1]):
        destination = parent_compact[0, :, corner]
        order = destination.argsort(stable=True)
        lengths = torch.bincount(destination, minlength=kept_tokens).view(1, -1)
        ordered_weight = weights[0, :, corner].index_select(0, order).float().view(1, -1, 1)
        mass = mass + torch.segment_reduce(
            ordered_weight, reduce="sum", lengths=lengths, axis=1,
        )
    return mass


@dataclass(frozen=True)
class TemporalLabelConfig:
    """Training-free token-trajectory thresholds, in cosine / patch-grid units.

    Absolute gates prevent a globally bad frame from defining its own low
    confidence as reliable. Relative gates may tighten, never relax, them.
    Radius/quantisation intentionally remain the original 5x5 integer search.
    """

    min_score: float = 0.60
    min_margin: float = 0.01
    max_cycle_error: float = 1.0
    min_boundary_jump: float = 0.50
    max_boundary_threshold: float = 0.75
    feature_boundary_mad_scale: float = 1.5
    max_interior_motion_jump: float = 0.50
    temporal_boundary_support: bool = True
    # Native patch-token branch.  These are fixed structural ratios, selected
    # offline but independent of any runtime pseudo-label/reference input.
    native_sensitive_top_ratio: float = 0.15
    native_sensitive_confidence_weight: float = 0.20
    native_unknown_mad_scale: float = 7.0
    # Experimental adaptive-native branch. The score uses native appearance
    # edges plus predecessor uncertainty; its threshold is derived from the
    # entire clip's score distribution, never from a token quota. A shared
    # clip threshold preserves the absolute difference between a close, large
    # moving object and a small/distant one, which per-frame ranking erases.
    adaptive_confidence_weight: float = 0.20
    # ``strong`` is a semantically plausible high-contrast contour on its
    # own.  The lower ``weak`` threshold is admitted only when it disagrees
    # with the clip's dominant (camera) motion.  This avoids turning static
    # monitor/desk contours into sensitive tokens merely because they are
    # visually sharp.
    adaptive_strong_mad_scale: float = 1.50
    adaptive_weak_mad_scale: float = 1.25
    adaptive_motion_mad_scale: float = -0.25
    # A true token-scale contour has local support on both sides of a grid
    # edge.  Requiring four candidates in a 3x3 footprint removes isolated
    # high-response speckles without prescribing a sensitive-token quota.
    adaptive_min_spatial_support: int = 4
    adaptive_temporal_boundary_support: bool = True
    adaptive_min_confidence_range: float = 1e-5
    # Predictability-topology branch.  These are fixed robust-tail gates, not
    # pseudo-label-calibrated quotas.  The branch is deliberately conservative:
    # a sharp static texture alone cannot become a temporal boundary.
    topology_unknown_mad_scale: float = 2.0
    topology_boundary_mad_scale: float = 1.0
    topology_transition_mad_scale: float = 0.0
    topology_min_transition: float = 0.01
    topology_min_spatial_support: int = 3
    topology_temporal_support_radius: int = 1
    # Segment-level rigid-scene gate.  The values operate on normalised patch
    # coordinates; a static sequence may contain isolated poor correspondences
    # but should not sustain this residual on most adjacent-frame pairs.
    topology_clip_epipolar_pair_median: float = 0.004
    topology_clip_nonrigid_min_fraction: float = 0.40


TEMPORAL_LABEL_CONFIG = TemporalLabelConfig()


def resolve_temporal_label_backend(backend: str) -> str:
    # Old launchers used an algorithm name as a backend. Preserve that alias
    # explicitly, but never claim the legacy 3x3 Triton kernel runs this method.
    if backend in {"reference", "trajectory"}:
        return "reference"
    if backend in {"deep_track_graph", "deep"}:
        return "deep_track_graph"
    if backend in {"adaptive_native", "adaptive"}:
        return "adaptive_native"
    if backend in {"predictability_topology", "topology", "ptopology"}:
        return "predictability_topology"
    raise ValueError(
        f"Unsupported trajectory-label backend {backend!r}: use 'reference' "
        "(legacy visibility labels), 'deep_track_graph' (fixed-quota native "
        "contours), 'adaptive_native' (motion-consistent native boundaries), or "
        "'predictability_topology' (post-global temporal-state topology); the old Triton similarity kernel does not "
        "implement either full labeler."
    )


@dataclass
class TokenPlan:
    """A compact-token map for one attention operation.

    ``pool_sources`` selects the Q/K/V reduction semantics.  The released
    FastVGGT plan is symmetric and therefore mean-pools every source into its
    destination.  Reference-anchor attachment is deliberately different: a
    source is removed *towards* an immutable reference destination, while the
    destination's Q/K/V stay unchanged.  Both variants expand the compact
    attention update through ``inverse`` afterwards.
    """

    keep: torch.Tensor                 # [B, K], original token indices
    inverse: torch.Tensor              # [B, N], original -> compact index
    original_tokens: int
    pool_sources: bool = True
    # Optional per-original-token contribution mask.  It permits one plan to
    # combine immutable directed attachments (False for their sources) with
    # ordinary symmetric cluster pooling (True for all cluster members).
    pool_mask: torch.Tensor | None = None
    metadata: dict[str, float | int] = field(default_factory=dict)
    # Reuse the projection used by an online planner in the actual attention
    # call, so model-aware selection does not pay for Q/K/V twice.
    precomputed_qkv: torch.Tensor | None = None

    @property
    def kept_tokens(self) -> int:
        return self.keep.shape[1]

    @property
    def keep_ratio(self) -> float:
        return self.kept_tokens / self.original_tokens

    def compact(self, value: torch.Tensor, *, merge: bool = True) -> torch.Tensor:
        if self.kept_tokens == self.original_tokens:
            return value
        if not merge and self.pool_mask is None:
            return value.gather(1, self.keep[..., None].expand(-1, -1, value.shape[-1]))
        accumulation_dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
        contribution = (
            torch.ones_like(self.inverse, dtype=torch.bool)
            if self.pool_mask is None else self.pool_mask
        )
        sums = torch.zeros(value.shape[0], self.kept_tokens, value.shape[-1], device=value.device, dtype=accumulation_dtype)
        sums.scatter_add_(1, self.inverse[..., None].expand_as(value),
                          value.to(accumulation_dtype) * contribution[..., None].to(accumulation_dtype))
        counts = torch.zeros(value.shape[0], self.kept_tokens, 1, device=value.device, dtype=accumulation_dtype)
        counts.scatter_add_(1, self.inverse[..., None], contribution[..., None].to(accumulation_dtype))
        return (sums / counts).to(value.dtype)

    def restore(self, value: torch.Tensor) -> torch.Tensor:
        if self.kept_tokens == self.original_tokens:
            return value
        return value.gather(1, self.inverse[..., None].expand(-1, -1, value.shape[-1]))


@dataclass
class RecursiveLocalTokenPlan:
    """Inference-only adjacent-frame 2x2 virtual-token plan.

    A removed patch has four direct parents in the preceding frame.  Restore
    proceeds in temporal order, so a parent may itself be virtual without
    expanding the chain to a first-frame support set.  Q/K/V are formed only
    for physical tokens; a removed token receives the convex interpolation of
    its parents' post-attention update.
    """

    keep: torch.Tensor                 # [1,K] original physical-token ids
    parent_original: torch.Tensor      # [1,N,4], direct predecessor ids
    weights: torch.Tensor              # [1,N,4], convex weights
    residual: torch.Tensor             # [1,N,D], true token minus virtual predecessor prediction
    merged: torch.Tensor               # [1,N]
    original_tokens: int
    frames: int
    tokens_per_frame: int
    pool_sources: bool = False
    metadata: dict[str, float | int] = field(default_factory=dict)

    @property
    def kept_tokens(self) -> int:
        return self.keep.shape[1]

    @property
    def keep_ratio(self) -> float:
        return self.kept_tokens / self.original_tokens

    def compact(self, value: torch.Tensor, *, merge: bool = True) -> torch.Tensor:
        del merge
        return value.gather(1, self.keep[..., None].expand(-1, -1, value.shape[-1]))

    def restore(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] != 1:
            raise RuntimeError("recursive local compaction currently supports inference batch size one")
        output = value.new_zeros((1, self.original_tokens, value.shape[-1]))
        kept_mask = ~self.merged
        output[:, self.keep[0]] = value
        # A frame's parents always belong to its immediately preceding frame,
        # which has already been materialised.  This is the recursive DAG
        # evaluation; no root-support expansion occurs.
        for frame in range(1, self.frames):
            begin = frame * self.tokens_per_frame
            end = begin + self.tokens_per_frame
            local_merged = self.merged[:, begin:end]
            parent = self.parent_original[:, begin:end]
            parent_values = output[:, parent[0]]
            virtual = (parent_values * self.weights[:, begin:end, :, None].to(value.dtype)).sum(dim=2)
            output[:, begin:end] = torch.where(local_merged[..., None], virtual, output[:, begin:end])
        return output


@dataclass
class WeightedBilinearTokenPlan:
    """Compact attention plan with four-way, information-preserving pooling.

    Each removed source contributes its Q/K/V to the four *physical* tokens
    surrounding its bilinear predecessor coordinate.  The same coefficients
    restore its post-attention update.  This is the weighted analogue of
    FastVGGT's source-to-destination pooling: a removed token is not simply
    deleted from the K/V memory.

    Parents are required to be physical kept tokens.  This intentionally
    forbids recursive virtual parents, because a chain of post-attention
    interpolation has no equivalent compact Q/K/V representation.
    """

    keep: torch.Tensor                         # [1,K] original physical ids
    source_original: torch.Tensor              # [1,M] removed source ids
    source_parent_compact: torch.Tensor        # [1,M,4] physical compact ids
    source_pool_weights: torch.Tensor          # [1,M,4], Q/K/V write weights
    source_restore_weights: torch.Tensor       # [1,M,4], output restore weights
    original_tokens: int
    normalizer: torch.Tensor                   # [1,K,1], physical + source mass
    pool_sources: bool = True
    # KV-only compression keeps every query and therefore needs no output
    # reconstruction.  Keys remain the physical destination keys; only values
    # collect source content.  ``key_log_mass`` below optionally preserves the
    # softmax evidence mass represented by each compact destination.
    preserve_queries: bool = False
    preserve_destination_keys: bool = False
    # Optional accuracy-preserving branch: camera/register queries read the
    # uncompressed K/V bank.  They are few, but they directly feed the camera
    # head and are disproportionately sensitive to K/V quotient errors.
    dense_special_queries: bool = False
    special_query_original: torch.Tensor | None = None
    special_query_compact: torch.Tensor | None = None
    # A linearly predictable source is redundant evidence.  This switch lets
    # us test dropping that evidence without also averaging it into and
    # thereby perturbing the physical parent K/V vectors.
    pool_source_content: bool = True
    # Optional QKV computed by the caller for an online consistency gate.
    # Reusing it in SelfAttention avoids paying the projection twice.
    precomputed_qkv: torch.Tensor | None = None
    # Residual-corrected transport: the four-parent output remains the base
    # prediction, but a removed source may evaluate its original query against
    # compact K/V and replace only the nonlinear query-innovation residual.
    source_query_correction: bool = False
    source_attention_output: torch.Tensor | None = None
    # Additive per-key attention bias in compact coordinates.  log(mass)
    # preserves the softmax probability mass of recursively represented
    # duplicate observations.
    key_log_mass: torch.Tensor | None = None
    # Optional asymmetric query quotient. K/V may recurse through long
    # trajectories, while a removed query is restricted to one bilinear step
    # whose four predecessor queries remain physical.
    query_keep: torch.Tensor | None = None
    query_source_original: torch.Tensor | None = None
    query_source_parent_compact: torch.Tensor | None = None
    query_restore_weights: torch.Tensor | None = None
    # Preserve dense attention queries, but evaluate the token-wise MLP only
    # on a one-step affine quotient and restore its update bilinearly.
    compact_mlp: bool = False
    metadata: dict[str, float | int] = field(default_factory=dict)
    # Stable destination groupings used by every Q/K/V compaction. CUDA
    # scatter_add_ uses atomic floating-point additions whose arrival order is
    # unspecified; tiny run-to-run changes then alter threshold decisions in
    # later layers. Sorting once per plan lets all subsequent reductions use a
    # fixed source order without paying the sort cost for Q, K, and V.
    _pool_orders: tuple[torch.Tensor, ...] = field(init=False, repr=False)
    _pool_lengths: tuple[torch.Tensor, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.source_parent_compact.shape[0] != 1:
            raise ValueError("weighted bilinear plans require one shared inference mapping")
        orders: list[torch.Tensor] = []
        lengths: list[torch.Tensor] = []
        for corner in range(self.source_parent_compact.shape[-1]):
            destination = self.source_parent_compact[0, :, corner]
            orders.append(destination.argsort(stable=True))
            lengths.append(torch.bincount(destination, minlength=self.kept_tokens))
        self._pool_orders = tuple(orders)
        self._pool_lengths = tuple(lengths)

    @property
    def kept_tokens(self) -> int:
        return self.keep.shape[1]

    @property
    def keep_ratio(self) -> float:
        return self.kept_tokens / self.original_tokens

    def compact(self, value: torch.Tensor, *, merge: bool = True) -> torch.Tensor:
        """Pool source Q/K/V mass into its four retained bilinear parents."""
        del merge
        compact = value.gather(1, self.keep[..., None].expand(-1, -1, value.shape[-1]))
        if self.source_original.shape[1] == 0 or not self.pool_source_content:
            return compact
        accumulation_dtype = torch.float32 if value.dtype in (torch.float16, torch.bfloat16) else value.dtype
        compact = compact.to(accumulation_dtype)
        source = value.gather(
            1, self.source_original[..., None].expand(-1, -1, value.shape[-1]),
        ).to(accumulation_dtype)
        # A stable segmented sum is deterministic on CUDA. Processing one
        # support corner at a time retains the old O(M*C) peak workspace rather
        # than materialising an [M,S,C] tensor at long-video resolution.
        for corner in range(self.source_parent_compact.shape[-1]):
            order = self._pool_orders[corner]
            lengths = self._pool_lengths[corner].view(1, -1).expand(value.shape[0], -1)
            ordered_source = source.index_select(1, order)
            ordered_weight = self.source_pool_weights[0, :, corner].index_select(
                0, order,
            ).view(1, -1, 1).to(accumulation_dtype)
            compact = compact + torch.segment_reduce(
                ordered_source * ordered_weight,
                reduce="sum",
                lengths=lengths,
                axis=1,
            )
        return (compact / self.normalizer.to(accumulation_dtype)).to(value.dtype)

    def compact_key(self, value: torch.Tensor) -> torch.Tensor:
        """Keep physical keys unchanged for the Q-preserving KV-only path."""
        if not self.preserve_destination_keys:
            return self.compact(value)
        return value.gather(1, self.keep[..., None].expand(-1, -1, value.shape[-1]))

    @property
    def query_kept_tokens(self) -> int:
        return self.kept_tokens if self.query_keep is None else self.query_keep.shape[1]

    def compact_query(self, value: torch.Tensor) -> torch.Tensor:
        if self.query_keep is None:
            return self.compact(value)
        return value.gather(1, self.query_keep[..., None].expand(-1, -1, value.shape[-1]))

    def restore(self, value: torch.Tensor) -> torch.Tensor:
        """Broadcast physical outputs and bilinearly recover source outputs."""
        if self.preserve_queries and value.shape[1] == self.original_tokens:
            return value
        if self.query_keep is not None:
            output = value.new_zeros((1, self.original_tokens, value.shape[-1]))
            output[:, self.query_keep[0]] = value
            if self.query_source_original is not None and self.query_source_original.shape[1]:
                parent = value[:, self.query_source_parent_compact[0]]
                restored = (parent * self.query_restore_weights[..., None].to(value.dtype)).sum(dim=2)
                output[:, self.query_source_original[0]] = restored.to(value.dtype)
            return output
        if self.preserve_queries:
            if value.shape[1] != self.original_tokens:
                raise RuntimeError("Q-preserving KV compaction must produce one output per original query")
            return value
        if value.shape[0] != 1:
            raise RuntimeError("weighted bilinear compaction currently supports inference batch size one")
        output = value.new_zeros((1, self.original_tokens, value.shape[-1]))
        output[:, self.keep[0]] = value
        if self.source_original.shape[1]:
            parent = value[:, self.source_parent_compact[0]]
            restored = (parent * self.source_restore_weights[..., None].to(value.dtype)).sum(dim=2)
            # Index assignment is not autocast-aware: under the model's BF16
            # autocast context the multiply/reduction may be promoted to FP32.
            # The compact attention output defines the contract here.
            output[:, self.source_original[0]] = restored.to(dtype=value.dtype)
        return output

    def restore_query(self, value: torch.Tensor) -> torch.Tensor:
        """Restore a tensor compacted with :meth:`compact_query`."""
        if self.query_keep is None:
            return self.restore(value)
        output = value.new_zeros((value.shape[0], self.original_tokens, value.shape[-1]))
        output[:, self.query_keep[0]] = value
        if self.query_source_original is not None and self.query_source_original.shape[1]:
            parent = value[:, self.query_source_parent_compact[0]]
            restored = (parent * self.query_restore_weights[..., None].to(value.dtype)).sum(dim=2)
            output[:, self.query_source_original[0]] = restored.to(value.dtype)
        return output


def _simplex_project(values: torch.Tensor) -> torch.Tensor:
    """Vectorised Euclidean projection of [...,4] onto the probability simplex."""
    sorted_values, _ = values.sort(dim=-1, descending=True)
    cssv = sorted_values.cumsum(dim=-1) - 1.0
    ranks = torch.arange(1, values.shape[-1] + 1, device=values.device, dtype=values.dtype)
    active = sorted_values > cssv / ranks
    rho = active.sum(dim=-1).clamp_min(1) - 1
    theta = cssv.gather(-1, rho[..., None]).squeeze(-1) / (rho.to(values.dtype) + 1.0)
    return (values - theta[..., None]).clamp_min_(0.0)


@torch.no_grad()
def build_recursive_local_2x2_plan(
    tokens: torch.Tensor,
    *,
    frames: int,
    tokens_per_frame: int,
    patch_token_start: int,
    patch_hw: tuple[int, int],
    similarity_threshold: float = .95,
    dominant_corner_only: bool = False,
) -> RecursiveLocalTokenPlan:
    """Build a local, recursive 2x2 virtual-token plan in O(T·P·D).

    Only the four 2x2 windows adjacent to the same patch coordinate in the
    preceding frame are examined.  Each window obtains a one-shot FP32
    constrained least-squares fit; this replaces the diagnostic's all-to-all
    match and 24-step optimiser.  When ``dominant_corner_only`` is enabled,
    each successor frame retains only candidates whose largest interpolation
    coefficient lies at that frame's modal 2x2 corner.  This is deliberately
    a strict ablation: candidates with another local displacement remain
    physical rather than being forced into the dominant motion pattern.
    """
    if tokens.shape[0] != 1:
        raise ValueError("recursive local 2x2 merge currently supports batch size one")
    if not -1.0 <= similarity_threshold <= 1.000001:
        raise ValueError("similarity_threshold must lie in [-1, 1.000001]")
    batch, total, dimension = tokens.shape
    height, width = patch_hw
    patches = height * width
    if total != frames * tokens_per_frame or tokens_per_frame != patch_token_start + patches:
        raise ValueError("Invalid recursive-local frame/patch layout")

    identity = torch.arange(total, device=tokens.device).view(1, -1)
    patch_local = torch.arange(patches, device=tokens.device)
    windows = _local_2x2_windows(patch_local, height=height, width=width)  # [P,4,4]
    reshaped = tokens.view(batch, frames, tokens_per_frame, dimension)
    merged = torch.zeros((batch, total), dtype=torch.bool, device=tokens.device)
    parents = torch.zeros((batch, total, 4), dtype=torch.long, device=tokens.device)
    weights_all = torch.zeros((batch, total, 4), dtype=torch.float32, device=tokens.device)
    weights_all[..., 0] = 1.0
    threshold_eligible_total = 0
    dominant_corner_selected_total = 0

    # Build the predictive graph from the tail towards the reference frame.
    # Parent edges nevertheless point backwards in time, so restoration is
    # the reverse topological traversal: frame one through frame T.
    residual = torch.zeros_like(tokens, dtype=torch.float32)
    for frame in range(frames - 1, 0, -1):
        source = F.normalize(reshaped[:, frame, patch_token_start:].float(), dim=-1, eps=1e-6)
        previous = F.normalize(reshaped[:, frame - 1, patch_token_start:].float(), dim=-1, eps=1e-6)
        reference = previous[:, windows]  # [1,P,4 windows,4 patches,D]
        # This small numerical solve must stay FP32 even when the surrounding
        # encoder runs BF16 AMP.
        gram = torch.einsum("bpvkd,bpvld->bpvkl", reference, reference).float()
        rhs = torch.einsum("bpvkd,bpd->bpvk", reference, source).float()
        eye = torch.eye(4, device=tokens.device, dtype=torch.float32).view(1, 1, 1, 4, 4)
        unconstrained = torch.linalg.solve(gram + 1e-4 * eye, rhs[..., None]).squeeze(-1)
        weights = _simplex_project(unconstrained)
        reconstruction = torch.einsum("bpvk,bpvkd->bpvd", weights, reference)
        score = F.cosine_similarity(source[:, :, None], reconstruction, dim=-1, eps=1e-6)
        best_score, variant = score.max(dim=-1)
        chosen_weights = weights.gather(2, variant[..., None, None].expand(-1, -1, 1, 4)).squeeze(2)
        chosen_windows = windows[patch_local, variant[0]].unsqueeze(0)
        offset = (frame - 1) * tokens_per_frame + patch_token_start
        original = frame * tokens_per_frame + patch_token_start + patch_local
        parents[:, original] = chosen_windows + offset
        weights_all[:, original] = chosen_weights
        threshold_eligible = best_score >= similarity_threshold
        threshold_eligible_total += int(threshold_eligible.sum().item())
        if dominant_corner_only and threshold_eligible.any():
            dominant_corner = chosen_weights.argmax(dim=-1)
            # One dominant interpolation corner is selected independently
            # for every target frame.  ``bincount`` is deterministic for the
            # rare ties (lowest corner index wins through argmax).
            modal_corner = torch.bincount(
                dominant_corner[0, threshold_eligible[0]], minlength=4,
            ).argmax()
            selected = threshold_eligible & (dominant_corner == modal_corner)
        else:
            selected = threshold_eligible
        dominant_corner_selected_total += int(selected.sum().item())
        merged[:, original] = selected
        selected_reconstruction = reconstruction.gather(
            2, variant[..., None, None].expand(-1, -1, 1, dimension)
        ).squeeze(2)
        residual[:, original] = source - selected_reconstruction

    keep = identity[~merged].view(1, -1)
    return RecursiveLocalTokenPlan(
        keep=keep, parent_original=parents, weights=weights_all, residual=residual, merged=merged,
        original_tokens=total, frames=frames, tokens_per_frame=tokens_per_frame,
        metadata={
            "merge_ratio": float(merged.float().mean().item()),
            "similarity_threshold": float(similarity_threshold),
            "threshold_eligible_patch_ratio": threshold_eligible_total / max((frames - 1) * patches, 1),
            "dominant_corner_selected_among_eligible": dominant_corner_selected_total / max(threshold_eligible_total, 1),
            "dominant_corner_only": int(dominant_corner_only),
            "residual_l2_mean": float(residual[:, patch_token_start:].norm(dim=-1).mean().item()),
            "planner": "local_adjacent_2x2_analytic_v2",
        },
    )


def _local_2x2_windows(patch_ids: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    """All four 2x2 windows that can contain each same-coordinate patch."""
    row = torch.div(patch_ids, width, rounding_mode="floor")
    col = patch_ids.remainder(width)
    top = torch.stack(((row - 1).clamp(0, height - 2), row.clamp(0, height - 2)), dim=1)
    left = torch.stack(((col - 1).clamp(0, width - 2), col.clamp(0, width - 2)), dim=1)
    starts = torch.stack((top[:, 0] * width + left[:, 0], top[:, 0] * width + left[:, 1],
                          top[:, 1] * width + left[:, 0], top[:, 1] * width + left[:, 1]), dim=1)
    return starts[..., None] + patch_ids.new_tensor((0, 1, width, width + 1))


@torch.no_grad()
def build_affine_warp_plan(
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
    kv_only: bool = False,
    dense_special_kv: bool = False,
    pool_source_content: bool = True,
    qkv_features: torch.Tensor | None = None,
    qkv_similarity_threshold: float | None = None,
    source_query_correction: bool = False,
    recursive_kv: bool = False,
    compress_queries: bool = False,
    compact_mlp: bool = False,
    recursive_queries: bool = False,
    query_chain_window: int = 0,
    star_window: int = 0,
    star_phase: int = 0,
    query_target_merge_ratio: float = 0.45,
    query_similarity_threshold: float = 0.95,
    attention_query: torch.Tensor | None = None,
    attention_key: torch.Tensor | None = None,
    query_logit_error_threshold: float | None = None,
    query_logit_anchors: int = 64,
    target_merge_ratio: float | None = None,
    token_importance: torch.Tensor | None = None,
    mass_bias_scale: float = 0.0,
) -> WeightedBilinearTokenPlan:
    """Build an information-preserving plan from adjacent-frame affine warp.

    A uniform ``anchor_grid`` by ``anchor_grid`` set of target patches finds
    its most similar predecessor patch.  A dominant integer displacement
    rejects moving/outlier anchors, then a tiny affine least-squares fit maps
    target grid coordinates to predecessor grid coordinates.  Bilinear
    sampling of that *single* warp supplies the four parent ids and weights
    for every patch.  Only tokens agreeing with the virtual warped feature
    above ``similarity_threshold`` are removed.

    This deliberately replaces per-token 2x2 fitting / weight clustering:
    the shared geometric warp is the mode, and its bilinear sampler provides
    the corresponding local convex coefficients implicitly.  A selected token
    is pooled into its four retained parents *before* attention, rather than
    being deleted and reconstructed only after attention.
    """
    if assignment_mode not in {"soft_soft", "hard_hard", "hard_soft"}:
        raise ValueError("assignment_mode must be soft_soft, hard_hard, or hard_soft")
    if kv_only and assignment_mode != "hard_hard":
        raise ValueError("kv_only affine compaction requires hard_hard assignment")
    if tokens.shape[0] != 1:
        raise ValueError("affine-warp compaction currently supports inference batch size one")
    if not -1.0 <= similarity_threshold <= 1.000001:
        raise ValueError("similarity_threshold must lie in [-1, 1.000001]")
    if not -1.0 <= anchor_match_threshold <= 1.000001:
        raise ValueError("anchor_match_threshold must lie in [-1, 1.000001]")
    if anchor_grid < 2:
        raise ValueError("anchor_grid must be at least two")
    if qkv_similarity_threshold is not None and not -1.0 <= qkv_similarity_threshold <= 1.000001:
        raise ValueError("qkv_similarity_threshold must lie in [-1, 1.000001]")
    if recursive_kv and (not source_query_correction or pool_source_content):
        raise ValueError("recursive_kv requires source_query_correction and dropped source content")
    if (compress_queries or compact_mlp or recursive_queries) and not recursive_kv:
        raise ValueError("query/MLP compaction requires recursive_kv")
    if recursive_queries or query_chain_window or star_window or star_phase:
        raise NotImplementedError("recursive query compaction requires --fast-affine")
    if not 0.0 < query_target_merge_ratio < 1.0:
        raise ValueError("query_target_merge_ratio must lie in (0, 1)")
    if not -1.0 <= query_similarity_threshold <= 1.000001:
        raise ValueError("query_similarity_threshold must lie in [-1, 1.000001]")
    if query_logit_error_threshold is not None and query_logit_error_threshold <= 0.0:
        raise ValueError("query_logit_error_threshold must be positive")
    if query_logit_anchors < 1:
        raise ValueError("query_logit_anchors must be positive")
    if query_logit_error_threshold is not None and (attention_query is None or attention_key is None):
        raise ValueError("query logit gating requires attention_query and attention_key")
    if target_merge_ratio is not None and not 0.0 < target_merge_ratio < 1.0:
        raise ValueError("target_merge_ratio must lie in (0, 1)")
    if target_merge_ratio is not None and token_importance is None:
        raise ValueError("target_merge_ratio requires token_importance")
    if mass_bias_scale < 0.0:
        raise ValueError("mass_bias_scale must be non-negative")
    if mass_bias_scale > 0.0 and not recursive_kv:
        raise ValueError("mass bias currently requires recursive_kv")
    batch, total, dimension = tokens.shape
    height, width = patch_hw
    patches = height * width
    if total != frames * tokens_per_frame or tokens_per_frame != patch_token_start + patches:
        raise ValueError("Invalid affine-warp frame/patch layout")

    device = tokens.device
    identity = torch.arange(total, device=device).view(1, -1)
    patch_rows = torch.arange(height, device=device)
    patch_cols = torch.arange(width, device=device)
    full_row, full_col = torch.meshgrid(patch_rows, patch_cols, indexing="ij")
    # Coordinates use (x=column, y=row); homogeneous form maps target ->
    # predecessor.  This makes the later bilinear-parent construction direct.
    full_xy = torch.stack((full_col.reshape(-1), full_row.reshape(-1)), dim=-1).float()
    full_h = torch.cat((full_xy, torch.ones((patches, 1), device=device)), dim=-1)
    anchor_rows = torch.linspace(0, height - 1, min(anchor_grid, height), device=device).round().long().unique()
    anchor_cols = torch.linspace(0, width - 1, min(anchor_grid, width), device=device).round().long().unique()
    anchor_row, anchor_col = torch.meshgrid(anchor_rows, anchor_cols, indexing="ij")
    anchor_local = (anchor_row * width + anchor_col).reshape(-1)
    anchor_xy = full_xy[anchor_local]
    anchor_h = torch.cat((anchor_xy, torch.ones((anchor_xy.shape[0], 1), device=device)), dim=-1)

    reshaped = tokens.view(batch, frames, tokens_per_frame, dimension)
    if qkv_features is not None:
        if qkv_features.shape != (batch, total, 3 * dimension):
            raise ValueError("qkv_features do not match affine-warp token layout")
        qkv_reshaped = qkv_features.view(batch, frames, tokens_per_frame, 3, dimension)
    else:
        qkv_reshaped = None
    merged = torch.zeros((batch, total), dtype=torch.bool, device=device)
    # For hard assignment, every source is ultimately represented by one
    # physical predecessor token.  This head-to-tail forest is path-compressed
    # as frames are planned; it never requires recursive output restoration.
    hard_root = torch.arange(total, device=device)
    parents = torch.zeros((batch, total, 4), dtype=torch.long, device=device)
    weights_all = torch.zeros((batch, total, 4), dtype=torch.float32, device=device)
    weights_all[..., 0] = 1.0
    residual = torch.zeros_like(tokens, dtype=torch.float32)
    eligible_total = 0
    valid_total = 0
    anchor_inlier_total = 0
    anchor_count_total = 0
    qkv_gate_candidate_total = 0
    qkv_gate_selected_total = 0

    # Build head-to-tail. Soft restoration needs four physical parents. Hard
    # assignment instead follows a predecessor's physical representative.
    for frame in range(1, frames):
        source = F.normalize(reshaped[:, frame, patch_token_start:].float(), dim=-1, eps=1e-6)[0]
        previous = F.normalize(reshaped[:, frame - 1, patch_token_start:].float(), dim=-1, eps=1e-6)[0]
        anchor_source = source[anchor_local]
        anchor_score, anchor_match = (anchor_source @ previous.transpose(0, 1)).max(dim=-1)
        match_xy = full_xy[anchor_match]
        displacement = (match_xy - anchor_xy).round().long()
        candidate = anchor_score >= anchor_match_threshold
        if candidate.any():
            displacement_values, displacement_counts = torch.unique(
                displacement[candidate], dim=0, return_counts=True,
            )
            modal_displacement = displacement_values[displacement_counts.argmax()]
            # One-patch tolerance admits sub-patch/affine variation without
            # letting another moving object define the camera warp.
            inlier = candidate & ((displacement - modal_displacement).abs().amax(dim=-1) <= 1)
        else:
            modal_displacement = torch.zeros(2, dtype=torch.long, device=device)
            inlier = candidate

        # Fit target -> predecessor affine on modal-motion anchors.  If the
        # clip has too few reliable matches, use the modal translation only.
        if int(inlier.sum().item()) >= 3:
            affine = torch.linalg.lstsq(anchor_h[inlier], match_xy[inlier]).solution
            predicted_anchor = anchor_h @ affine
            reprojection = (predicted_anchor - match_xy).norm(dim=-1)
            refined = inlier & (reprojection <= 1.5)
            if int(refined.sum().item()) >= 3:
                affine = torch.linalg.lstsq(anchor_h[refined], match_xy[refined]).solution
                inlier = refined
        else:
            affine = torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [float(modal_displacement[0]), float(modal_displacement[1])]],
                device=device,
            )

        warped_xy = full_h @ affine
        x, y = warped_xy.unbind(dim=-1)
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
        virtual = (previous[local_parents] * local_weights[..., None]).sum(dim=1)
        agreement = F.cosine_similarity(source, virtual, dim=-1, eps=1e-6)
        original = frame * tokens_per_frame + patch_token_start + torch.arange(patches, device=device)
        # Soft restoration needs all four physical parents. Hard assignment
        # can safely path-compress an already merged predecessor to its final
        # physical representative; it has no soft recursive restore chain.
        global_parents = local_parents + (frame - 1) * tokens_per_frame + patch_token_start
        if assignment_mode == "hard_hard":
            hard_parent = global_parents.gather(1, local_weights.argmax(dim=-1, keepdim=True)).squeeze(1)
            parent_is_physical = torch.ones_like(valid)
        elif recursive_kv:
            # Dense source queries need no output interpolation.  Therefore a
            # source may be removed from K/V even if its virtual predecessor
            # was itself removed: the relationship is only a redundancy
            # certificate, not a recursive output-restoration graph.
            parent_is_physical = torch.ones_like(valid)
        else:
            parent_is_physical = (~merged[0, global_parents]).all(dim=-1)
        selected = valid & (agreement >= similarity_threshold) & parent_is_physical
        if qkv_similarity_threshold is not None:
            if qkv_reshaped is None:
                raise ValueError("qkv_similarity_threshold requires qkv_features")
            source_qkv = qkv_reshaped[0, frame, patch_token_start:].float()  # [P,3,D]
            previous_qkv = qkv_reshaped[0, frame - 1, patch_token_start:].float()
            virtual_qkv = (
                previous_qkv[local_parents] * local_weights[:, :, None, None]
            ).sum(dim=1)
            # One shared temporal transport must explain all three roles.  A
            # minimum, rather than average, prevents a strong V fit from
            # hiding an unsafe query or key direction.
            qkv_agreement = F.cosine_similarity(source_qkv, virtual_qkv, dim=-1, eps=1e-6).amin(dim=-1)
            qkv_gate_candidate_total += int(selected.sum().item())
            selected &= qkv_agreement >= qkv_similarity_threshold
            qkv_gate_selected_total += int(selected.sum().item())

        offset = (frame - 1) * tokens_per_frame + patch_token_start
        parents[:, original] = (local_parents + offset).unsqueeze(0)
        weights_all[:, original] = local_weights.unsqueeze(0)
        merged[:, original] = selected.unsqueeze(0)
        if assignment_mode == "hard_hard" and selected.any():
            hard_root[original[selected]] = hard_root[hard_parent[selected]]
        residual[:, original] = (source - virtual).unsqueeze(0)
        eligible_total += int(selected.sum().item())
        valid_total += int(valid.sum().item())
        anchor_inlier_total += int(inlier.sum().item())
        anchor_count_total += int(anchor_local.numel())

    query_candidates = merged.clone()
    candidate_merge_ratio = float(merged.float().mean().item())
    candidate_importance_fraction = 0.0
    removed_importance_fraction = 0.0
    if target_merge_ratio is not None:
        if token_importance is None or token_importance.numel() != total:
            raise ValueError("token_importance must contain one score per original token")
        candidate_ids = torch.where(merged[0])[0]
        total_importance = token_importance.float().clamp_min(0.0).sum().clamp_min(1e-12)
        candidate_importance_fraction = float(
            token_importance[candidate_ids].float().clamp_min(0.0).sum().div(total_importance).item()
        )
        target_count = min(candidate_ids.numel(), int(round(target_merge_ratio * total)))
        if target_count < candidate_ids.numel():
            # Delete the least camera-relevant candidates; high-contribution
            # K/V remain physical even when they satisfy the geometric fit.
            # Selection needs no ordering among the removed tokens: CUDA
            # top-k avoids the full O(N log N) stable sort used previously.
            selected_local = torch.topk(
                token_importance[candidate_ids], target_count,
                largest=False, sorted=False,
            ).indices
            selected_ids = candidate_ids[selected_local]
            merged.zero_()
            merged[0, selected_ids] = True
        removed_ids = torch.where(merged[0])[0]
        removed_importance_fraction = float(
            token_importance[removed_ids].float().clamp_min(0.0).sum().div(total_importance).item()
        )

    keep = identity[~merged].view(1, -1)
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=device).view(1, -1))
    source_original = identity[merged].view(1, -1)
    source_parents = parents[:, source_original[0]]
    if assignment_mode == "hard_hard" and source_original.shape[1]:
        # Hard mode uses exactly one path-compressed physical representative
        # in both pooling and restoration.
        source_parents = hard_root[source_original[0]].view(1, -1, 1).expand(-1, -1, 4)
    source_parent_compact = original_to_compact.gather(1, source_parents.reshape(1, -1)).reshape_as(source_parents)
    if (source_parent_compact < 0).any() and not recursive_kv:
        raise RuntimeError("affine-warp source has a non-physical parent")
    if recursive_kv:
        # These indices are deliberately unused: dense Query produces every
        # output and dropped source content is not scattered into K/V.
        source_parent_compact = source_parent_compact.clamp_min(0)
    key_log_mass = None
    max_key_mass = 1.0
    if mass_bias_scale > 0.0:
        # A dominant bilinear edge defines a unique backward-time forest.
        # After the final keep set is known, path propagation maps every
        # removed observation to one physical compact key.  This affects only
        # softmax multiplicity; physical K/V feature vectors stay untouched.
        dominant_parent = parents[0].gather(
            1, weights_all[0].argmax(dim=-1, keepdim=True),
        ).squeeze(1)
        identity_root = torch.arange(total, device=device)
        root = torch.where(merged[0], dominant_parent, identity_root)
        # Every edge points exactly one frame backward. Pointer jumping thus
        # resolves a T-frame chain in ceil(log2(T)) GPU gathers instead of T
        # Python-dispatched frame kernels, without changing its physical root.
        for _ in range(max(1, math.ceil(math.log2(max(frames, 2))))):
            root = root[root]
        representative = torch.where(merged[0], root, torch.arange(total, device=device))
        representative_compact = original_to_compact[0, representative]
        if (representative_compact < 0).any():
            raise RuntimeError("recursive multiplicity did not resolve to physical compact keys")
        mass = torch.zeros(keep.shape[1], device=device, dtype=torch.float32)
        mass.scatter_add_(0, representative_compact, torch.ones(total, device=device, dtype=torch.float32))
        max_key_mass = float(mass.max().item())
        key_log_mass = mass.clamp_min(1.0).log().mul_(mass_bias_scale).view(1, -1)
    query_keep = None
    query_source_original = None
    query_source_parent_compact = None
    query_restore_weights = None
    query_merge_ratio = 0.0
    if compress_queries or compact_mlp:
        # A query may be represented only by four physical queries in the
        # immediately preceding frame.  This prevents long recursive query
        # clusters from averaging unrelated temporal states.
        query_merged = torch.zeros_like(merged)
        sampled_key = None
        if query_logit_error_threshold is not None:
            anchor_ids = torch.linspace(
                0, total - 1, min(query_logit_anchors, total), device=device,
            ).round().long().unique()
            sampled_key = attention_key[0].index_select(1, anchor_ids)  # [H,A,d]
        for frame in range(1, frames):
            begin = frame * tokens_per_frame + patch_token_start
            end = begin + patches
            physical_parents = (~query_merged[0, parents[0, begin:end]]).all(dim=-1)
            if qkv_reshaped is None:
                raise RuntimeError("query compression requires projected Q/K/V")
            source_query = qkv_reshaped[0, frame, patch_token_start:, 0].float()
            previous_query = qkv_reshaped[0, frame - 1, patch_token_start:, 0].float()
            local_parent = parents[0, begin:end] - ((frame - 1) * tokens_per_frame + patch_token_start)
            virtual_query = (
                previous_query[local_parent]
                * weights_all[0, begin:end, :, None]
            ).sum(dim=1)
            query_agreement = F.cosine_similarity(source_query, virtual_query, dim=-1, eps=1e-6)
            selected_query = (
                query_candidates[0, begin:end]
                & physical_parents
                & (query_agreement >= query_similarity_threshold)
            )
            if sampled_key is not None:
                source_attention_query = attention_query[0, :, begin:end]
                parent_attention_query = attention_query[0, :, parents[0, begin:end]]
                virtual_attention_query = (
                    parent_attention_query
                    * weights_all[0, begin:end][None, :, :, None]
                ).sum(dim=2)
                delta_query = source_attention_query - virtual_attention_query
                max_logit_error = torch.einsum(
                    "hpd,had->hpa", delta_query.float(), sampled_key.float(),
                ).abs_().amax(dim=(0, 2)).mul_(sampled_key.shape[-1] ** -0.5)
                selected_query &= max_logit_error <= query_logit_error_threshold
            query_merged[0, begin:end] = selected_query
        candidate_ids = torch.where(query_merged[0])[0]
        target_count = min(candidate_ids.numel(), int(round(query_target_merge_ratio * total)))
        if target_count < candidate_ids.numel():
            # Reuse the K/V importance direction so queries with the largest
            # special-token influence remain exact.
            selected = torch.topk(
                token_importance[candidate_ids], target_count,
                largest=False, sorted=False,
            ).indices
            query_merged.zero_()
            query_merged[0, candidate_ids[selected]] = True
        query_keep = identity[~query_merged].view(1, -1)
        query_original_to_compact = torch.full_like(identity, -1)
        query_original_to_compact.scatter_(
            1, query_keep,
            torch.arange(query_keep.shape[1], device=device).view(1, -1),
        )
        query_source_original = identity[query_merged].view(1, -1)
        query_source_parent_compact = query_original_to_compact.gather(
            1, parents[:, query_source_original[0]].reshape(1, -1),
        ).reshape(1, -1, 4)
        if (query_source_parent_compact < 0).any():
            raise RuntimeError("query quotient contains a recursive parent")
        query_restore_weights = weights_all[:, query_source_original[0]]
        query_merge_ratio = float(query_merged.float().mean().item())
    bilinear_weights = weights_all[:, source_original[0]]
    if assignment_mode == "soft_soft":
        source_pool_weights = bilinear_weights
        source_restore_weights = bilinear_weights
    else:
        # A hard assignment writes all Q/K/V mass to the nearest (largest
        # bilinear coefficient) physical parent.  ``hard_soft`` deliberately
        # retains soft restoration as a diagnostic, whereas ``hard_hard`` is
        # a fully self-consistent nearest-parent quotient map.
        source_pool_weights = torch.zeros_like(bilinear_weights)
        source_pool_weights.scatter_(2, bilinear_weights.argmax(dim=-1, keepdim=True), 1.0)
        source_restore_weights = source_pool_weights if assignment_mode == "hard_hard" else bilinear_weights
    normalizer = torch.ones((1, keep.shape[1], 1), device=device, dtype=torch.float32)
    if pool_source_content:
        for corner in range(4):
            normalizer.scatter_add_(
                1,
                source_parent_compact[:, :, corner, None],
                source_pool_weights[:, :, corner, None],
            )
    special_query_original = (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + torch.arange(patch_token_start, device=device)[None, :]
    ).reshape(-1)
    special_query_compact = original_to_compact[0, special_query_original]
    if (special_query_compact < 0).any():
        raise RuntimeError("affine-warp plan unexpectedly removed a camera/register token")
    return WeightedBilinearTokenPlan(
        keep=keep,
        source_original=source_original,
        source_parent_compact=source_parent_compact,
        source_pool_weights=source_pool_weights,
        source_restore_weights=source_restore_weights,
        original_tokens=total,
        normalizer=normalizer,
        # Evaluating the kept queries and the source innovation queries in one
        # dense-Q/compact-KV SDPA is algebraically identical to the explicit
        # base-plus-query-residual form, but avoids a second attention launch.
        preserve_queries=bool((compact_mlp and not compress_queries) or ((kv_only or source_query_correction) and not compress_queries)),
        preserve_destination_keys=bool(kv_only),
        dense_special_queries=bool(dense_special_kv),
        special_query_original=special_query_original,
        special_query_compact=special_query_compact,
        pool_source_content=bool(pool_source_content),
        precomputed_qkv=qkv_features,
        source_query_correction=bool(source_query_correction and not compress_queries),
        key_log_mass=key_log_mass,
        query_keep=query_keep,
        query_source_original=query_source_original,
        query_source_parent_compact=query_source_parent_compact,
        query_restore_weights=query_restore_weights,
        compact_mlp=bool(compact_mlp),
        metadata={
            "merge_ratio": float(merged.float().mean().item()),
            "similarity_threshold": float(similarity_threshold),
            "anchor_grid": int(anchor_grid),
            "anchor_match_threshold": float(anchor_match_threshold),
            "assignment_mode": assignment_mode,
            "kv_only": bool(kv_only),
            "dense_special_kv": bool(dense_special_kv),
            "pool_source_content": bool(pool_source_content),
            "qkv_similarity_threshold": qkv_similarity_threshold,
            "qkv_gate_pass_ratio": (
                qkv_gate_selected_total / max(qkv_gate_candidate_total, 1)
                if qkv_similarity_threshold is not None else 1.0
            ),
            "source_query_correction": bool(source_query_correction),
            "recursive_kv": bool(recursive_kv),
            "compress_queries": bool(compress_queries),
            "query_target_merge_ratio": float(query_target_merge_ratio),
            "query_similarity_threshold": float(query_similarity_threshold),
            "query_logit_error_threshold": query_logit_error_threshold,
            "query_logit_anchors": int(query_logit_anchors),
            "query_merge_ratio": query_merge_ratio,
            "candidate_merge_ratio": candidate_merge_ratio,
            "target_merge_ratio": target_merge_ratio,
            "importance_policy": "camera_query_qk" if target_merge_ratio is not None else "none",
            "candidate_importance_fraction": candidate_importance_fraction,
            "removed_importance_fraction": removed_importance_fraction,
            "mass_bias_scale": float(mass_bias_scale),
            "max_key_mass": max_key_mass,
            "valid_patch_ratio": valid_total / max((frames - 1) * patches, 1),
            "anchor_inlier_ratio": anchor_inlier_total / max(anchor_count_total, 1),
            "planner": "adjacent_affine_warp_bilinear_v1",
        },
    )


@dataclass(frozen=True)
class DeepTrackGraphEvidence:
    """Reusable, unthresholded associations from one post-global token tensor."""

    forward_score: torch.Tensor
    forward_margin: torch.Tensor
    forward_flow: torch.Tensor
    backward_score: torch.Tensor
    backward_margin: torch.Tensor
    backward_flow: torch.Tensor
    feature_jump: torch.Tensor


def _patch_ids(frames: int, tokens_per_frame: int, patch_token_start: int, device: torch.device) -> torch.Tensor:
    patches = tokens_per_frame - patch_token_start
    return (
        torch.arange(frames, device=device)[:, None] * tokens_per_frame
        + patch_token_start
        + torch.arange(patches, device=device)
    )


@torch.no_grad()
def build_fastvggt_plan(
    tokens: torch.Tensor,
    *,
    frames: int,
    tokens_per_frame: int,
    patch_token_start: int,
    patch_hw: tuple[int, int],
    merge_ratio: float = 0.9,
    protect_ratio: float = 0.10,
    seed: int = 0,
    match_chunk_size: int = FAST_MATCH_SOURCE_CHUNK,
) -> TokenPlan:
    """Port of Any4D ``build_plan`` for Omega's interleaved token layout."""
    if not (0.0 <= merge_ratio <= 1.0 and 0.0 <= protect_ratio <= 1.0):
        raise ValueError("FastVGGT ratios must lie in [0, 1]")
    batch, total, _ = tokens.shape
    if total != frames * tokens_per_frame or patch_token_start + math.prod(patch_hw) != tokens_per_frame:
        raise ValueError("Invalid Omega FastVGGT frame/patch/special-token layout")
    identity = torch.arange(total, device=tokens.device).expand(batch, -1)
    if frames == 1 or merge_ratio == 0:
        return TokenPlan(identity, identity, total)

    height, width = patch_hw
    patches = height * width
    patch_ids = _patch_ids(frames, tokens_per_frame, patch_token_start, tokens.device)
    destination = torch.zeros(total, device=tokens.device, dtype=torch.bool)
    destination[patch_ids[0]] = True
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rows = torch.arange(0, height, 2)
    cols = torch.arange(0, width, 2)
    choices = torch.rand(frames - 1, len(rows), len(cols), 2, generator=generator)
    row = rows[None, :, None] + (
        choices[..., 0] * (height - rows).clamp(max=2)[None, :, None]
    ).long()
    col = cols[None, None, :] + (
        choices[..., 1] * (width - cols).clamp(max=2)[None, None, :]
    ).long()
    selected = (row * width + col).reshape(frames - 1, -1).to(tokens.device)
    destination[patch_ids[1:].gather(1, selected).flatten()] = True

    # Omega's camera/register tokens are the architecture's special tokens.
    # They take the role of Any4D's extra/special tokens and are never merged.
    protected = torch.ones(total, device=tokens.device, dtype=torch.bool)
    protected[patch_ids.flatten()] = False
    protected[patch_ids[0]] = True
    count = int(total * protect_ratio)
    if count:
        protected[torch.arange(0, total, max(1, total // count), device=tokens.device)[:count]] = True

    source_indices = (~(destination | protected)).nonzero().flatten()
    destination_indices = destination.nonzero().flatten()
    removed = min(int(total * merge_ratio), source_indices.numel())
    if removed == 0:
        return TokenPlan(identity, identity, total)

    # This is intentionally the same BF16, all-destination GEMM planner used
    # by Any4D, rather than the former nine-neighbour approximation.
    source_metric = F.normalize(tokens[:, source_indices].float(), dim=-1).to(torch.bfloat16)
    destination_metric_t = F.normalize(tokens[:, destination_indices].float(), dim=-1).to(torch.bfloat16).transpose(1, 2).contiguous()
    best = torch.empty(batch, source_indices.numel(), device=tokens.device, dtype=torch.float32)
    targets = torch.empty(batch, source_indices.numel(), device=tokens.device, dtype=torch.long)
    for start in range(0, source_indices.numel(), match_chunk_size):
        end = min(start + match_chunk_size, source_indices.numel())
        scores = torch.bmm(source_metric[:, start:end], destination_metric_t)
        values, indices = scores.max(dim=-1)
        best[:, start:end] = values.float()
        targets[:, start:end] = indices
    ranked = best.argsort(dim=-1, descending=True, stable=True)
    removed_source = source_indices[ranked[:, :removed]]
    target_original = destination_indices[targets.gather(1, ranked[:, :removed])]
    kept_mask = torch.ones(batch, total, device=tokens.device, dtype=torch.bool)
    kept_mask.scatter_(1, removed_source, False)
    keep = identity[kept_mask].reshape(batch, total - removed)
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(total - removed, device=tokens.device).expand(batch, -1))
    inverse = original_to_compact.clone()
    inverse.scatter_(1, removed_source, original_to_compact.gather(1, target_original))
    return TokenPlan(keep, inverse, total)


@torch.no_grad()
def build_operator_fastvggt_plan(
    tokens: torch.Tensor,
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    qkv_features: torch.Tensor,
    frames: int,
    tokens_per_frame: int,
    patch_token_start: int,
    patch_hw: tuple[int, int],
    merge_ratio: float = 0.9,
    protect_ratio: float = 0.10,
    probe_count: int = 8,
    match_chunk_size: int = FAST_MATCH_SOURCE_CHUNK,
) -> TokenPlan:
    """FastVGGT algebra with online attention-role medoids and assignments.

    The compact operator is deliberately unchanged: hard non-overlapping
    clusters mean-pool Q/K/V and broadcast one compact update to every member.
    Only representative selection and source assignment use a joint read/write
    signature built from the current layer's normalized Q/K and V.
    """
    if tokens.shape[0] != 1:
        raise ValueError("operator-signature FastVGGT currently supports batch size one")
    if not (0.0 <= merge_ratio <= 1.0 and 0.0 <= protect_ratio <= 1.0):
        raise ValueError("FastVGGT ratios must lie in [0, 1]")
    if probe_count < 2:
        raise ValueError("operator-signature probe_count must be at least two")
    batch, total, _ = tokens.shape
    height, width = patch_hw
    patches = height * width
    if total != frames * tokens_per_frame or patch_token_start + patches != tokens_per_frame:
        raise ValueError("Invalid Omega FastVGGT frame/patch/special-token layout")
    identity = torch.arange(total, device=tokens.device).expand(batch, -1)
    if frames == 1 or merge_ratio == 0:
        return TokenPlan(identity, identity, total, precomputed_qkv=qkv_features)

    patch_ids = _patch_ids(frames, tokens_per_frame, patch_token_start, tokens.device)
    camera_frames = torch.linspace(
        0, frames - 1, min(max(2, probe_count // 2), frames), device=tokens.device,
    ).round().long().unique()
    camera_probes = camera_frames * tokens_per_frame
    patch_probe_count = max(1, probe_count - camera_probes.numel())
    flat_patches = patch_ids.flatten()
    patch_probes = flat_patches[
        torch.linspace(0, flat_patches.numel() - 1, patch_probe_count, device=tokens.device)
        .round().long().unique()
    ]
    probes = torch.cat((camera_probes, patch_probes)).unique()

    scale = query.shape[-1] ** -0.5
    probe_key = key.index_select(2, probes)
    probe_query = query.index_select(2, probes)
    read = torch.einsum("bhnd,bhpd->bnhp", query.float(), probe_key.float()).mul_(scale)
    write = torch.einsum("bhpd,bhnd->bnhp", probe_query.float(), key.float()).mul_(scale)
    # Softmax ignores a constant logit-row offset; centre each head's probe
    # response before comparing read roles.
    read.sub_(read.mean(dim=-1, keepdim=True))
    write.sub_(write.mean(dim=-1, keepdim=True))
    read = F.normalize(read.flatten(2), dim=-1, eps=1e-6)
    write = F.normalize(write.flatten(2), dim=-1, eps=1e-6)
    value_flat = value.transpose(1, 2).reshape(batch, total, -1).float()
    value_stride = max(1, value_flat.shape[-1] // 64)
    value_signature = F.normalize(value_flat[..., ::value_stride], dim=-1, eps=1e-6)
    signature = F.normalize(
        torch.cat((read, write, value_signature), dim=-1), dim=-1, eps=1e-6,
    )

    # Keep FastVGGT's one-destination-per-2x2 budget, but choose the medoid in
    # current-layer attention-role space instead of a random spatial member.
    row0 = torch.arange(0, height, 2, device=tokens.device)
    col0 = torch.arange(0, width, 2, device=tokens.device)
    grid_r, grid_c = torch.meshgrid(row0, col0, indexing="ij")
    bases = (grid_r * width + grid_c).flatten()
    offsets = torch.tensor((0, 1, width, width + 1), device=tokens.device)
    local = bases[:, None] + offsets[None]
    valid = (
        (grid_r.flatten()[:, None] + torch.tensor((0, 0, 1, 1), device=tokens.device)[None] < height)
        & (grid_c.flatten()[:, None] + torch.tensor((0, 1, 0, 1), device=tokens.device)[None] < width)
    )
    local = local.clamp_max(patches - 1)
    frame_offsets = (
        torch.arange(1, frames, device=tokens.device)[:, None, None] * tokens_per_frame
        + patch_token_start
    )
    groups = frame_offsets + local[None]
    group_signature = signature[0, groups]
    pair_similarity = torch.einsum("fgid,fgjd->fgij", group_signature, group_signature)
    pair_similarity.masked_fill_(~valid[None, :, None, :], 0.0)
    medoid_score = pair_similarity.sum(dim=-1)
    medoid_score.masked_fill_(~valid[None], -torch.inf)
    selected_local = medoid_score.argmax(dim=-1)
    selected = groups.gather(2, selected_local[..., None]).squeeze(-1)

    destination = torch.zeros(total, device=tokens.device, dtype=torch.bool)
    destination[patch_ids[0]] = True
    destination[selected.flatten()] = True
    protected = torch.ones(total, device=tokens.device, dtype=torch.bool)
    protected[patch_ids.flatten()] = False
    protected[patch_ids[0]] = True
    count = int(total * protect_ratio)
    if count:
        protected[torch.arange(0, total, max(1, total // count), device=tokens.device)[:count]] = True

    source_indices = (~(destination | protected)).nonzero().flatten()
    destination_indices = destination.nonzero().flatten()
    removed = min(int(total * merge_ratio), source_indices.numel())
    if removed == 0:
        return TokenPlan(identity, identity, total, precomputed_qkv=qkv_features)

    source_metric = signature[:, source_indices].to(torch.bfloat16)
    destination_metric_t = signature[:, destination_indices].to(torch.bfloat16).transpose(1, 2).contiguous()
    best = torch.empty(batch, source_indices.numel(), device=tokens.device, dtype=torch.float32)
    targets = torch.empty(batch, source_indices.numel(), device=tokens.device, dtype=torch.long)
    for start in range(0, source_indices.numel(), match_chunk_size):
        end = min(start + match_chunk_size, source_indices.numel())
        scores = torch.bmm(source_metric[:, start:end], destination_metric_t)
        values, indices = scores.max(dim=-1)
        best[:, start:end] = values.float()
        targets[:, start:end] = indices
    ranked = best.argsort(dim=-1, descending=True, stable=True)
    removed_source = source_indices[ranked[:, :removed]]
    target_original = destination_indices[targets.gather(1, ranked[:, :removed])]
    kept_mask = torch.ones(batch, total, device=tokens.device, dtype=torch.bool)
    kept_mask.scatter_(1, removed_source, False)
    keep = identity[kept_mask].reshape(batch, total - removed)
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(
        1, keep, torch.arange(total - removed, device=tokens.device).expand(batch, -1),
    )
    inverse = original_to_compact.clone()
    inverse.scatter_(1, removed_source, original_to_compact.gather(1, target_original))
    return TokenPlan(
        keep, inverse, total,
        metadata={
            "merge_ratio": removed / total,
            "probe_count": int(probes.numel()),
            "signature_dim": int(signature.shape[-1]),
            "assignment_cosine_mean": float(best.gather(1, ranked[:, :removed]).mean().item()),
            "planner": "operator_signature_fastvggt_v1",
        },
        precomputed_qkv=qkv_features,
    )


@torch.no_grad()
def build_reference_anchor_plan(
    tokens: torch.Tensor,
    *,
    frames: int,
    tokens_per_frame: int,
    patch_token_start: int,
    patch_hw: tuple[int, int],
    similarity_threshold: float,
    reference_frame_count: int = 1,
    match_chunk_size: int = FAST_MATCH_SOURCE_CHUNK,
) -> TokenPlan:
    """Attach only high-confidence non-reference patches to reference patches.

    This is a deliberately narrow ablation of reference-conditioned
    redundancy.  Every non-reference patch whose best cosine match in the
    retained reference frames clears ``similarity_threshold`` is removed and
    receives that reference patch's attention update on restore.  Reference
    frames are uniformly spaced, always include the first and last frame, and
    remain fully uncompressed.  A non-reference frame compares only against
    the patch banks of its two temporally bracketing references, then attaches
    to the best patch across those two banks.  Thus a frame between anchors
    ``r_k`` and ``r_(k+1)`` cannot spuriously attach to an unrelated distant
    reference.  All medium- and low-similarity patches stay independent.

    Crucially, this is a *directed* attachment, not an average: source Q/K/V
    are excluded from the compact attention and cannot perturb the protected
    first-frame Q/K/V.  ``TokenPlan.pool_sources=False`` encodes that rule.
    The current inference path is batch-one; rejecting larger batches is safer
    than silently giving differently sized compact plans an invalid reshape.
    """
    # A value infinitesimally above one is the explicit no-attachment endpoint:
    # cosine can be exactly one, so threshold==1 is not a dense-baseline plan.
    if not -1.0 <= similarity_threshold <= 1.000001:
        raise ValueError("reference-anchor similarity threshold must lie in [-1, 1.000001]")
    if not 1 <= reference_frame_count <= frames:
        raise ValueError("reference-anchor frame count must lie in [1, frames]")
    batch, total, _ = tokens.shape
    if batch != 1:
        raise ValueError("reference-anchor merging currently requires batch size 1")
    if total != frames * tokens_per_frame or patch_token_start + math.prod(patch_hw) != tokens_per_frame:
        raise ValueError("Invalid Omega reference-anchor frame/patch/special-token layout")

    identity = torch.arange(total, device=tokens.device).expand(batch, -1)
    if frames == 1 or reference_frame_count == frames:
        return TokenPlan(identity, identity, total, pool_sources=False)

    patch_ids = _patch_ids(frames, tokens_per_frame, patch_token_start, tokens.device)
    if reference_frame_count == 1:
        # Preserve the original first-frame ablation exactly: it has no
        # temporal right bracket, so every later frame is compared only with
        # frame zero.
        reference_indices = patch_ids[0]
        source_indices = patch_ids[1:].flatten()
        source_metric = F.normalize(tokens[:, source_indices].float(), dim=-1).to(torch.bfloat16)
        reference_metric_t = F.normalize(tokens[:, reference_indices].float(), dim=-1).to(torch.bfloat16).transpose(1, 2).contiguous()
        best = torch.empty(batch, source_indices.numel(), device=tokens.device, dtype=torch.float32)
        targets = torch.empty(batch, source_indices.numel(), device=tokens.device, dtype=torch.long)
        for start in range(0, source_indices.numel(), match_chunk_size):
            end = min(start + match_chunk_size, source_indices.numel())
            scores = torch.bmm(source_metric[:, start:end], reference_metric_t)
            values, indices = scores.max(dim=-1)
            best[:, start:end] = values.float()
            targets[:, start:end] = indices
        removed_mask = best[0] >= similarity_threshold
        if not removed_mask.any():
            return TokenPlan(identity, identity, total, pool_sources=False)
        removed_source = source_indices[removed_mask].unsqueeze(0)
        target_original = reference_indices[targets[:, removed_mask]]
        kept_mask = torch.ones(batch, total, device=tokens.device, dtype=torch.bool)
        kept_mask.scatter_(1, removed_source, False)
        keep = identity[kept_mask].reshape(batch, total - removed_source.shape[1])
        original_to_compact = torch.full_like(identity, -1)
        original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=tokens.device).expand(batch, -1))
        inverse = original_to_compact.clone()
        inverse.scatter_(1, removed_source, original_to_compact.gather(1, target_original))
        return TokenPlan(keep, inverse, total, pool_sources=False)
    # Rounding follows NumPy/PyTorch's deterministic nearest-even convention;
    # with count=5 and T=300 this gives [0, 75, 150, 224, 299].
    reference_frames = torch.linspace(0, frames - 1, reference_frame_count, device=tokens.device).round().long()
    reference_mask = torch.zeros(frames, device=tokens.device, dtype=torch.bool)
    reference_mask[reference_frames] = True
    source_frames = (~reference_mask).nonzero(as_tuple=False).flatten()
    source_by_frame = patch_ids[source_frames]  # [S, P]
    patches = source_by_frame.shape[1]
    # Every source has a strict left and right bracketing reference because
    # the first and final video frames are always retained references.
    right_slot = torch.searchsorted(reference_frames, source_frames)
    left_slot = right_slot - 1
    reference_pair_by_frame = torch.cat((
        patch_ids[reference_frames[left_slot]],
        patch_ids[reference_frames[right_slot]],
    ), dim=1)  # [S, 2P]

    source_metric = F.normalize(tokens[0, source_by_frame].float(), dim=-1).to(torch.bfloat16)
    reference_metric = F.normalize(tokens[0, reference_pair_by_frame].float(), dim=-1).to(torch.bfloat16)
    # Keep peak planner memory bounded by the old source-token budget.
    frame_chunk = max(1, match_chunk_size // patches)
    best = torch.empty(source_by_frame.shape, device=tokens.device, dtype=torch.float32)
    targets = torch.empty(source_by_frame.shape, device=tokens.device, dtype=torch.long)
    for start in range(0, source_by_frame.shape[0], frame_chunk):
        end = min(start + frame_chunk, source_by_frame.shape[0])
        scores = torch.bmm(source_metric[start:end], reference_metric[start:end].transpose(1, 2))
        values, indices = scores.max(dim=-1)
        best[start:end] = values.float()
        targets[start:end] = indices

    source_indices = source_by_frame.flatten()
    target_original = reference_pair_by_frame.gather(1, targets).flatten().unsqueeze(0)
    removed_mask = best.flatten() >= similarity_threshold
    if not removed_mask.any():
        return TokenPlan(identity, identity, total, pool_sources=False)
    removed_source = source_indices[removed_mask].unsqueeze(0)
    target_original = target_original[:, removed_mask]
    kept_mask = torch.ones(batch, total, device=tokens.device, dtype=torch.bool)
    kept_mask.scatter_(1, removed_source, False)
    keep = identity[kept_mask].reshape(batch, total - removed_source.shape[1])
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=tokens.device).expand(batch, -1))
    inverse = original_to_compact.clone()
    inverse.scatter_(1, removed_source, original_to_compact.gather(1, target_original))
    return TokenPlan(keep, inverse, total, pool_sources=False)


@torch.no_grad()
def build_reference_anchor_cluster_plan(
    tokens: torch.Tensor,
    *,
    frames: int,
    tokens_per_frame: int,
    patch_token_start: int,
    patch_hw: tuple[int, int],
    high_similarity_threshold: float = .9,
    medium_similarity_lower: float = .7,
    cluster_similarity_threshold: float = .9,
    max_clusters_per_reference: int = 4,
    residual_fastvggt: bool = False,
    residual_keep_every: int = 10,
    residual_grid_position: int | None = None,
    residual_strong_clusters: bool = False,
    residual_max_clusters_per_reference: int = 16,
    residual_cluster_similarity_threshold: float | None = None,
    match_chunk_size: int = FAST_MATCH_SOURCE_CHUNK,
) -> TokenPlan:
    """Two-stage first-reference compaction with anchor-conditioned clusters.

    High-confidence patches attach *directionally* to an immutable first-frame
    patch, exactly as :func:`build_reference_anchor_plan`.  Surviving medium
    patches are partitioned by that same nearest first-frame patch.  Within
    each bucket, a high-reference-affinity seed gathers every still-unassigned
    member inside a *strong spherical cap* around that seed. The cap is set so
    any two selected members are guaranteed to have pairwise cosine similarity
    at least ``cluster_similarity_threshold``. Matching only the representative
    is therefore insufficient for a pooled cluster.  Optionally, a third
    stage applies either FastVGGT-style global similarity pooling only to the patch
    tokens not actually resolved by either reference-conditioned stage: one in every
    ``residual_keep_every`` becomes a destination, and every other survivor
    is mean-pooled into its most similar such destination. Optionally,
    ``residual_grid_position`` selects one fixed row-major location (0..8)
    from each 3x3 patch grid as another destination. Grid positions which
    already participated in stages one/two are not replaced by neighbours.
    Alternatively, ``residual_strong_clusters`` applies the same strict
    pairwise-consistent clustering rule to the unresolved tokens.  Tokens
    which find no >= ``residual_cluster_similarity_threshold`` partner are deliberately
    kept as singleton tokens; there is no periodic/grid protection in that
    mode.  The first frame and all special tokens are never part of either
    residual mode.
    """
    if not (-1.0 <= medium_similarity_lower < high_similarity_threshold <= 1.000001):
        raise ValueError("Reference cluster thresholds must satisfy -1 <= medium < high <= 1.000001")
    if not -1.0 <= cluster_similarity_threshold <= 1.000001:
        raise ValueError("Cluster similarity threshold must lie in [-1, 1.000001]")
    if max_clusters_per_reference < 1:
        raise ValueError("max_clusters_per_reference must be positive")
    if residual_keep_every < 2:
        raise ValueError("residual_keep_every must be at least two")
    if residual_grid_position is not None and not 0 <= residual_grid_position < 9:
        raise ValueError("residual_grid_position must be in [0, 8] or None")
    if residual_strong_clusters and residual_fastvggt:
        raise ValueError("Residual FastVGGT pooling and residual strong clustering are mutually exclusive")
    if residual_grid_position is not None and not residual_fastvggt:
        raise ValueError("residual_grid_position requires residual_fastvggt")
    if residual_max_clusters_per_reference < 1:
        raise ValueError("residual_max_clusters_per_reference must be positive")
    if residual_cluster_similarity_threshold is None:
        residual_cluster_similarity_threshold = cluster_similarity_threshold
    if not -1.0 <= residual_cluster_similarity_threshold <= 1.000001:
        raise ValueError("residual_cluster_similarity_threshold must lie in [-1, 1.000001]")
    # Let theta = arccos(tau). If each member lies at angular distance at most
    # theta/2 from the representative, any member pair is at distance <= theta.
    # Thus cos(theta/2) = sqrt((1 + tau)/2) is a sufficient, vectorisable
    # gate for the desired all-pairs threshold. The guard covers bf16 rounding
    # in the dot product below.
    strong_member_gate = min(
        1.0,
        math.sqrt((1.0 + cluster_similarity_threshold) * .5) + 2e-3,
    )
    residual_strong_member_gate = min(
        1.0,
        math.sqrt((1.0 + residual_cluster_similarity_threshold) * .5) + 2e-3,
    )
    batch, total, _ = tokens.shape
    if batch != 1:
        raise ValueError("Reference-anchor clustering currently requires batch size 1")
    if total != frames * tokens_per_frame or patch_token_start + math.prod(patch_hw) != tokens_per_frame:
        raise ValueError("Invalid Omega reference-anchor-cluster frame/patch/special-token layout")
    identity = torch.arange(total, device=tokens.device).expand(batch, -1)
    if frames == 1:
        return TokenPlan(identity, identity, total, pool_sources=False)

    patch_ids = _patch_ids(frames, tokens_per_frame, patch_token_start, tokens.device)
    reference_indices = patch_ids[0]
    source_indices = patch_ids[1:].flatten()
    source_metric = F.normalize(tokens[:, source_indices].float(), dim=-1).to(torch.bfloat16)
    reference_metric_t = F.normalize(tokens[:, reference_indices].float(), dim=-1).to(torch.bfloat16).transpose(1, 2).contiguous()
    best = torch.empty(source_indices.numel(), device=tokens.device, dtype=torch.float32)
    targets = torch.empty(source_indices.numel(), device=tokens.device, dtype=torch.long)
    for start in range(0, source_indices.numel(), match_chunk_size):
        end = min(start + match_chunk_size, source_indices.numel())
        values, indices = (source_metric[:, start:end] @ reference_metric_t).max(dim=-1)
        best[start:end] = values[0].float()
        targets[start:end] = indices[0]

    high = best >= high_similarity_threshold
    medium = (best >= medium_similarity_lower) & ~high
    target_original = reference_indices[targets]
    replacement = torch.full((total,), -1, device=tokens.device, dtype=torch.long)
    # Stage one: high matches restore the protected first-frame attention
    # update, but never contribute to its pooled Q/K/V state.
    replacement[source_indices[high]] = target_original[high]
    pool_mask = torch.ones((1, total), device=tokens.device, dtype=torch.bool)
    pool_mask[0, source_indices[high]] = False
    # This marks tokens actually resolved by stages one/two, not merely ones
    # considered as candidates. A retained stage-two representative has
    # participated in a real pooled cluster and cannot become a stage-three
    # source; otherwise its earlier members would retain an inverse mapping
    # to a token removed later. An isolated stage-two candidate is left clear
    # and must reach stage three unchanged.
    resolved_by_stage12 = high.clone()

    # Vectorised disjoint strong-cap clustering. Each round picks the highest
    # first-reference-affinity remaining seed in every anchor bucket at once,
    # then takes only members in the half-angle cap above. This avoids a Python
    # token-by-token complete-link loop while preserving its key invariant:
    # every pooled cluster obeys the original all-pairs similarity threshold.
    candidate = medium.clone()
    source_slots = torch.arange(source_indices.numel(), device=tokens.device)
    anchor_count = reference_indices.numel()
    cluster_count = 0
    clustered_medium_sources = 0
    for _ in range(max_clusters_per_reference):
        masked_best = torch.where(candidate, best, torch.full_like(best, float("-inf")))
        anchor_best = torch.full((anchor_count,), float("-inf"), device=tokens.device, dtype=best.dtype)
        anchor_best.scatter_reduce_(0, targets, masked_best, reduce="amax", include_self=True)
        is_seed = candidate & (best == anchor_best[targets])
        # Stable deterministic tie break: earliest source token wins.
        seed_slots = torch.full((anchor_count,), source_indices.numel(), device=tokens.device, dtype=torch.long)
        seed_slots.scatter_reduce_(0, targets[is_seed], source_slots[is_seed], reduce="amin", include_self=True)
        has_seed = seed_slots < source_indices.numel()
        gathered_seed_slots = seed_slots[targets].clamp_max(source_indices.numel() - 1)
        direct_similarity = (source_metric[0] * source_metric[0, gathered_seed_slots]).sum(dim=-1).float()
        matched = candidate & has_seed[targets] & (direct_similarity >= strong_member_gate)
        matched_per_anchor = torch.zeros(anchor_count, device=tokens.device, dtype=torch.long)
        matched_per_anchor.scatter_add_(0, targets, matched.long())
        active_anchor = matched_per_anchor >= 2
        selected = matched & active_anchor[targets]
        selected_seed = selected & (source_slots == gathered_seed_slots)
        removed_medium = selected & ~selected_seed
        replacement[source_indices[removed_medium]] = source_indices[gathered_seed_slots[removed_medium]]
        resolved_by_stage12 |= selected
        candidate[selected] = False
        # An isolated seed stays as an independent token, but is removed from
        # later seed competition so another member can represent a new mode.
        candidate[is_seed & ~active_anchor[targets]] = False
        cluster_count += int(active_anchor.sum().item())
        clustered_medium_sources += int(removed_medium.sum().item())

    # Stage three: make the unresolved remainder follow FastVGGT's actual
    # attention semantics (all-destination cosine match, Q/K/V mean pooling,
    # then broadcast restoration), but do *not* revisit anything already
    # resolved by stages one/two.  Unlike the released 2x2 representative
    # selector, this branch deliberately uses the requested exact 1-of-10
    # residual destination schedule.
    residual_candidates = ~resolved_by_stage12
    residual_indices = source_indices[residual_candidates]
    residual_destination_count = 0
    residual_merged_sources = 0
    residual_grid_candidates = 0
    residual_grid_destinations = 0
    residual_periodic_destinations = 0
    residual_strong_cluster_count = 0
    residual_strong_clustered_sources = 0
    residual_strong_singletons = 0
    if residual_strong_clusters and residual_indices.numel() >= 2:
        # The residual pass uses exactly the same anchor-bucketed, strong-cap
        # construction as stage two.  A selected group has at least two
        # members and every pair is guaranteed above the requested threshold.
        # An isolated seed is removed from candidate competition but remains
        # in the sequence, implementing the requested direct preservation of
        # every token without a qualifying 0.9-similar partner.
        residual_candidate = residual_candidates.clone()
        for _ in range(residual_max_clusters_per_reference):
            masked_best = torch.where(residual_candidate, best, torch.full_like(best, float("-inf")))
            anchor_best = torch.full((anchor_count,), float("-inf"), device=tokens.device, dtype=best.dtype)
            anchor_best.scatter_reduce_(0, targets, masked_best, reduce="amax", include_self=True)
            is_seed = residual_candidate & (best == anchor_best[targets])
            seed_slots = torch.full((anchor_count,), source_indices.numel(), device=tokens.device, dtype=torch.long)
            seed_slots.scatter_reduce_(0, targets[is_seed], source_slots[is_seed], reduce="amin", include_self=True)
            has_seed = seed_slots < source_indices.numel()
            gathered_seed_slots = seed_slots[targets].clamp_max(source_indices.numel() - 1)
            direct_similarity = (source_metric[0] * source_metric[0, gathered_seed_slots]).sum(dim=-1).float()
            matched = residual_candidate & has_seed[targets] & (direct_similarity >= residual_strong_member_gate)
            matched_per_anchor = torch.zeros(anchor_count, device=tokens.device, dtype=torch.long)
            matched_per_anchor.scatter_add_(0, targets, matched.long())
            active_anchor = matched_per_anchor >= 2
            selected = matched & active_anchor[targets]
            selected_seed = selected & (source_slots == gathered_seed_slots)
            removed_residual = selected & ~selected_seed
            replacement[source_indices[removed_residual]] = source_indices[gathered_seed_slots[removed_residual]]
            residual_candidate[selected] = False
            singleton_seed = is_seed & ~active_anchor[targets]
            residual_candidate[singleton_seed] = False
            residual_strong_singletons += int(singleton_seed.sum().item())
            residual_strong_cluster_count += int(active_anchor.sum().item())
            residual_strong_clustered_sources += int(removed_residual.sum().item())
        # Any candidates left after the configurable cluster budget are also
        # retained unchanged rather than falling through to another heuristic.
        residual_strong_singletons += int(residual_candidate.sum().item())
    elif residual_fastvggt and residual_indices.numel() >= 2:
        periodic_destination_indices = residual_indices[::residual_keep_every]
        residual_periodic_destinations = int(periodic_destination_indices.numel())
        if residual_grid_position is None:
            grid_destination_indices = source_indices.new_empty((0,))
        else:
            height, width = patch_hw
            row_offset, col_offset = divmod(residual_grid_position, 3)
            rows = torch.arange(row_offset, height, 3, device=tokens.device)
            cols = torch.arange(col_offset, width, 3, device=tokens.device)
            grid_flat = (rows[:, None] * width + cols[None, :]).reshape(-1)
            raw_grid_indices = patch_ids[1:, grid_flat].flatten()
            residual_grid_candidates = int(raw_grid_indices.numel())
            source_slot_of_original = torch.full((total,), -1, device=tokens.device, dtype=torch.long)
            source_slot_of_original[source_indices] = torch.arange(source_indices.numel(), device=tokens.device)
            grid_slots = source_slot_of_original[raw_grid_indices]
            grid_destination_indices = raw_grid_indices[~resolved_by_stage12[grid_slots]]
            residual_grid_destinations = int(grid_destination_indices.numel())
        residual_destination_indices = torch.unique(
            torch.cat((periodic_destination_indices, grid_destination_indices)),
            sorted=True,
        )
        residual_destination_mask = torch.zeros(total, device=tokens.device, dtype=torch.bool)
        residual_destination_mask[residual_destination_indices] = True
        residual_source_indices = residual_indices[~residual_destination_mask[residual_indices]]
        if residual_source_indices.numel():
            residual_source_metric = F.normalize(tokens[:, residual_source_indices].float(), dim=-1).to(torch.bfloat16)
            residual_destination_metric_t = F.normalize(
                tokens[:, residual_destination_indices].float(), dim=-1,
            ).to(torch.bfloat16).transpose(1, 2).contiguous()
            residual_targets = torch.empty(
                residual_source_indices.numel(), device=tokens.device, dtype=torch.long,
            )
            for start in range(0, residual_source_indices.numel(), match_chunk_size):
                end = min(start + match_chunk_size, residual_source_indices.numel())
                _, indices = (residual_source_metric[:, start:end] @ residual_destination_metric_t).max(dim=-1)
                residual_targets[start:end] = indices[0]
            replacement[residual_source_indices] = residual_destination_indices[residual_targets]
            residual_merged_sources = int(residual_source_indices.numel())
        residual_destination_count = int(residual_destination_indices.numel())

    removed = replacement >= 0
    if not removed.any():
        return TokenPlan(identity, identity, total, pool_sources=False)
    keep_mask = ~removed
    keep = identity[:, keep_mask]
    original_to_compact = torch.full_like(identity, -1)
    original_to_compact.scatter_(1, keep, torch.arange(keep.shape[1], device=tokens.device).expand(1, -1))
    inverse = original_to_compact.clone()
    removed_original = torch.nonzero(removed, as_tuple=False).flatten().unsqueeze(0)
    inverse.scatter_(1, removed_original, original_to_compact.gather(1, replacement[removed].unsqueeze(0)))
    return TokenPlan(
        keep, inverse, total, pool_sources=True, pool_mask=pool_mask,
        metadata={
            "high_reference_attachments": int(high.sum().item()),
            "medium_clustered_sources": clustered_medium_sources,
            "medium_cluster_count": cluster_count,
            "medium_candidates": int(medium.sum().item()),
            "max_clusters_per_reference": max_clusters_per_reference,
            "pairwise_similarity_threshold": cluster_similarity_threshold,
            "strong_member_gate": strong_member_gate,
            "residual_fastvggt": int(residual_fastvggt),
            "residual_keep_every": residual_keep_every if residual_fastvggt else 0,
            "residual_candidates": int(residual_indices.numel()),
            "residual_destinations": residual_destination_count,
            "residual_merged_sources": residual_merged_sources,
            "residual_grid_position": residual_grid_position,
            "residual_grid_candidates": residual_grid_candidates,
            "residual_grid_destinations": residual_grid_destinations,
            "residual_periodic_destinations": residual_periodic_destinations,
            "residual_strong_clusters": int(residual_strong_clusters),
            "residual_max_clusters_per_reference": (
                residual_max_clusters_per_reference if residual_strong_clusters else 0
            ),
            "residual_cluster_similarity_threshold": (
                residual_cluster_similarity_threshold if residual_strong_clusters else None
            ),
            "residual_strong_member_gate": (
                residual_strong_member_gate if residual_strong_clusters else None
            ),
            "residual_strong_cluster_count": residual_strong_cluster_count,
            "residual_strong_clustered_sources": residual_strong_clustered_sources,
            "residual_strong_singletons": residual_strong_singletons,
        },
    )


if triton is not None:
    @triton.jit
    def _local_temporal_score_kernel(x, out, total_locations: tl.constexpr,
                                     FRAMES: tl.constexpr, HEIGHT: tl.constexpr, WIDTH: tl.constexpr,
                                     CHANNELS: tl.constexpr, BLOCK_D: tl.constexpr):
        """Fused ±1-frame, 3x3 cosine-score reduction on normalized BF16 features."""
        pid = tl.program_id(0)
        valid_output = pid < total_locations
        per_batch = FRAMES * HEIGHT * WIDTH
        b = pid // per_batch
        rem = pid - b * per_batch
        t = rem // (HEIGHT * WIDTH)
        rem = rem - t * HEIGHT * WIDTH
        row = rem // WIDTH
        col = rem - row * WIDTH
        best = -float("inf")
        channels = tl.arange(0, BLOCK_D)
        for dt in range(-1, 2, 2):
            nt = t + dt
            valid_t = (nt >= 0) & (nt < FRAMES)
            for dy in range(-1, 2):
                nr = row + dy
                valid_r = (nr >= 0) & (nr < HEIGHT)
                for dx in range(-1, 2):
                    nc = col + dx
                    # ``F.unfold(..., padding=1)`` contributes a zero vector
                    # for off-image spatial neighbours; only a missing *time*
                    # neighbour is absent from the max reduction.
                    spatial_valid = valid_r & (nc >= 0) & (nc < WIDTH)
                    pair_valid = valid_output & valid_t
                    dot = 0.0
                    for dstart in range(0, CHANNELS, BLOCK_D):
                        d = dstart + channels
                        mask = pair_valid & spatial_valid & (d < CHANNELS)
                        left = (((b * FRAMES + t) * HEIGHT + row) * WIDTH + col) * CHANNELS + d
                        right = (((b * FRAMES + nt) * HEIGHT + nr) * WIDTH + nc) * CHANNELS + d
                        dot += tl.sum(tl.load(x + left, mask=mask, other=0.0) * tl.load(x + right, mask=mask, other=0.0), axis=0)
                    best = tl.maximum(best, tl.where(pair_valid, dot, -float("inf")))
        tl.store(out + pid, best, mask=valid_output)


@torch.no_grad()
def _reference_local_temporal_scores(image_tokens: torch.Tensor, frame_chunk: int = LABEL_FRAME_CHUNK) -> torch.Tensor:
    """The Any4D default ``unfold`` implementation, adapted only by shape."""
    b, t, h, w, _ = image_tokens.shape
    score = torch.full((b, t, h, w), -1.0, device=image_tokens.device)
    for delta in (-1, 1):
        src = torch.arange(t, device=image_tokens.device)
        dst = src + delta
        ok = (dst >= 0) & (dst < t)
        src, dst = src[ok], dst[ok]
        for start in range(0, src.numel(), frame_chunk):
            a = F.normalize(image_tokens[:, src[start:start + frame_chunk]].float(), dim=-1).to(torch.bfloat16)
            z = F.normalize(image_tokens[:, dst[start:start + frame_chunk]].float(), dim=-1).to(torch.bfloat16)
            patches = F.unfold(z.permute(0, 1, 4, 2, 3).reshape(-1, z.shape[-1], h, w), 3, padding=1)
            patches = patches.reshape(b, -1, z.shape[-1], 9, h, w).permute(0, 1, 4, 5, 3, 2)
            r = (a.unsqueeze(-2) * patches).sum(-1).max(-1).values.float()
            score[:, src[start:start + frame_chunk]] = torch.maximum(score[:, src[start:start + frame_chunk]], r)
    return score


@torch.no_grad()
def local_temporal_scores(image_tokens: torch.Tensor, *, backend: str = "triton") -> torch.Tensor:
    """Legacy Any4D similarity score; NOT used by ``temporal_labels``.

    ``reference`` is retained for A/B validation. The Triton kernel computes
    the ±1-frame 3x3 reduction without the 9x expanded unfold tensor; floating
    point reduction order need not be bitwise identical to the reference.
    """
    if backend not in {"triton", "reference"}:
        raise ValueError(f"Unknown temporal label backend: {backend}")
    if backend == "reference" or triton is None or not image_tokens.is_cuda:
        return _reference_local_temporal_scores(image_tokens)
    b, t, h, w, c = image_tokens.shape
    # Both implementations normalize in fp32 then score BF16 features, as in Any4D.
    normalized = F.normalize(image_tokens.float(), dim=-1).to(torch.bfloat16).contiguous()
    out = torch.empty((b, t, h, w), device=image_tokens.device, dtype=torch.float32)
    block_d = 128
    total = b * t * h * w
    try:
        _local_temporal_score_kernel[(total,)](
            normalized, out, total, FRAMES=t, HEIGHT=h, WIDTH=w, CHANNELS=c, BLOCK_D=block_d,
            num_warps=4,
        )
        return out
    except Exception:
        # CUDA-less nodes and mismatched Triton installations must preserve the
        # method's numerical definition rather than fail an evaluation job.
        return _reference_local_temporal_scores(image_tokens)


@torch.no_grad()
def _pairwise_local_tracks(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    radius: int = 2,
    frame_chunk: int = LABEL_FRAME_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate local feature tracks, top-2 margins and flow for frame pairs.

    ``source`` and ``target`` have ``[B,T,H,W,C]`` layouts, where pair ``i``
    always relates source frame ``i`` to target frame ``i``.  A track is a
    local feature correspondence, not an optical-flow prediction head: its
    role is solely to determine temporal predictability for token coarsening.
    """
    b, pair_count, h, w, c = source.shape
    if target.shape != source.shape:
        raise ValueError("source and target track tensors must have equal shape")
    if radius < 1 or frame_chunk < 1 or pair_count < 1:
        raise ValueError("local tracking radius, frame chunk and pair count must be positive")
    scores, margins, flows = [], [], []
    row = torch.arange(h, device=source.device)[:, None]
    col = torch.arange(w, device=source.device)[None, :]
    neg_inf = torch.tensor(float("-inf"), device=source.device)
    for start in range(0, pair_count, frame_chunk):
        end = min(start + frame_chunk, pair_count)
        src = F.normalize(source[:, start:end].float(), dim=-1).to(torch.bfloat16)
        dst = F.normalize(target[:, start:end].float(), dim=-1).to(torch.bfloat16)
        best = torch.full(src.shape[:-1], float("-inf"), device=source.device)
        second = torch.full_like(best, float("-inf"))
        flow_x = torch.zeros_like(best, dtype=torch.int16)
        flow_y = torch.zeros_like(best, dtype=torch.int16)
        # Invalid spatial candidates never win. This mask alone cannot detect
        # new field of view: confidence/null matches and forward coverage below
        # supply that evidence instead.
        padded = F.pad(
            dst.permute(0, 1, 4, 2, 3).reshape(-1, c, h, w),
            (radius, radius, radius, radius),
        ).reshape(b, end - start, c, h + 2 * radius, w + 2 * radius)
        for dy in range(-radius, radius + 1):
            valid_y = (row + dy >= 0) & (row + dy < h)
            for dx in range(-radius, radius + 1):
                valid = valid_y & (col + dx >= 0) & (col + dx < w)
                candidate = padded[:, :, :, radius + dy:radius + dy + h, radius + dx:radius + dx + w]
                candidate = candidate.permute(0, 1, 3, 4, 2)
                score = (src * candidate).sum(dim=-1).float()
                score = torch.where(valid[None, None], score, neg_inf)
                replace_best = score > best
                second = torch.where(replace_best, best, torch.maximum(second, score))
                best = torch.where(replace_best, score, best)
                flow_x = torch.where(replace_best, torch.full_like(flow_x, dx), flow_x)
                flow_y = torch.where(replace_best, torch.full_like(flow_y, dy), flow_y)
        scores.append(best)
        margins.append((best - second).clamp_min(0.0))
        flows.append(torch.stack((flow_x, flow_y), dim=-1))
        del src, dst, padded
    return torch.cat(scores, dim=1), torch.cat(margins, dim=1), torch.cat(flows, dim=1)


@torch.no_grad()
def _round_trip_error(forward: torch.Tensor, reverse: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward/backward track closure error and in-bounds validity."""
    b, pairs, h, w, _ = forward.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=forward.device), torch.arange(w, device=forward.device), indexing="ij"
    )
    target_x = xx[None, None] + forward[..., 0].long()
    target_y = yy[None, None] + forward[..., 1].long()
    valid = (target_x >= 0) & (target_x < w) & (target_y >= 0) & (target_y < h)
    linear = (target_y.clamp(0, h - 1) * w + target_x.clamp(0, w - 1)).reshape(b, pairs, h * w)
    reverse_at_target = reverse.reshape(b, pairs, h * w, 2).gather(
        2, linear[..., None].expand(-1, -1, -1, 2)
    ).reshape(b, pairs, h, w, 2)
    error = (forward.float() + reverse_at_target.float()).square().sum(dim=-1).sqrt()
    return error, valid


@torch.no_grad()
def _flow_discontinuity(flow: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Largest reliable four-neighbour motion jump, on BOTH sides of an edge.

    Unlike centre-minus-median, this detects a straight boundary between two
    internally constant flows. Unreliable matches cannot manufacture an edge.
    """
    b, pairs, h, w, _ = flow.shape
    x = flow.float().permute(0, 1, 4, 2, 3).reshape(b * pairs, 2, h, w)
    mask = valid.reshape(b * pairs, h, w)
    padded = F.pad(x, (1, 1, 1, 1))
    padded_mask = F.pad(mask, (1, 1, 1, 1), value=False)
    result = torch.zeros_like(mask, dtype=torch.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = padded[:, :, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        pair_valid = mask & padded_mask[:, 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]
        jump = (x - neighbour).square().sum(dim=1).sqrt()
        result = torch.maximum(result, torch.where(pair_valid, jump, 0.0))
    return result.reshape(b, pairs, h, w)


@torch.no_grad()
def _minimum_neighbour_motion_jump(flow: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Smallest reliable same-side motion difference for each token.

    A real object boundary has a large cross-edge jump while each side remains
    locally coherent.  A failed token match usually has large differences to
    all of its neighbours.  This inexpensive distinction is computed only
    from the existing deep-token tracks.
    """
    b, pairs, height, width, _ = flow.shape
    source = flow.float().permute(0, 1, 4, 2, 3).reshape(b * pairs, 2, height, width)
    mask = valid.reshape(b * pairs, height, width)
    padded = F.pad(source, (1, 1, 1, 1))
    padded_mask = F.pad(mask, (1, 1, 1, 1), value=False)
    result = torch.full_like(mask, float("inf"), dtype=torch.float32)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        neighbour = padded[:, :, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        pair_valid = mask & padded_mask[:, 1 + dy:1 + dy + height, 1 + dx:1 + dx + width]
        difference = (source - neighbour).square().sum(dim=1).sqrt()
        result = torch.minimum(result, torch.where(pair_valid, difference, torch.full_like(difference, float("inf"))))
    return result.reshape(b, pairs, height, width)


@torch.no_grad()
def _feature_discontinuity(image_tokens: torch.Tensor) -> torch.Tensor:
    """Put the deep-token appearance jump on both sides of each grid edge."""
    b, t, h, w, _ = image_tokens.shape
    tokens = F.normalize(image_tokens.float(), dim=-1)
    result = torch.zeros((b, t, h, w), device=image_tokens.device, dtype=torch.float32)
    if w > 1:
        horizontal = 1.0 - (tokens[:, :, :, :-1] * tokens[:, :, :, 1:]).sum(dim=-1)
        result[:, :, :, :-1] = torch.maximum(result[:, :, :, :-1], horizontal)
        result[:, :, :, 1:] = torch.maximum(result[:, :, :, 1:], horizontal)
    if h > 1:
        vertical = 1.0 - (tokens[:, :, :-1] * tokens[:, :, 1:]).sum(dim=-1)
        result[:, :, :-1] = torch.maximum(result[:, :, :-1], vertical)
        result[:, :, 1:] = torch.maximum(result[:, :, 1:], vertical)
    return result


@torch.no_grad()
def _clip_robust_upper(values: torch.Tensor, valid: torch.Tensor, *, scale: float) -> torch.Tensor:
    """One robust upper threshold per video, rather than a per-frame quota."""
    flat = values.flatten(start_dim=1)
    valid_flat = valid.flatten(start_dim=1) & torch.isfinite(flat)
    masked = torch.where(valid_flat, flat, torch.full_like(flat, float("nan")))
    median = torch.nanmedian(masked, dim=-1, keepdim=True).values.nan_to_num(0.0)
    mad = torch.nanmedian((masked - median).abs(), dim=-1, keepdim=True).values.nan_to_num(0.0)
    return median + scale * 1.4826 * mad.clamp_min(1e-4)


@torch.no_grad()
def _camera_compensated_residual(flow: torch.Tensor) -> torch.Tensor:
    """Residual local-track motion after robust affine camera compensation.

    The local matcher already yields a dense, integer patch displacement.  A
    robust affine fit describes its dominant image-plane component (camera
    translation/rotation/zoom at patch scale); the remaining magnitude is the
    motion evidence used only to disambiguate otherwise weak contours.  No RGB
    flow network, depth, pose head, cached label, or second forward is used.
    """
    batch, pairs, height, width, _ = flow.shape
    count = height * width
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=torch.float32),
        torch.arange(width, device=flow.device, dtype=torch.float32),
        indexing="ij",
    )
    source = torch.stack((xx, yy), dim=-1).reshape(1, 1, count, 2)
    target = source + flow.float().reshape(batch, pairs, count, 2)
    design = torch.cat((source.expand(batch, pairs, -1, -1), torch.ones_like(source[..., :1]).expand(batch, pairs, -1, -1)), dim=-1)
    design = design.reshape(batch * pairs, count, 3)
    target = target.reshape(batch * pairs, count, 2)
    # Iteratively reweighted least squares keeps independently moving objects
    # from defining the global camera fit.  Three inexpensive 3-parameter
    # solves are enough for the coarse 5x5 local-track field.
    weights = torch.ones((batch * pairs, count), device=flow.device, dtype=torch.float32)
    residual = torch.zeros((batch * pairs, count), device=flow.device, dtype=torch.float32)
    for _ in range(3):
        root = weights.sqrt().unsqueeze(-1)
        affine = torch.linalg.lstsq(design * root, target * root).solution
        residual = ((design @ affine - target).square().sum(dim=-1)).sqrt()
        median = residual.median(dim=-1, keepdim=True).values
        weights = (residual <= (2.5 * median).clamp_min(0.25)).to(torch.float32)
    return residual.reshape(batch, pairs, height, width)


@torch.no_grad()
def _spatially_supported(candidate: torch.Tensor, *, minimum: int) -> torch.Tensor:
    """Reject isolated patch-grid responses while retaining connected contours."""
    if minimum <= 1:
        return candidate
    batch, pairs, height, width = candidate.shape
    neighbourhood = F.avg_pool2d(candidate.float().reshape(-1, 1, height, width), 3, stride=1, padding=1) * 9.0
    return candidate & (neighbourhood.reshape(batch, pairs, height, width) >= minimum)


@torch.no_grad()
def _spatial_rank01(values: torch.Tensor) -> torch.Tensor:
    """Rank a token cue independently in every frame to [0, 1]."""
    flat = values.flatten(start_dim=-2)
    order = torch.argsort(flat, dim=-1, stable=True)
    rank = torch.empty_like(flat, dtype=torch.float32)
    positions = torch.arange(flat.shape[-1], device=flat.device, dtype=torch.float32).expand_as(flat)
    rank.scatter_(-1, order, positions)
    return (rank / max(flat.shape[-1] - 1, 1)).reshape_as(values)


@torch.no_grad()
def _native_patch_labels(image_tokens: torch.Tensor) -> torch.Tensor:
    """Classify patch roles from native DINO tokens of this same forward.

    Native features preserve sharper object contours than post-global tokens.
    The first global block is still executed full resolution; this function is
    called only after it and reuses the original patch tensor already produced
    by the encoder.  Thus it adds neither an image-model pass nor external
    weights.  Stable includes coherent object interiors; sensitive is a sparse
    high-contrast boundary cue, and unknown is reserved for an extreme
    predecessor mismatch rather than generic low texture.
    """
    batch, frames, height, width, _ = image_tokens.shape
    labels = torch.full((batch, frames, height, width), 2, device=image_tokens.device, dtype=torch.long)
    if frames == 1:
        return labels
    evidence = build_deep_track_graph_evidence(image_tokens)
    edge = _feature_discontinuity(image_tokens)[:, 1:]
    confidence_loss = 1.0 - evidence.backward_score.float()
    cue = _spatial_rank01(edge) + TEMPORAL_LABEL_CONFIG.native_sensitive_confidence_weight * _spatial_rank01(confidence_loss)
    count = max(1, int(round(height * width * TEMPORAL_LABEL_CONFIG.native_sensitive_top_ratio)))
    sensitive = torch.zeros_like(cue, dtype=torch.bool)
    sensitive.flatten(start_dim=-2).scatter_(
        -1, torch.topk(cue.flatten(start_dim=-2), count, dim=-1, largest=True, sorted=False).indices, True,
    )
    valid = torch.isfinite(evidence.backward_score)
    lower, _ = _robust_bounds(
        evidence.backward_score.float(), valid, scale=TEMPORAL_LABEL_CONFIG.native_unknown_mad_scale,
    )
    unknown = valid & (evidence.backward_score.float() <= lower)
    labels[:, 1:] = torch.where(unknown, 2, torch.where(sensitive, 1, 0))
    return labels


@torch.no_grad()
def adaptive_native_patch_labels(image_tokens: torch.Tensor) -> torch.Tensor:
    """Self-calibrated boundary roles from native Omega patch tokens.

    Sensitive is an appearance boundary whose predecessor is comparatively
    difficult to explain. Its threshold is robustly self-calibrated over the
    clip, so a static scene may yield no sensitive tokens and a small/distant
    object need not occupy the same quota as a close one. This method never
    forces a Top-K quota. Only extreme predecessor mismatches become unknown.
    All evidence comes from this forward's existing patch features.
    """
    batch, frames, height, width, _ = image_tokens.shape
    labels = torch.full((batch, frames, height, width), 2, device=image_tokens.device, dtype=torch.long)
    if frames == 1:
        return labels

    evidence = build_deep_track_graph_evidence(image_tokens)
    # Strict closed-track visibility is intentionally not used as an unknown
    # mask: native patch descriptors often have ambiguous local matches on
    # otherwise ordinary surfaces, which would turn most of a clip into false
    # "newly visible" content. Unknown remains a rare predecessor outlier.
    finite = torch.isfinite(evidence.backward_score)
    lower, _ = _robust_bounds(
        evidence.backward_score.float(), finite, scale=TEMPORAL_LABEL_CONFIG.native_unknown_mad_scale,
    )
    unknown = finite & (evidence.backward_score.float() <= lower)
    edge = _feature_discontinuity(image_tokens)[:, 1:]
    cfg = TEMPORAL_LABEL_CONFIG
    confidence_loss = 1.0 - evidence.backward_score.float()
    # Keep the cues in their native cosine-distance scale. Unlike a per-frame
    # rank, this retains the fact that a close/strong boundary carries more
    # evidence than a weak distant boundary; clip-level robust thresholds then
    # select evidence tails rather than forcing a per-frame quota.
    score = edge + cfg.adaptive_confidence_weight * confidence_loss
    strong_threshold = _clip_robust_upper(score, ~unknown, scale=cfg.adaptive_strong_mad_scale)
    weak_threshold = _clip_robust_upper(score, ~unknown, scale=cfg.adaptive_weak_mad_scale)
    # A weak visual contour becomes meaningful only if its local displacement
    # is not explained by the dominant affine camera motion.  Strong contours
    # remain eligible even for an almost-stationary person, so the rule does
    # not conflate "motion" with semantic importance.
    residual = _camera_compensated_residual(evidence.backward_flow)
    motion_threshold = _clip_robust_upper(residual, torch.ones_like(residual, dtype=torch.bool), scale=cfg.adaptive_motion_mad_scale)
    strong = score >= strong_threshold.reshape(-1, 1, 1, 1)
    weak_dynamic = (score >= weak_threshold.reshape(-1, 1, 1, 1)) & (residual >= motion_threshold.reshape(-1, 1, 1, 1))
    # If every patch has the same predecessor confidence, a rank tie must not
    # manufacture a boundary quota from arbitrary raster order.  This allows a
    # perfectly time-invariant frame to have zero sensitive tokens.
    confidence_range = confidence_loss.flatten(start_dim=-2).amax(-1) - confidence_loss.flatten(start_dim=-2).amin(-1)
    temporally_active = confidence_range > cfg.adaptive_min_confidence_range
    active = (~unknown) & temporally_active[..., None, None]
    # Very strong, spatially coherent contours retain their direct evidence.
    # Applying the temporal gate to them indiscriminately erased persistent
    # person silhouettes in short / slow-motion clips.  The lower-confidence
    # branch is where trajectory continuity disambiguates flicker from a real
    # dynamic boundary.
    strong = _spatially_supported(active & strong, minimum=cfg.adaptive_min_spatial_support)
    weak_dynamic = _spatially_supported(active & weak_dynamic, minimum=cfg.adaptive_min_spatial_support)
    # A local correspondence failure often flashes for one frame on screens,
    # reflections, or repeated texture.  A physical object boundary persists
    # after transport by the already-computed forward track.  This support is
    # deliberately evaluated after the spatial rule: isolated points cannot
    # become valid merely by coinciding with a different-frame outlier.
    if cfg.adaptive_temporal_boundary_support:
        weak_dynamic = _persistent_boundary(weak_dynamic, evidence.forward_flow)
    sensitive = strong | weak_dynamic
    labels[:, 1:] = torch.where(unknown, 2, torch.where(sensitive, 1, 0))
    return labels


@torch.no_grad()
def _scalar_spatial_gradient(values: torch.Tensor) -> torch.Tensor:
    """Maximum four-neighbour discontinuity of a scalar token field."""
    batch, pairs, height, width = values.shape
    result = torch.zeros((batch, pairs, height, width), device=values.device, dtype=values.dtype)
    if height > 1:
        delta = (values[:, :, 1:] - values[:, :, :-1]).abs()
        result[:, :, 1:] = torch.maximum(result[:, :, 1:], delta)
        result[:, :, :-1] = torch.maximum(result[:, :, :-1], delta)
    if width > 1:
        delta = (values[:, :, :, 1:] - values[:, :, :, :-1]).abs()
        result[:, :, :, 1:] = torch.maximum(result[:, :, :, 1:], delta)
        result[:, :, :, :-1] = torch.maximum(result[:, :, :, :-1], delta)
    return result


@torch.no_grad()
def _epipolar_residual(flow: torch.Tensor, *, iterations: int = 3) -> torch.Tensor:
    """Robust Sampson residual of dense patch correspondences on the GPU.

    A camera moving through a rigid 3-D scene generally cannot be explained
    by one affine image warp: depth parallax is expected at desk/chair edges.
    It *is* constrained by a per-pair fundamental matrix.  This light-weight
    batched normalized eight-point fit therefore separates rigid camera
    parallax from independently moving regions without depth, pose, optical
    flow networks, or a second forward.  It is an evidence score, never a
    claim of metrically calibrated geometry.
    """
    batch, pairs, height, width, _ = flow.shape
    count = height * width
    if count < 8:
        return torch.zeros((batch, pairs, height, width), device=flow.device, dtype=torch.float32)
    yy, xx = torch.meshgrid(
        torch.arange(height, device=flow.device, dtype=torch.float32),
        torch.arange(width, device=flow.device, dtype=torch.float32), indexing="ij",
    )
    # Isotropic normalisation keeps the SVD well-conditioned across the
    # non-square token grids used by balanced image preprocessing.
    scale = float(max(height, width))
    source = torch.stack(((xx - (width - 1) * .5) / scale, (yy - (height - 1) * .5) / scale), dim=-1)
    source = source.reshape(1, 1, count, 2).expand(batch, pairs, -1, -1)
    target = source + flow.float().reshape(batch, pairs, count, 2) / scale
    ones = torch.ones_like(source[..., :1])
    x = torch.cat((source, ones), dim=-1)
    xp = torch.cat((target, ones), dim=-1)

    def design_matrix(weights: torch.Tensor) -> torch.Tensor:
        sx, sy = x.unbind(-1)[:2]
        tx, ty = xp.unbind(-1)[:2]
        rows = torch.stack((tx * sx, tx * sy, tx, ty * sx, ty * sy, ty, sx, sy, torch.ones_like(sx)), dim=-1)
        return rows * weights.sqrt().unsqueeze(-1)

    weights = torch.ones((batch, pairs, count), device=flow.device, dtype=torch.float32)
    residual = torch.zeros_like(weights)
    for _ in range(iterations):
        _, _, vh = torch.linalg.svd(design_matrix(weights), full_matrices=False)
        fundamental = vh[..., -1, :].reshape(batch, pairs, 3, 3)
        u, singular, vh_f = torch.linalg.svd(fundamental)
        singular[..., -1] = 0.0
        fundamental = (u * singular.unsqueeze(-2)) @ vh_f
        fx = x @ fundamental.transpose(-1, -2)
        ftxp = xp @ fundamental
        algebraic = (xp * fx).sum(dim=-1)
        denom = (fx[..., :2].square().sum(dim=-1) + ftxp[..., :2].square().sum(dim=-1)).clamp_min(1e-6)
        residual = (algebraic.square() / denom).sqrt()
        median = residual.median(dim=-1, keepdim=True).values.clamp_min(1e-4)
        weights = (residual <= 2.5 * median).to(torch.float32)
        # Avoid an empty robust set on a pathological correspondence field.
        weights = torch.where(weights.sum(dim=-1, keepdim=True) >= 8, weights, torch.ones_like(weights))
    return residual.reshape(batch, pairs, height, width)


@torch.no_grad()
def _persistent_boundary_tolerant(candidate: torch.Tensor, forward_flow: torch.Tensor, *, radius: int) -> torch.Tensor:
    """Temporal support after a small spatial tolerance around a transported edge.

    Patch-grid tracks are integer valued.  Requiring an identical grid cell in
    consecutive frames spuriously removes real contours that quantise one cell
    differently, while no support at all admits isolated flicker.  A 3x3
    tolerance is the smallest symmetric compromise and remains entirely
    trajectory-derived.
    """
    if candidate.shape[1] <= 1 or radius < 1:
        return _persistent_boundary(candidate, forward_flow)
    batch, pairs, height, width = candidate.shape
    dilated = F.max_pool2d(
        candidate.float().reshape(-1, 1, height, width), 2 * radius + 1,
        stride=1, padding=radius,
    ).reshape(batch, pairs, height, width).bool()
    transport = forward_flow[:, 1:]
    prior_arrival = _forward_coverage(transport, dilated[:, :-1]) > 0
    future_at_current = _sample_at_flow(dilated[:, 1:], transport)
    support = torch.zeros_like(candidate)
    support[:, 1:] |= prior_arrival
    support[:, :-1] |= future_at_current
    return candidate & support


@torch.no_grad()
def predictability_topology_labels(image_tokens: torch.Tensor) -> torch.Tensor:
    """Partition post-global tokens by temporal predictability topology.

    This is the runtime form of the audited decomposition, using no semantic
    model, RGB flow, depth/pose prediction, cached labels or label quota.
    ``unknown`` is a robust tail of failed temporal prediction. ``sensitive``
    is an *interface*: a persistent appearance boundary whose neighbouring
    tokens have different temporal predictability.  Thus a coherent moving
    object interior remains stable, and a static texture edge cannot become
    sensitive only because it is visually sharp.
    """
    batch, frames, height, width, _ = image_tokens.shape
    labels = torch.full((batch, frames, height, width), 2, device=image_tokens.device, dtype=torch.long)
    if frames == 1:
        return labels

    cfg = TEMPORAL_LABEL_CONFIG
    evidence = build_deep_track_graph_evidence(image_tokens)

    # Temporal surprise is a continuous field, not merely a binary visibility
    # test.  The robust upper tail prevents repeated/low-texture descriptors
    # from turning an entire frame into false novel content.
    confidence_loss = 1.0 - evidence.backward_score.float()
    finite = torch.isfinite(confidence_loss)
    surprise_high = _clip_robust_upper(confidence_loss, finite, scale=cfg.topology_unknown_mad_scale)
    unknown_tail = finite & (confidence_loss >= surprise_high.reshape(batch, 1, 1, 1))
    # Do not promote every strict cycle/coverage failure to unknown.  At this
    # coarse local-match scale, that would classify ordinary low-texture floor
    # patches as novel content. The decision remains the conservative
    # distributional tail of the continuous predictability field.
    raw_unknown = unknown_tail
    unknown = _spatially_supported(raw_unknown, minimum=cfg.topology_min_spatial_support)

    # A sensitive role is a boundary between different temporal states.  It
    # must satisfy both an appearance discontinuity and a local transition of
    # temporal surprise.  This rejects high-contrast but temporally uniform
    # static contours, and avoids treating a moving object's coherent interior
    # as an edge.
    appearance = _feature_discontinuity(image_tokens)[:, 1:]
    transition = _scalar_spatial_gradient(confidence_loss)
    # Confidence itself remains continuous, so ``visible`` is not a hard gate
    # for a boundary.  Its strict version is too brittle for small local
    # search windows under smooth camera motion.
    eligible = ~unknown
    appearance_high = _clip_robust_upper(appearance, eligible, scale=cfg.topology_boundary_mad_scale)
    transition_high = _clip_robust_upper(transition, eligible, scale=cfg.topology_transition_mad_scale)
    candidate = eligible & (appearance >= appearance_high.reshape(batch, 1, 1, 1))
    candidate &= transition >= transition_high.reshape(batch, 1, 1, 1).clamp_min(cfg.topology_min_transition)
    # A static 3-D scene has parallax at depth edges, so affine residual alone
    # is insufficient.  A per-token epipolar hard gate is also too brittle at
    # patch resolution.  Instead, use epipolar geometry to decide whether the
    # *clip* contains sustained non-rigid motion: isolated mismatches in a
    # rigid camera sweep suppress all sensitive labels, whereas a moving actor
    # keeps the local boundary rule active.
    epipolar = _epipolar_residual(evidence.backward_flow)
    pair_median = epipolar.flatten(start_dim=-2).median(dim=-1).values
    sustained_nonrigid = (pair_median >= cfg.topology_clip_epipolar_pair_median).float().mean(dim=1)
    candidate &= (sustained_nonrigid >= cfg.topology_clip_nonrigid_min_fraction).reshape(batch, 1, 1, 1)
    candidate = _spatially_supported(candidate, minimum=cfg.topology_min_spatial_support)
    sensitive = _persistent_boundary_tolerant(
        candidate, evidence.forward_flow, radius=cfg.topology_temporal_support_radius,
    )
    labels[:, 1:] = torch.where(unknown, 2, torch.where(sensitive, 1, 0))
    return labels


@torch.no_grad()
def fuse_native_rgb_temporal_labels(native_labels: torch.Tensor, rgb_labels: torch.Tensor, *, radius: int = 1) -> torch.Tensor:
    """Use RGB motion only to extend a nearby native temporal boundary.

    Native labels provide a high-precision semantic-feature anchor.  The
    training-free RGB visibility/affine partition may add its motion boundary
    only within ``radius`` patch cells of that anchor; hence a display edge or
    other RGB-only response cannot become sensitive in isolation.  The native
    trajectory branch remains the sole authority for ``unknown``: RGB is a
    coarse, input-side boundary proposal and must not overwrite a native
    visible trajectory with a spurious occlusion label.
    """
    if native_labels.shape != rgb_labels.shape or native_labels.ndim != 4:
        raise ValueError("native and RGB temporal labels must share [B,T,H,W]")
    if radius < 0:
        raise ValueError("hybrid consensus radius must be non-negative")
    native_sensitive, rgb_sensitive = native_labels == 1, rgb_labels == 1
    unknown = native_labels == 2
    if radius:
        batch, frames, height, width = native_labels.shape
        nearby_native = F.max_pool2d(
            native_sensitive.float().reshape(-1, 1, height, width),
            2 * radius + 1, stride=1, padding=radius,
        ).reshape(batch, frames, height, width).bool()
    else:
        nearby_native = native_sensitive
    sensitive = (native_sensitive | (rgb_sensitive & nearby_native)) & ~unknown
    return torch.where(unknown, torch.full_like(native_labels, 2), torch.where(sensitive, torch.ones_like(native_labels), torch.zeros_like(native_labels)))


@torch.no_grad()
def _persistent_boundary(candidate: torch.Tensor, forward_flow: torch.Tensor) -> torch.Tensor:
    """Keep a boundary only if its token tracks support it in a neighbour frame.

    ``candidate[:, i]`` belongs to target frame ``i + 1``; therefore the
    transport from candidate ``i`` to ``i + 1`` is forward-flow pair ``i+1``.
    This off-by-one is essential: using pair ``i`` would compare maps in two
    different frame coordinate systems.
    """
    if candidate.shape[1] <= 1 or not TEMPORAL_LABEL_CONFIG.temporal_boundary_support:
        return candidate
    transport = forward_flow[:, 1:]
    prior_arrival = _forward_coverage(transport, candidate[:, :-1]) > 0
    future_at_current = _sample_at_flow(candidate[:, 1:], transport)
    support = torch.zeros_like(candidate)
    support[:, 1:] |= prior_arrival
    support[:, :-1] |= future_at_current
    return candidate & support


@torch.no_grad()
def _robust_bounds(values: torch.Tensor, valid: torch.Tensor, *, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame median/MAD lower and upper bounds for nonstationary videos."""
    flat = values.reshape(values.shape[0], values.shape[1], -1)
    valid_flat = valid.reshape(valid.shape[0], valid.shape[1], -1) & torch.isfinite(flat)
    nan = torch.full_like(flat, float("nan"))
    masked = torch.where(valid_flat, flat, nan)
    median = torch.nanmedian(masked, dim=-1, keepdim=True).values.nan_to_num(0.0)
    mad = torch.nanmedian((masked - median).abs(), dim=-1, keepdim=True).values.nan_to_num(0.0).clamp_min(1e-4)
    return (median - scale * mad).reshape(*values.shape[:2], 1, 1), (median + scale * mad).reshape(*values.shape[:2], 1, 1)


def _track_quality(score: torch.Tensor, margin: torch.Tensor) -> torch.Tensor:
    cfg = TEMPORAL_LABEL_CONFIG
    eligible = torch.isfinite(score) & torch.isfinite(margin) & (score >= cfg.min_score) & (margin >= cfg.min_margin)
    score_low, _ = _robust_bounds(score, eligible, scale=2.5)
    margin_low, _ = _robust_bounds(margin, eligible, scale=2.0)
    return eligible & (score >= score_low.clamp_min(cfg.min_score)) & (margin >= margin_low.clamp_min(cfg.min_margin))


def _sample_at_flow(values: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    """Gather a scalar map at in-bounds integer track endpoints."""
    b, pairs, h, w = values.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    x = (xx + flow[..., 0].long()).clamp(0, w - 1)
    y = (yy + flow[..., 1].long()).clamp(0, h - 1)
    return values.reshape(b, pairs, -1).gather(2, (y * w + x).reshape(b, pairs, -1)).reshape(b, pairs, h, w)


def _forward_coverage(flow: torch.Tensor, trusted: torch.Tensor) -> torch.Tensor:
    """Count trusted previous-frame tracks arriving at each current patch.

    Zero means no established predecessor; >1 is an ambiguous collision.
    Neither is asserted to be a geometrically proven disocclusion.
    """
    b, pairs, h, w, _ = flow.shape
    yy, xx = torch.meshgrid(torch.arange(h, device=flow.device), torch.arange(w, device=flow.device), indexing="ij")
    x, y = xx + flow[..., 0].long(), yy + flow[..., 1].long()
    valid = trusted & (x >= 0) & (x < w) & (y >= 0) & (y < h)
    index = (y.clamp(0, h - 1) * w + x.clamp(0, w - 1)).reshape(b, pairs, -1)
    coverage = torch.zeros((b, pairs, h * w), device=flow.device, dtype=torch.int32)
    coverage.scatter_add_(2, index, valid.reshape(b, pairs, -1).to(torch.int32))
    return coverage.reshape(b, pairs, h, w)


@torch.no_grad()
def build_deep_track_graph_evidence(image_tokens: torch.Tensor) -> DeepTrackGraphEvidence:
    """Build expensive unthresholded deep-token associations once.

    Threshold selection is intentionally separated from association building so
    an offline pseudo-reference can audit/calibrate fixed rules without a
    second Omega forward or re-running the local token matcher.
    """
    if image_tokens.ndim != 5:
        raise ValueError(f"Expected [B,T,H,W,C] tokens, got {tuple(image_tokens.shape)}")
    if image_tokens.shape[1] < 2:
        raise ValueError("Deep-track evidence requires at least two frames")
    forward_score, forward_margin, forward_flow = _pairwise_local_tracks(image_tokens[:, :-1], image_tokens[:, 1:])
    backward_score, backward_margin, backward_flow = _pairwise_local_tracks(image_tokens[:, 1:], image_tokens[:, :-1])
    return DeepTrackGraphEvidence(
        forward_score=forward_score, forward_margin=forward_margin, forward_flow=forward_flow,
        backward_score=backward_score, backward_margin=backward_margin, backward_flow=backward_flow,
        feature_jump=_feature_discontinuity(image_tokens[:, 1:]),
    )


@torch.no_grad()
def deep_track_graph_labels(
    image_tokens: torch.Tensor,
    *,
    evidence: DeepTrackGraphEvidence | None = None,
    native_image_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Classify token roles from the current layer's deep, bidirectional tracks.

    This is deliberately a *single-forward* operation.  It takes current
    forward tensors only: no RGB flow, predicted depth/pose, external model or
    preliminary model pass is used.  When the aggregator supplies its native
    patch view, the sharper native DINO contour/track rule is used; otherwise
    the post-global deep-track graph fallback is retained for compatibility.
    """
    b, frames, height, width, _ = image_tokens.shape
    if native_image_tokens is not None:
        if native_image_tokens.shape[:4] != (b, frames, height, width):
            raise ValueError("native_image_tokens must match image-token batch/time/grid dimensions")
        return _native_patch_labels(native_image_tokens)
    labels = torch.full((b, frames, height, width), 2, device=image_tokens.device, dtype=torch.long)
    if frames == 1:
        return labels
    evidence = build_deep_track_graph_evidence(image_tokens) if evidence is None else evidence
    forward_score, forward_margin, forward_flow = evidence.forward_score, evidence.forward_margin, evidence.forward_flow
    backward_score, backward_margin, backward_flow = evidence.backward_score, evidence.backward_margin, evidence.backward_flow
    forward_closure, forward_in_bounds = _round_trip_error(forward_flow, backward_flow)
    backward_closure, backward_in_bounds = _round_trip_error(backward_flow, forward_flow)
    forward_quality = _track_quality(forward_score, forward_margin)
    backward_quality = _track_quality(backward_score, backward_margin)
    cfg = TEMPORAL_LABEL_CONFIG

    def closed_tracks(error: torch.Tensor, bounds: torch.Tensor, quality: torch.Tensor) -> torch.Tensor:
        eligible = bounds & quality & torch.isfinite(error) & (error <= cfg.max_cycle_error)
        _, upper = _robust_bounds(error, eligible, scale=2.5)
        return eligible & (error <= upper.clamp(0.0, cfg.max_cycle_error))

    trusted_forward = closed_tracks(forward_closure, forward_in_bounds, forward_quality)
    trusted_forward &= _sample_at_flow(backward_quality, forward_flow)
    coverage = _forward_coverage(forward_flow, trusted_forward)
    visible = closed_tracks(backward_closure, backward_in_bounds, backward_quality)
    visible &= _sample_at_flow(trusted_forward, backward_flow) & (coverage == 1)

    motion_jump = _flow_discontinuity(backward_flow, visible)
    _, motion_high = _robust_bounds(motion_jump, visible, scale=2.0)
    motion_high = motion_high.clamp(cfg.min_boundary_jump, cfg.max_boundary_threshold)
    feature_jump = evidence.feature_jump
    _, feature_high = _robust_bounds(feature_jump, visible, scale=cfg.feature_boundary_mad_scale)
    interior_jump = _minimum_neighbour_motion_jump(backward_flow, visible)
    coherent = torch.isfinite(interior_jump) & (interior_jump <= cfg.max_interior_motion_jump)
    candidate = visible & coherent & (motion_jump >= motion_high) & (feature_jump >= feature_high)
    sensitive = _persistent_boundary(candidate, forward_flow)
    labels[:, 1:] = torch.where(~visible, 2, torch.where(sensitive, 1, 0))
    return labels


@torch.no_grad()
def temporal_labels(
    tokens: torch.Tensor,
    *,
    frames: int,
    tokens_per_frame: int,
    patch_token_start: int,
    patch_hw: tuple[int, int],
    backend: str = "reference",
    native_image_tokens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Classify tokens by visibility, trajectory coherence and motion boundary.

    Class 0 (stable) has a reliable, locally coherent bidirectional track.
    Class 1 (sensitive) remains visible but sits on a motion discontinuity.
    Class 2 (unknown) lacks trustworthy visibility evidence: possible newly
    visible areas, ambiguous texture, collisions or failed local matching.
    Features alone do not prove physical disocclusion or object semantics.
    The protected reference frame has no predecessor and is marked unknown.
    """
    backend = resolve_temporal_label_backend(backend)
    b, total, c = tokens.shape
    if b != 1:
        raise ValueError("trajectory-aware temporal merging currently requires batch size one")
    if frames < 1 or min(patch_hw) < 1 or total != frames * tokens_per_frame or patch_token_start + math.prod(patch_hw) != tokens_per_frame:
        raise ValueError("Invalid Omega temporal-label frame/patch/special-token layout")
    h, w = patch_hw
    # Slice/view avoids the former full-sequence advanced-index feature copy.
    image_tokens = tokens.reshape(b, frames, tokens_per_frame, c)[:, :, patch_token_start:].reshape(b, frames, h, w, c)
    if backend == "deep_track_graph":
        return deep_track_graph_labels(image_tokens, native_image_tokens=native_image_tokens)
    if backend == "adaptive_native":
        # Classify the immutable native patch view. This is causal with
        # respect to the encoder and is not post-global label refinement.
        return adaptive_native_patch_labels(native_image_tokens if native_image_tokens is not None else image_tokens)
    if backend == "predictability_topology":
        # Unlike the native branches, this rule intentionally consumes the
        # first full-resolution post-global representation: its temporal
        # predictability field is the object under study.  The resulting map
        # is then cached for later compacted global blocks.
        return predictability_topology_labels(image_tokens)
    labels = torch.full((b, frames, h, w), 2, device=tokens.device, dtype=torch.long)
    if frames == 1:
        return labels

    # Pair i stores i -> i+1; reverse pair i stores i+1 -> i.  The latter is
    # the causal visibility test for current-frame disocclusions.
    forward_score, forward_margin, forward_flow = _pairwise_local_tracks(image_tokens[:, :-1], image_tokens[:, 1:])
    backward_score, backward_margin, backward_flow = _pairwise_local_tracks(image_tokens[:, 1:], image_tokens[:, :-1])
    forward_closure, forward_in_bounds = _round_trip_error(forward_flow, backward_flow)
    backward_closure, backward_in_bounds = _round_trip_error(backward_flow, forward_flow)
    forward_quality = _track_quality(forward_score, forward_margin)
    backward_quality = _track_quality(backward_score, backward_margin)
    cfg = TEMPORAL_LABEL_CONFIG

    def closed_tracks(error, bounds, quality):
        eligible = bounds & quality & torch.isfinite(error) & (error <= cfg.max_cycle_error)
        _, upper = _robust_bounds(error, eligible, scale=2.5)
        return eligible & (error <= upper.clamp(0.0, cfg.max_cycle_error))

    trusted_forward = closed_tracks(forward_closure, forward_in_bounds, forward_quality)
    trusted_forward &= _sample_at_flow(backward_quality, forward_flow)
    coverage = _forward_coverage(forward_flow, trusted_forward)
    visible = closed_tracks(backward_closure, backward_in_bounds, backward_quality)
    visible &= _sample_at_flow(trusted_forward, backward_flow) & (coverage == 1)
    boundary = _flow_discontinuity(backward_flow, visible)
    _, boundary_high = _robust_bounds(boundary, visible, scale=2.0)
    boundary_high = boundary_high.clamp(cfg.min_boundary_jump, cfg.max_boundary_threshold)
    sensitive = visible & (boundary >= boundary_high)
    labels[:, 1:] = torch.where(~visible, 2, torch.where(sensitive, 1, 0))
    return labels


@torch.no_grad()
def build_temporal_plan(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    frames: int,
    tokens_per_frame: int,
    patch_token_start: int,
    patch_hw: tuple[int, int],
    dynamic_unknown_keep_ratio: float = 0.65,
    unknown_keep_ratio: float | None = None,
    static_keep_ratio: float = 0.07,
    periodic_protect_ratio: float = 0.0,
) -> TokenPlan:
    """Class-local global similarity merge, NOT track-constrained merging.

    Stable / sensitive / unknown quotas remain 7% / 65% / 65% by default.
    ``unknown_keep_ratio`` can independently tighten unknown/disoccluded
    regions without changing the sensitive-boundary quota.
    no extra periodic protection is enabled. Unknown is an uncertainty class,
    NOT a shared object/track identity. Same-class merging is a computational
    approximation and cannot be claimed to establish physical correspondence.
    """
    if tokens.shape[0] != 1:
        raise ValueError("Any4D-equivalent temporal merging currently requires batch size one")
    unknown_keep_ratio = dynamic_unknown_keep_ratio if unknown_keep_ratio is None else unknown_keep_ratio
    if not (0.0 < dynamic_unknown_keep_ratio <= 1.0 and 0.0 < unknown_keep_ratio <= 1.0 and 0.0 < static_keep_ratio <= 1.0):
        raise ValueError("temporal keep ratios must lie in (0, 1]")
    if not 0.0 <= periodic_protect_ratio <= 1.0:
        raise ValueError("temporal periodic protect ratio must lie in [0, 1]")
    b, n, _ = tokens.shape
    if n != frames * tokens_per_frame or labels.shape != (1, frames, *patch_hw) or patch_token_start + math.prod(patch_hw) != tokens_per_frame:
        raise ValueError("temporal labels/layout do not match Omega tokens")

    patch_ids = _patch_ids(frames, tokens_per_frame, patch_token_start, tokens.device)
    image_ids = patch_ids.flatten()
    cls = labels.reshape(-1)
    special_mask = torch.ones(n, device=tokens.device, dtype=torch.bool)
    special_mask[image_ids] = False
    keep = [torch.where(special_mask)[0]]
    protected = torch.zeros(n, device=tokens.device, dtype=torch.bool)
    protected[:tokens_per_frame] = True  # full first frame, precisely as Any4D
    count = int(n * periodic_protect_ratio)
    if count:
        protected[torch.arange(0, n, max(1, n // count), device=tokens.device)[:count]] = True
    pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

    for category, ratio in ((0, static_keep_ratio), (1, dynamic_unknown_keep_ratio), (2, unknown_keep_ratio)):
        category_ids = image_ids[cls == category]
        if not category_ids.numel():
            continue
        selected_count = max(1, min(category_ids.numel(), int(category_ids.numel() * ratio + 0.5)))
        selected_positions = (torch.arange(selected_count, device=tokens.device) * category_ids.numel()) // selected_count
        is_destination = torch.zeros(category_ids.numel(), device=tokens.device, dtype=torch.bool)
        is_destination[selected_positions] = True
        is_destination |= protected[category_ids]
        destination = category_ids[is_destination]
        source = category_ids[~is_destination]
        keep.append(destination)
        if not source.numel():
            continue
        # The exact Any4D global, class-local BF16 matching loop.
        best = torch.full((b, source.numel()), float("-inf"), device=tokens.device)
        match = torch.zeros((b, source.numel()), device=tokens.device, dtype=torch.long)
        for dstart in range(0, destination.numel(), TEMPORAL_MATCH_DESTINATION_CHUNK):
            dend = min(destination.numel(), dstart + TEMPORAL_MATCH_DESTINATION_CHUNK)
            dest_metric_t = F.normalize(tokens[:, destination[dstart:dend]].float(), dim=-1).to(torch.bfloat16).transpose(1, 2)
            for start in range(0, source.numel(), TEMPORAL_MATCH_SOURCE_CHUNK):
                end = min(source.numel(), start + TEMPORAL_MATCH_SOURCE_CHUNK)
                source_metric = F.normalize(tokens[:, source[start:end]].float(), dim=-1).to(torch.bfloat16)
                values, indices = torch.bmm(source_metric, dest_metric_t).max(-1)
                update = values > best[:, start:end]
                best[:, start:end] = torch.where(update, values, best[:, start:end])
                match[:, start:end] = torch.where(update, indices + dstart, match[:, start:end])
        pairs.append((source, destination[match]))

    kept = torch.cat(keep).sort().values.expand(b, -1)
    inverse = torch.full((b, n), -1, device=tokens.device, dtype=torch.long)
    inverse.scatter_(1, kept, torch.arange(kept.shape[1], device=tokens.device).expand(b, -1))
    for source, destination in pairs:
        inverse.scatter_(1, source.expand(b, -1), inverse.gather(1, destination))
    return TokenPlan(kept, inverse, n)
