# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from dataclasses import replace

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.utils import named_apply
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer, init_weights_vit
from vggt_omega.models.token_merging import (
    build_fastvggt_plan,
    build_operator_fastvggt_plan,
    build_reference_anchor_plan,
    build_reference_anchor_cluster_plan,
    build_recursive_local_2x2_plan,
    build_temporal_plan,
    fuse_native_rgb_temporal_labels,
    resolve_temporal_label_backend,
    temporal_labels,
)
from vggt_omega.models.fast_affine import build_affine_warp_plan_fast as build_affine_warp_plan
from vggt_omega.models.sparse_convex import (
    build_adaptive_basis_plan,
    build_hash_convex_plan,
    build_kmeans_basis_plan,
    build_sparse_convex_plan,
    build_ttt_protected_uniform_convex_plan,
    build_ttt_partitioned_convex_plan,
)


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
# Global-layer Q schedule for the sparse-convex method.  Layers 8, 10, 11,
# and 12 are the empirically sensitive query layers and therefore remain
# KV-only by default; all other global layers compact Q/K/V.
_DEFAULT_SPARSE_CONVEX_QKV_LAYERS = tuple(
    layer for layer in range(19) if layer not in {8, 10, 11, 12}
)


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(
            patch_size=patch_size, embed_dim=embed_dim, use_checkpoint=use_checkpoint
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.cached_layer_indices = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        self.fastvggt_enabled = False
        self.fastvggt_merge_ratio = 0.9
        self.fastvggt_protect_ratio = 0.10
        self.fastvggt_operator_signature = False
        self.fastvggt_operator_probe_count = 8
        self.fastvggt_last_plan_stats: list[dict[str, float | int]] = []
        self.sparse_convex_enabled = False
        self.sparse_convex_threshold = .90
        self.sparse_convex_planner = "fixed_topology"
        self.sparse_convex_max_support = 9
        self.sparse_convex_coefficient_limit = 1.0
        self.sparse_convex_basis_ratio = .05
        self.sparse_convex_hash_bits = 12
        self.sparse_convex_4d_geometry: torch.Tensor | None = None
        self.sparse_convex_4d_global_layers: frozenset[int] = frozenset(range(19))
        self.sparse_convex_4d_selection_weight = .01
        self.sparse_convex_4d_support_weight = .01
        self.sparse_convex_4d_guard_tolerance: float | None = 0.0
        self.sparse_convex_4d_metadata: dict[str, object] = {}
        self.sparse_convex_ttt_labels: torch.Tensor | None = None
        self.sparse_convex_ttt_sensitivity: torch.Tensor | None = None
        self.sparse_convex_ttt_fixed_dynamic: torch.Tensor | None = None
        self.sparse_convex_ttt_top1_pruning = True
        self.sparse_convex_ttt_sensitive_event_ratio = .02
        self.sparse_convex_dense_global_layers: frozenset[int] = frozenset()
        self._sparse_convex_reuse_plan = None
        self.sparse_convex_qkv_global_blocks = 15
        # Optional explicit per-layer schedules.  ``None`` keeps the legacy
        # prefix rule; a set replaces it with an arbitrary global-ordinal mask.
        self.sparse_convex_qkv_global_layers: frozenset[int] | None = None
        self.sparse_convex_physical_key_layers: frozenset[int] = frozenset()
        self.sparse_convex_last_plan_stats: list[dict[str, float | int]] = []
        # Optional immutable patch-encoder output for flow-free TTT.  The
        # preparation stage needs these exact tokens for correspondence; the
        # main forward reuses them instead of evaluating patch_embed twice.
        self._input_patch_tokens_cache: torch.Tensor | None = None
        self.sampled_update_operator_enabled = False
        self.sampled_update_operator_stats: list[dict[str, float | int]] = []
        # ``None`` preserves released FastVGGT semantics (all global blocks).
        # A finite prefix is diagnostic/hybrid infrastructure: it lets an
        # early coarse merger be audited without changing later blocks.
        self.fastvggt_global_prefix_blocks: int | None = None
        self.reference_anchor_merge_enabled = False
        self.reference_anchor_similarity_threshold = 0.95
        self.reference_anchor_frame_count = 1
        # Last-forward, Python-only accounting.  It never retains model
        # activations and is used solely to report the realised attachment
        # rate for reference-anchor experiments.
        self.reference_anchor_last_plan_stats: list[dict[str, float | int]] = []
        self.reference_anchor_cluster_merge_enabled = False
        self.reference_anchor_cluster_high_similarity_threshold = .9
        self.reference_anchor_cluster_medium_similarity_lower = .7
        self.reference_anchor_cluster_similarity_threshold = .9
        self.reference_anchor_cluster_residual_fastvggt = False
        self.reference_anchor_cluster_residual_keep_every = 10
        self.reference_anchor_cluster_residual_grid_position: int | None = None
        self.reference_anchor_cluster_residual_strong_clusters = False
        self.reference_anchor_cluster_residual_max_clusters_per_reference = 16
        self.reference_anchor_cluster_residual_cluster_similarity_threshold: float | None = None
        self.reference_anchor_cluster_last_plan_stats: list[dict[str, float | int]] = []
        self.recursive_local_merge_enabled = False
        self.recursive_local_similarity_threshold = .95
        self.recursive_local_dominant_corner_only = False
        self.recursive_local_last_plan_stats: list[dict[str, float | int]] = []
        self.affine_warp_merge_enabled = False
        self.affine_warp_similarity_threshold = .95
        self.affine_warp_anchor_grid = 6
        self.affine_warp_assignment_mode = "soft_soft"
        self.affine_warp_dense_special_kv = False
        self.affine_warp_dense_special_start_block: int | None = None
        self.affine_warp_pool_source_content = True
        self.affine_warp_qkv_similarity_threshold: float | None = None
        self.affine_warp_star_window = 2
        self.affine_warp_target_merge_ratio: float | None = None
        self.affine_warp_camera_importance_anchors = 8
        self.affine_warp_importance_policy = "camera_qk"
        self.affine_warp_temporal_importance_weight = 0.5
        self.affine_warp_last_plan_stats: list[dict[str, float | int]] = []
        self.temporal_merge_enabled = False
        self.temporal_dynamic_unknown_keep_ratio = 0.65
        self.temporal_unknown_keep_ratio = 0.65
        self.temporal_static_keep_ratio = 0.07
        # Default: only the reference frame and required special tokens are
        # protected. Periodic anchors are explicit ablation settings.
        self.temporal_periodic_protect_ratio = 0.0
        self.temporal_label_backend = "deep_track_graph"
        self.temporal_label_recorder = None
        self.feature_recorder = None
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def set_fastvggt(
        self, enabled: bool = True, merge_ratio: float = 0.9,
        protect_ratio: float = 0.10,
    ) -> None:
        if not 0.0 <= merge_ratio <= 1.0 or not 0.0 <= protect_ratio <= 1.0:
            raise ValueError("FastVGGT ratios must lie in [0, 1]")
        if enabled and (self.sparse_convex_enabled or self.temporal_merge_enabled
                        or self.reference_anchor_merge_enabled
                        or self.reference_anchor_cluster_merge_enabled
                        or self.recursive_local_merge_enabled
                        or self.affine_warp_merge_enabled):
            raise ValueError("FastVGGT, temporal merging, reference-anchor, and reference-cluster merging are mutually exclusive")
        self.fastvggt_enabled, self.fastvggt_merge_ratio, self.fastvggt_protect_ratio = enabled, merge_ratio, protect_ratio

    def set_input_patch_tokens_cache(self, tokens: torch.Tensor | None) -> None:
        """Install or clear an input-specific frozen patch-token cache."""
        if tokens is not None:
            if tokens.ndim != 3 or tokens.shape[-1] != self.camera_token.shape[-1]:
                raise ValueError("cached patch tokens must be [batch*frames,patches,embed_dim]")
            if not bool(torch.isfinite(tokens).all()):
                raise ValueError("cached patch tokens must be finite")
            tokens = tokens.detach().contiguous()
        self._input_patch_tokens_cache = tokens

    def set_sparse_convex(
        self, enabled: bool = True, threshold: float = .90, qkv_global_blocks: int = 15,
        qkv_global_layers: tuple[int, ...] | None = _DEFAULT_SPARSE_CONVEX_QKV_LAYERS,
        physical_key_layers: tuple[int, ...] = (),
        planner: str = "fixed_topology", max_support: int = 9,
        coefficient_limit: float = 1.0, basis_ratio: float = .05, hash_bits: int = 12,
        dense_global_layers: tuple[int, ...] = (),
    ) -> None:
        if not -1.0 <= threshold <= 1.000001:
            raise ValueError("sparse-convex threshold must lie in [-1,1]")
        global_count = sum(kind == "global" for kind in self.inter_frame_attention_types)
        if not 0 <= qkv_global_blocks <= global_count:
            raise ValueError("invalid sparse-convex QKV global-block count")
        if planner not in {"fixed_topology", "adaptive_convex", "adaptive_linear", "kmeans_convex", "kmeans_convex_fast", "kmeans_convex_fused", "kmeans_convex_fused_batched", "kmeans_convex_4d_fused_batched", "kmeans_convex_reuse", "hash_convex", "ttt_partitioned_convex", "ttt_protected_uniform_convex"}:
            raise ValueError("invalid sparse-convex planner")
        if max_support < 1 or coefficient_limit <= 0 or not 0 < basis_ratio < 1 or not 4 <= hash_bits <= 16:
            raise ValueError("adaptive support and coefficient limit must be positive")
        if planner in {"kmeans_convex_fused", "kmeans_convex_fused_batched", "kmeans_convex_4d_fused_batched", "ttt_partitioned_convex", "ttt_protected_uniform_convex"} and max_support != 4:
            raise ValueError("fused and TTT-partitioned convex planners require exactly four supports")
        for name, layers in (("qkv_global_layers", qkv_global_layers), ("physical_key_layers", physical_key_layers), ("dense_global_layers", dense_global_layers)):
            if layers is None:
                continue
            if any(not 0 <= int(layer) < global_count for layer in layers):
                raise ValueError(f"invalid sparse-convex {name}: {layers}")
        if enabled and (self.fastvggt_enabled or self.temporal_merge_enabled or self.reference_anchor_merge_enabled
                        or self.reference_anchor_cluster_merge_enabled or self.recursive_local_merge_enabled
                        or self.affine_warp_merge_enabled):
            raise ValueError("sparse-convex merging is mutually exclusive with other merge modes")
        self.sparse_convex_enabled = bool(enabled)
        self.sparse_convex_threshold = float(threshold)
        self.sparse_convex_planner = planner
        self.sparse_convex_max_support = int(max_support)
        self.sparse_convex_coefficient_limit = float(coefficient_limit)
        self.sparse_convex_basis_ratio = float(basis_ratio)
        self.sparse_convex_hash_bits = int(hash_bits)
        self.sparse_convex_qkv_global_blocks = int(qkv_global_blocks)
        self.sparse_convex_qkv_global_layers = (
            None if qkv_global_layers is None else frozenset(int(layer) for layer in qkv_global_layers)
        )
        self.sparse_convex_physical_key_layers = frozenset(int(layer) for layer in physical_key_layers)
        self.sparse_convex_dense_global_layers = frozenset(int(layer) for layer in dense_global_layers)
        if not enabled:
            self.sparse_convex_4d_geometry = None
            self.sparse_convex_ttt_labels = None
            self.sparse_convex_ttt_sensitivity = None
            self.sparse_convex_ttt_fixed_dynamic = None
            self.sparse_convex_4d_metadata = {}

    def set_sparse_convex_ttt_partition(
        self, labels: torch.Tensor | None, sensitivity: torch.Tensor | None,
    ) -> None:
        if (labels is None) != (sensitivity is None):
            raise ValueError("TTT labels and sensitivity must be provided together")
        if labels is not None and (labels.ndim != 1 or sensitivity.ndim != 1
                                   or labels.numel() != sensitivity.numel()):
            raise ValueError("TTT labels and sensitivity must be matching vectors")
        if labels is not None:
            if not bool(torch.isin(labels, labels.new_tensor((0, 1, 2))).all()):
                raise ValueError("TTT labels must contain only stable=0, sensitive=1, uncertain=2")
            if not bool(torch.isfinite(sensitivity).all()):
                raise ValueError("TTT sensitivity must be finite")
        self.sparse_convex_ttt_labels = None if labels is None else labels.detach().contiguous()
        self.sparse_convex_ttt_sensitivity = (
            None if sensitivity is None else sensitivity.detach().contiguous()
        )

    def set_sparse_convex_ttt_fixed_dynamic(
        self, mask: torch.Tensor | None,
    ) -> None:
        if mask is not None and (mask.ndim != 1 or mask.dtype != torch.bool):
            raise ValueError("fixed dynamic basis must be a flat boolean mask")
        self.sparse_convex_ttt_fixed_dynamic = (
            None if mask is None else mask.detach().contiguous()
        )

    def set_sparse_convex_ttt_top1_pruning(self, enabled: bool = True) -> None:
        """Enable the protected skeleton's basis-internal Top-1 pruning."""
        self.sparse_convex_ttt_top1_pruning = bool(enabled)

    def set_sparse_convex_ttt_sensitive_event_ratio(self, ratio: float = .02) -> None:
        """Set the no-pruning TTT residual-event budget as a patch ratio."""
        if not 0.0 <= ratio < 1.0:
            raise ValueError("TTT sensitive-event ratio must lie in [0,1)")
        self.sparse_convex_ttt_sensitive_event_ratio = float(ratio)

    def set_sparse_convex_4d(
        self,
        geometry: torch.Tensor | None,
        *,
        global_layers: tuple[int, ...] = tuple(range(19)),
        selection_weight: float = .01,
        support_weight: float = .01,
        guard_tolerance: float | None = 0.0,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Attach a patch-aligned predicted-4D pilot descriptor to the planner."""
        global_count = sum(kind == "global" for kind in self.inter_frame_attention_types)
        if (selection_weight < 0 or support_weight < 0
                or (guard_tolerance is not None and guard_tolerance < 0)):
            raise ValueError("sparse-convex 4-D weights must be nonnegative")
        if any(not 0 <= int(layer) < global_count for layer in global_layers):
            raise ValueError(f"invalid sparse-convex 4-D global layers: {global_layers}")
        if geometry is not None and (geometry.ndim != 2 or geometry.shape[-1] != 4):
            raise ValueError(f"4-D geometry must be [patch_tokens,4], got {tuple(geometry.shape)}")
        if geometry is not None and not bool(torch.isfinite(geometry).all()):
            raise ValueError("4-D geometry must be finite")
        self.sparse_convex_4d_geometry = None if geometry is None else geometry.detach().contiguous()
        self.sparse_convex_4d_global_layers = frozenset(int(layer) for layer in global_layers)
        self.sparse_convex_4d_selection_weight = float(selection_weight)
        self.sparse_convex_4d_support_weight = float(support_weight)
        self.sparse_convex_4d_guard_tolerance = (
            None if guard_tolerance is None else float(guard_tolerance)
        )
        self.sparse_convex_4d_metadata = dict(metadata or {})

    def set_sampled_update_operator(
        self, enabled: bool, *, frames: int, ratio: float = .10, rank: int = 128,
        ridge: float = 1e-3, pca_tokens: int = 8192, capture_matrices: bool = False,
        exact_global_layers: frozenset[int] = frozenset(), sampling_mode: str = "frame",
    ) -> None:
        """Predict held-out patch-query updates from exact sampled-frame queries."""
        if enabled and not (0.0 < ratio < 1.0 and frames >= 3 and rank >= 1):
            raise ValueError("sampled update operator requires ratio in (0,1), frames>=3, rank>=1")
        self.sampled_update_operator_enabled = bool(enabled)
        self.sampled_update_operator_stats = []
        self.sampled_update_operator_matrices = []
        ordinal = 0
        for index, kind in enumerate(self.inter_frame_attention_types):
            if kind != "global":
                continue
            attention = self.inter_frame_blocks[index].attn
            attention.sampled_update_operator_config = ({
                "frames": int(frames), "patch_start": int(self.patch_token_start),
                "ratio": float(ratio), "rank": int(rank), "ridge": float(ridge),
                "sampling_mode": str(sampling_mode),
                "pca_tokens": int(pca_tokens), "stats": self.sampled_update_operator_stats,
                "matrix_capture": self.sampled_update_operator_matrices if capture_matrices else None,
                "global_ordinal": ordinal,
            } if enabled and ordinal not in exact_global_layers else None)
            ordinal += 1

    def set_dense_update_diagnostic(
        self, *, frames: int, ridge: float = 1e-3, references: list[dict] | None = None,
    ) -> None:
        """Fit affine maps to exact baseline attention updates as a side channel."""
        self.dense_update_diagnostic_matrices = []
        reference_by_layer = ({int(row["global_ordinal"]): row for row in references}
                              if references is not None else {})
        ordinal = 0
        for index, kind in enumerate(self.inter_frame_attention_types):
            if kind != "global":
                continue
            self.inter_frame_blocks[index].attn.dense_update_diagnostic_config = {
                "frames": int(frames), "patch_start": int(self.patch_token_start),
                "ridge": float(ridge), "global_ordinal": ordinal,
                "capture": self.dense_update_diagnostic_matrices,
                "reference": reference_by_layer.get(ordinal),
            }
            ordinal += 1

    def set_baseline_trajectory_probe(self, *, frames: int, probe_count: int = 8192) -> None:
        """Capture fixed patch-token states and exact updates from baseline."""
        self.baseline_trajectory_probes = []
        ordinal = 0
        for index, kind in enumerate(self.inter_frame_attention_types):
            if kind != "global":
                continue
            self.inter_frame_blocks[index].attn.baseline_trajectory_probe_config = {
                "frames": int(frames), "patch_start": int(self.patch_token_start),
                "probe_count": int(probe_count), "global_ordinal": ordinal,
                "capture": self.baseline_trajectory_probes,
            }
            ordinal += 1

    def set_sampled_trajectory_diagnostic(self, references: list[dict]) -> None:
        """Compare sampled trajectory with corresponding baseline probes."""
        self.sampled_trajectory_diagnostics = []
        reference_by_layer = {int(row["global_ordinal"]): row for row in references}
        ordinal = 0
        for index, kind in enumerate(self.inter_frame_attention_types):
            if kind != "global":
                continue
            config = self.inter_frame_blocks[index].attn.sampled_update_operator_config
            if config is None:
                raise ValueError("trajectory comparison requires every global layer to use sampled updates")
            config["trajectory_diagnostic"] = {
                "reference": reference_by_layer[ordinal],
                "capture": self.sampled_trajectory_diagnostics,
            }
            ordinal += 1

    def set_fixed_update_operator(self, *, frames: int, matrices: list[dict]) -> None:
        """Replace patch-query attention updates with pre-fitted affine maps."""
        matrix_by_layer = {int(row["global_ordinal"]): row for row in matrices}
        ordinal = 0
        device = self.camera_token.device
        for index, kind in enumerate(self.inter_frame_attention_types):
            if kind != "global":
                continue
            row = matrix_by_layer[ordinal]
            self.inter_frame_blocks[index].attn.fixed_update_operator_config = {
                "frames": int(frames), "patch_start": int(self.patch_token_start),
                "global_ordinal": ordinal,
                "weight": row["weight"].to(device=device, dtype=torch.float32),
                "bias": row["bias"].to(device=device, dtype=torch.float32),
            }
            ordinal += 1

    def set_fastvggt_operator_signature(
        self, enabled: bool = True, probe_count: int = 8,
    ) -> None:
        """Replace only FastVGGT representative selection and assignment."""
        if probe_count < 2:
            raise ValueError("FastVGGT operator-signature probe count must be at least two")
        self.fastvggt_operator_signature = bool(enabled)
        self.fastvggt_operator_probe_count = int(probe_count)

    def set_reference_anchor_merge(
        self,
        enabled: bool = True,
        similarity_threshold: float = .95,
        reference_frame_count: int = 1,
    ) -> None:
        """Enable high-similarity attachment to uniformly spaced full references."""
        if not -1.0 <= similarity_threshold <= 1.000001:
            raise ValueError("reference-anchor similarity threshold must lie in [-1, 1.000001]")
        if reference_frame_count < 1:
            raise ValueError("reference-anchor frame count must be positive")
        if enabled and (self.fastvggt_enabled or self.temporal_merge_enabled or self.reference_anchor_cluster_merge_enabled or self.recursive_local_merge_enabled or self.affine_warp_merge_enabled):
            raise ValueError("FastVGGT, temporal merging, reference-anchor, and reference-cluster merging are mutually exclusive")
        self.reference_anchor_merge_enabled = enabled
        self.reference_anchor_similarity_threshold = similarity_threshold
        self.reference_anchor_frame_count = reference_frame_count

    def set_recursive_local_merge(
        self,
        enabled: bool = True,
        similarity_threshold: float = .95,
        dominant_corner_only: bool = False,
    ) -> None:
        """Enable adjacent-frame recursive virtual 2x2 compaction."""
        if not -1.0 <= similarity_threshold <= 1.000001:
            raise ValueError("recursive-local similarity threshold must lie in [-1, 1.000001]")
        if enabled and (self.fastvggt_enabled or self.temporal_merge_enabled or self.reference_anchor_merge_enabled or self.reference_anchor_cluster_merge_enabled or self.affine_warp_merge_enabled):
            raise ValueError("recursive-local merging is mutually exclusive with other merge modes")
        self.recursive_local_merge_enabled = enabled
        self.recursive_local_similarity_threshold = similarity_threshold
        self.recursive_local_dominant_corner_only = bool(dominant_corner_only)

    def set_affine_warp_merge(
        self,
        enabled: bool = True,
        similarity_threshold: float = .95,
        anchor_grid: int = 6,
        assignment_mode: str = "soft_soft",
        dense_special_kv: bool = False,
        dense_special_start_block: int | None = None,
        pool_source_content: bool = True,
        qkv_similarity_threshold: float | None = None,
        star_window: int = 2,
        target_merge_ratio: float | None = None,
        camera_importance_anchors: int = 8,
        importance_policy: str = "camera_qk",
        temporal_importance_weight: float = 0.5,
    ) -> None:
        """Enable bounded anchor-window affine/bilinear Q/K/V merging."""
        if not -1.0 <= similarity_threshold <= 1.000001:
            raise ValueError("affine-warp similarity_threshold must lie in [-1, 1.000001]")
        if anchor_grid < 2:
            raise ValueError("affine-warp anchor_grid must be at least two")
        if assignment_mode not in {"soft_soft", "hard_hard"}:
            raise ValueError("affine-warp assignment_mode must be soft_soft or hard_hard")
        if qkv_similarity_threshold is not None and not -1.0 <= qkv_similarity_threshold <= 1.000001:
            raise ValueError("affine-warp qkv_similarity_threshold must lie in [-1, 1.000001]")
        if dense_special_start_block is not None and not 0 <= dense_special_start_block < self.depth:
            raise ValueError("affine-warp dense special start block is out of range")
        if star_window < 2:
            raise ValueError("affine-warp star window must be at least two")
        if target_merge_ratio is not None and not 0.0 < target_merge_ratio < 1.0:
            raise ValueError("affine-warp target_merge_ratio must lie in (0, 1)")
        if camera_importance_anchors < 1:
            raise ValueError("affine-warp camera_importance_anchors must be positive")
        if importance_policy not in {"camera_qk", "reference_special_influence", "hybrid_special_influence"}:
            raise ValueError("unknown affine-warp importance policy")
        if temporal_importance_weight < 0.0:
            raise ValueError("affine-warp temporal importance weight must be non-negative")
        if enabled and (self.fastvggt_enabled or self.temporal_merge_enabled or self.reference_anchor_merge_enabled or self.reference_anchor_cluster_merge_enabled or self.recursive_local_merge_enabled):
            raise ValueError("affine-warp merging is mutually exclusive with other merge modes")
        self.affine_warp_merge_enabled = enabled
        self.affine_warp_similarity_threshold = similarity_threshold
        self.affine_warp_anchor_grid = anchor_grid
        self.affine_warp_assignment_mode = assignment_mode
        self.affine_warp_dense_special_kv = bool(dense_special_kv)
        self.affine_warp_dense_special_start_block = dense_special_start_block
        self.affine_warp_pool_source_content = bool(pool_source_content)
        self.affine_warp_qkv_similarity_threshold = qkv_similarity_threshold
        self.affine_warp_star_window = int(star_window)
        self.affine_warp_target_merge_ratio = target_merge_ratio
        self.affine_warp_camera_importance_anchors = int(camera_importance_anchors)
        self.affine_warp_importance_policy = importance_policy
        self.affine_warp_temporal_importance_weight = float(temporal_importance_weight)

    def set_reference_anchor_cluster_merge(
        self,
        enabled: bool = True,
        high_similarity_threshold: float = .9,
        medium_similarity_lower: float = .7,
        cluster_similarity_threshold: float = .9,
        residual_fastvggt: bool = False,
        residual_keep_every: int = 10,
        residual_grid_position: int | None = None,
        residual_strong_clusters: bool = False,
        residual_max_clusters_per_reference: int = 16,
        residual_cluster_similarity_threshold: float | None = None,
    ) -> None:
        """Enable high attachment plus strong pairwise medium pooling."""
        if not (-1.0 <= medium_similarity_lower < high_similarity_threshold <= 1.000001):
            raise ValueError("Reference-cluster thresholds must satisfy -1 <= medium < high <= 1.000001")
        if not -1.0 <= cluster_similarity_threshold <= 1.000001:
            raise ValueError("Reference-cluster similarity threshold must lie in [-1, 1.000001]")
        if residual_keep_every < 2:
            raise ValueError("Reference-cluster residual keep interval must be at least two")
        if residual_grid_position is not None and not 0 <= residual_grid_position < 9:
            raise ValueError("Reference-cluster residual grid position must be in [0, 8] or None")
        if residual_fastvggt and residual_strong_clusters:
            raise ValueError("Reference-cluster residual FastVGGT and strong clusters are mutually exclusive")
        if residual_grid_position is not None and not residual_fastvggt:
            raise ValueError("Reference-cluster residual grid position requires FastVGGT residual pooling")
        if residual_max_clusters_per_reference < 1:
            raise ValueError("Reference-cluster residual strong-cluster count must be positive")
        if residual_cluster_similarity_threshold is not None and not -1.0 <= residual_cluster_similarity_threshold <= 1.000001:
            raise ValueError("Reference-cluster residual similarity threshold must lie in [-1, 1.000001]")
        if enabled and (self.fastvggt_enabled or self.temporal_merge_enabled or self.reference_anchor_merge_enabled or self.recursive_local_merge_enabled or self.affine_warp_merge_enabled):
            raise ValueError("FastVGGT, temporal merging, reference-anchor, and reference-cluster merging are mutually exclusive")
        self.reference_anchor_cluster_merge_enabled = enabled
        self.reference_anchor_cluster_high_similarity_threshold = high_similarity_threshold
        self.reference_anchor_cluster_medium_similarity_lower = medium_similarity_lower
        self.reference_anchor_cluster_similarity_threshold = cluster_similarity_threshold
        self.reference_anchor_cluster_residual_fastvggt = residual_fastvggt
        self.reference_anchor_cluster_residual_keep_every = residual_keep_every
        self.reference_anchor_cluster_residual_grid_position = residual_grid_position
        self.reference_anchor_cluster_residual_strong_clusters = residual_strong_clusters
        self.reference_anchor_cluster_residual_max_clusters_per_reference = residual_max_clusters_per_reference
        self.reference_anchor_cluster_residual_cluster_similarity_threshold = residual_cluster_similarity_threshold

    def set_fastvggt_global_prefix(self, global_blocks: int | None = None) -> None:
        """Restrict FastVGGT to the first ``global_blocks`` global layers.

        ``None`` means every global layer and is exactly the released/default
        FastVGGT behaviour.  ``0`` is useful for diagnostic comparisons.
        This setting does not enable FastVGGT itself.
        """
        if global_blocks is not None and global_blocks < 0:
            raise ValueError("FastVGGT global prefix must be non-negative or None")
        self.fastvggt_global_prefix_blocks = global_blocks

    def set_feature_recorder(self, recorder) -> None:
        """Attach a read-only per-global-layer diagnostic callback.

        The callback receives transient post-frame and post-global tensors and
        must immediately reduce/copy what it needs.  Normal inference leaves
        this as ``None`` and therefore retains no extra activations.
        """
        self.feature_recorder = recorder

    def set_temporal_class_aware_merge(
        self,
        enabled: bool = True,
        dynamic_unknown_keep_ratio: float = .65,
        unknown_keep_ratio: float | None = None,
        static_keep_ratio: float = .07,
        periodic_protect_ratio: float = .0,
        label_backend: str = "deep_track_graph",
    ) -> None:
        unknown_keep_ratio = dynamic_unknown_keep_ratio if unknown_keep_ratio is None else unknown_keep_ratio
        if not 0.0 < dynamic_unknown_keep_ratio <= 1.0 or not 0.0 < unknown_keep_ratio <= 1.0 or not 0.0 < static_keep_ratio <= 1.0:
            raise ValueError("temporal keep ratios must lie in (0, 1]")
        if not 0.0 <= periodic_protect_ratio <= 1.0:
            raise ValueError("temporal periodic protect ratio must lie in [0, 1]")
        label_backend = resolve_temporal_label_backend(label_backend)
        if enabled and (self.fastvggt_enabled or self.reference_anchor_merge_enabled or self.reference_anchor_cluster_merge_enabled or self.recursive_local_merge_enabled or self.affine_warp_merge_enabled):
            raise ValueError("FastVGGT, temporal merging, reference-anchor, and reference-cluster merging are mutually exclusive")
        self.temporal_merge_enabled = enabled
        self.temporal_dynamic_unknown_keep_ratio = dynamic_unknown_keep_ratio
        self.temporal_unknown_keep_ratio = unknown_keep_ratio
        self.temporal_static_keep_ratio = static_keep_ratio
        self.temporal_periodic_protect_ratio = periodic_protect_ratio
        self.temporal_label_backend = label_backend

    def set_external_temporal_labels(self, labels: torch.Tensor | None) -> None:
        """Install fixed, externally supplied patch classes for temporal merging.

        ``labels`` uses the same 0/1/2 stable/sensitive/unknown convention as
        :func:`temporal_labels`, with shape ``[T, H, W]`` (or ``[1, T, H, W]``).
        They are deliberately *not* recomputed after global layers: this hook
        is for controlled pseudo-reference/GT-label ablations, rather than a
        second temporal-labeling algorithm.
        """
        if labels is not None:
            if labels.ndim not in (3, 4) or (labels.ndim == 4 and labels.shape[0] != 1):
                raise ValueError("external temporal labels must have shape [T,H,W] or [1,T,H,W]")
            labels = labels.detach().to(dtype=torch.long)
            if labels.numel() and ((labels < 0).any() or (labels > 2).any()):
                raise ValueError("external temporal labels must use only classes 0, 1, and 2")
        self._external_temporal_labels = labels

    def set_input_temporal_consensus_labels(self, labels: torch.Tensor | None) -> None:
        """Install image-derived labels to locally extend, never replace, native ones.

        Unlike ``set_external_temporal_labels``, these classes do not bypass
        the first full-resolution global block: the causal native partition is
        still computed, then used as the only anchor for RGB-motion additions.
        """
        if labels is not None:
            if labels.ndim not in (3, 4) or (labels.ndim == 4 and labels.shape[0] != 1):
                raise ValueError("input consensus labels must have shape [T,H,W] or [1,T,H,W]")
            labels = labels.detach().to(dtype=torch.long)
            if labels.numel() and ((labels < 0).any() or (labels > 2).any()):
                raise ValueError("input consensus labels must use only classes 0, 1, and 2")
        self._input_temporal_consensus_labels = labels

    def _external_labels_for_forward(
        self, *, frames: int, patch_hw: tuple[int, int], device: torch.device,
    ) -> torch.Tensor | None:
        """Validate and discretely align fixed labels to this forward's grid."""
        labels = getattr(self, "_external_temporal_labels", None)
        if labels is None:
            return None
        if labels.ndim == 3:
            labels = labels.unsqueeze(0)
        if labels.shape[1] != frames:
            raise ValueError(
                f"external temporal labels contain {labels.shape[1]} frames, "
                f"but this forward contains {frames}"
            )
        labels = labels.to(device=device, non_blocking=True)
        if tuple(labels.shape[-2:]) != patch_hw:
            # The source labels are categorical raw-image patches.  Nearest
            # neighbour preserves classes while adapting to Omega's dynamic
            # balanced-resolution patch grid; it is not label re-estimation.
            labels = torch.nn.functional.interpolate(labels.float(), size=patch_hw, mode="nearest").long()
        return labels

    def _input_consensus_labels_for_forward(
        self, *, frames: int, patch_hw: tuple[int, int], device: torch.device,
    ) -> torch.Tensor | None:
        labels = getattr(self, "_input_temporal_consensus_labels", None)
        if labels is None:
            return None
        if labels.ndim == 3:
            labels = labels.unsqueeze(0)
        if labels.shape[1] != frames:
            raise ValueError(
                f"input consensus labels contain {labels.shape[1]} frames, but this forward contains {frames}"
            )
        labels = labels.to(device=device, non_blocking=True)
        if tuple(labels.shape[-2:]) != patch_hw:
            labels = torch.nn.functional.interpolate(labels.float(), size=patch_hw, mode="nearest").long()
        return labels

    def set_temporal_label_recorder(self, recorder) -> None:
        """Attach a read-only recorder for post-global temporal partitions."""
        self.temporal_label_recorder = recorder
        if recorder is not None:
            recorder.label_backend = self.temporal_label_backend

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)
        # patch_embed initialises itself in _build_patch_embed.
        named_apply(init_weights_vit, self.frame_blocks)
        named_apply(init_weights_vit, self.inter_frame_blocks)

    def _run_block(self, block: nn.Module, tokens: torch.Tensor, rope_sincos) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, tokens, rope_sincos, use_reentrant=False)
        return block(tokens, rope_sincos)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self._input_patch_tokens_cache
        if patch_tokens is None:
            patch_tokens = self.patch_embed(images)
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]
        elif (patch_tokens.shape[0] != batch_size * num_frames
              or patch_tokens.shape[1] != (height // self.patch_size) * (width // self.patch_size)
              or patch_tokens.device != images.device):
            raise ValueError("cached patch tokens do not match the current model input")

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        # A view of the encoder output, retained only for this forward.  It
        # reuses the existing patch embedding and lets the temporal labeler
        # preserve high-frequency object contours after the first full global
        # block without a second encoder call.
        native_image_tokens = patch_tokens.view(
            batch_size, num_frames, patch_grid_size[0], patch_grid_size[1], embed_dim,
        )
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        # Labels belong to a single forward only.  This mirrors Any4D's hook:
        # a global layer refreshes labels only for its successor, never for a
        # subsequent clip or after the final global attention layer.
        self._temporal_labels = None
        self.reference_anchor_last_plan_stats = []
        self.reference_anchor_cluster_last_plan_stats = []
        self.recursive_local_last_plan_stats = []
        self.affine_warp_last_plan_stats = []
        self.fastvggt_last_plan_stats = []
        self.sparse_convex_last_plan_stats = []
        self._sparse_convex_reuse_plan = None
        try:
            outputs = []
            for block_idx in range(self.depth):
                tokens, frame_tokens = self._run_frame_block(
                    tokens,
                    batch_size,
                    num_frames,
                    num_tokens,
                    embed_dim,
                    block_idx,
                    frame_rope,
                )
                tokens = self._run_inter_frame_attention_block(
                    tokens,
                    batch_size,
                    num_frames,
                    num_tokens,
                    embed_dim,
                    block_idx,
                    self.inter_frame_attention_types[block_idx],
                    patch_grid_size,
                    native_image_tokens,
                )
                if self.feature_recorder is not None and self.inter_frame_attention_types[block_idx] == "global":
                    self.feature_recorder.record(
                        block_idx=block_idx,
                        frame_tokens=frame_tokens,
                        global_tokens=tokens,
                        patch_token_start=self.patch_token_start,
                        patch_hw=patch_grid_size,
                    )
                if block_idx in self.cached_layer_indices:
                    outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
                else:
                    outputs.append(None)
            return outputs, self.patch_token_start
        finally:
            self._temporal_labels = None

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self._run_block(self.frame_blocks[block_idx], tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        patch_hw: tuple[int, int],
        native_image_tokens: torch.Tensor,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            global_ordinal = sum(kind == "global" for kind in self.inter_frame_attention_types[:block_idx])
            use_fastvggt = self.fastvggt_enabled and (
                self.fastvggt_global_prefix_blocks is None
                or global_ordinal < self.fastvggt_global_prefix_blocks
            )
            use_sparse_convex = self.sparse_convex_enabled
            use_reference_anchor = self.reference_anchor_merge_enabled
            use_reference_anchor_cluster = self.reference_anchor_cluster_merge_enabled
            use_recursive_local = self.recursive_local_merge_enabled
            use_affine_warp = self.affine_warp_merge_enabled
            if use_fastvggt or use_sparse_convex or use_reference_anchor or use_reference_anchor_cluster or use_recursive_local or use_affine_warp or self.temporal_merge_enabled:
                if sum((bool(self.fastvggt_enabled), bool(self.sparse_convex_enabled), bool(self.temporal_merge_enabled), bool(self.reference_anchor_merge_enabled), bool(self.reference_anchor_cluster_merge_enabled), bool(self.recursive_local_merge_enabled), bool(self.affine_warp_merge_enabled))) > 1:
                    raise RuntimeError("all token merge modes are mutually exclusive")
                # Native labels must be causal with respect to the encoder:
                # the first global layer establishes time-aware features in
                # full resolution, and only its output labels the *next*
                # global layer.  Explicit external pseudo-labels retain their
                # controlled ablation semantics and may be used immediately.
                if self.temporal_merge_enabled and self._temporal_labels is None:
                    external_labels = self._external_labels_for_forward(
                        frames=num_frames, patch_hw=patch_hw, device=tokens.device,
                    )
                    if external_labels is None:
                        tokens = self._run_block(self.inter_frame_blocks[block_idx], tokens, None)
                        self._temporal_labels = temporal_labels(
                            tokens, frames=num_frames, tokens_per_frame=num_tokens,
                            patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                            backend=self.temporal_label_backend,
                            native_image_tokens=native_image_tokens if self.temporal_label_backend in {"deep_track_graph", "adaptive_native"} else None,
                        )
                        input_consensus = self._input_consensus_labels_for_forward(
                            frames=num_frames, patch_hw=patch_hw, device=tokens.device,
                        )
                        if input_consensus is not None:
                            self._temporal_labels = fuse_native_rgb_temporal_labels(
                                self._temporal_labels, input_consensus,
                            )
                        if self.temporal_label_recorder is not None:
                            self.temporal_label_recorder.record(block_idx, self._temporal_labels)
                        return tokens.view(batch_size, num_frames, num_tokens, embed_dim)
                    self._temporal_labels = external_labels
                if (use_sparse_convex
                        and global_ordinal in self.sparse_convex_dense_global_layers):
                    tokens = self._run_block(self.inter_frame_blocks[block_idx], tokens, None)
                    self.sparse_convex_last_plan_stats.append({
                        "block_index": block_idx,
                        "global_ordinal": global_ordinal,
                        "original_tokens": int(tokens.shape[1]),
                        "kept_tokens": int(tokens.shape[1]),
                        "merged_tokens": 0,
                        "merge_ratio": 0.0,
                        "compact_queries": False,
                        "physical_keys": True,
                        "reused_plan": False,
                        "attention_query_tokens": int(tokens.shape[1]),
                        "planner": "dense_layer_bypass",
                    })
                    return tokens.view(batch_size, num_frames, num_tokens, embed_dim)
                normalized = self.inter_frame_blocks[block_idx].norm1(tokens)
                if use_sparse_convex:
                    if self.sparse_convex_qkv_global_layers is not None:
                        compact_queries = global_ordinal in self.sparse_convex_qkv_global_layers
                    else:
                        compact_queries = global_ordinal < self.sparse_convex_qkv_global_blocks
                    preserve_queries = not compact_queries
                    preserve_keys = preserve_queries or global_ordinal in self.sparse_convex_physical_key_layers
                    kmeans_planners = {"kmeans_convex", "kmeans_convex_fast", "kmeans_convex_fused", "kmeans_convex_fused_batched", "kmeans_convex_4d_fused_batched", "kmeans_convex_reuse"}
                    ttt_planners = {"ttt_partitioned_convex", "ttt_protected_uniform_convex"}
                    use_ttt_4d = (
                        self.sparse_convex_planner not in ttt_planners
                        or global_ordinal in self.sparse_convex_4d_global_layers
                    )
                    effective_planner = (
                        self.sparse_convex_planner if use_ttt_4d else "fixed_topology"
                    )
                    plan_builder = (build_sparse_convex_plan if effective_planner == "fixed_topology"
                                    else build_hash_convex_plan if self.sparse_convex_planner == "hash_convex"
                                    else build_kmeans_basis_plan if self.sparse_convex_planner in kmeans_planners
                                    else build_ttt_partitioned_convex_plan if effective_planner == "ttt_partitioned_convex"
                                    else build_ttt_protected_uniform_convex_plan if effective_planner == "ttt_protected_uniform_convex"
                                    else build_adaptive_basis_plan)
                    plan_kwargs = dict(
                        frames=num_frames, tokens_per_frame=num_tokens,
                        patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                        threshold=self.sparse_convex_threshold, preserve_queries=preserve_queries,
                        preserve_destination_keys=preserve_keys,
                    )
                    if effective_planner in ttt_planners:
                        if (self.sparse_convex_4d_geometry is None
                                or self.sparse_convex_ttt_labels is None
                                or self.sparse_convex_ttt_sensitivity is None):
                            raise RuntimeError(
                                "TTT partitioned planner requires geometry, labels and sensitivity"
                            )
                        plan_kwargs.update(
                            labels=self.sparse_convex_ttt_labels,
                            sensitivity=self.sparse_convex_ttt_sensitivity,
                            geometry=self.sparse_convex_4d_geometry,
                            max_support=self.sparse_convex_max_support,
                            selection_geometry_weight=self.sparse_convex_4d_selection_weight,
                            support_geometry_weight=self.sparse_convex_4d_support_weight,
                            support_geometry_guard_tolerance=self.sparse_convex_4d_guard_tolerance,
                        )
                        if effective_planner == "ttt_protected_uniform_convex":
                            plan_kwargs.update(top1_prune_threshold=(
                                .95 if self.sparse_convex_ttt_top1_pruning else 2.0
                            ), sensitive_event_ratio=(
                                self.sparse_convex_ttt_sensitive_event_ratio
                            ), fixed_dynamic_mask=self.sparse_convex_ttt_fixed_dynamic)
                    elif effective_planner == "hash_convex":
                        plan_kwargs.update(hash_bits=self.sparse_convex_hash_bits)
                    elif effective_planner in kmeans_planners:
                        plan_kwargs.update(basis_ratio=self.sparse_convex_basis_ratio,
                                           max_support=self.sparse_convex_max_support,
                                           fast_solver=self.sparse_convex_planner == "kmeans_convex_fast",
                                           fused_solver=self.sparse_convex_planner in {"kmeans_convex_fused", "kmeans_convex_fused_batched", "kmeans_convex_4d_fused_batched"},
                                           fused_batching=self.sparse_convex_planner in {"kmeans_convex_fused_batched", "kmeans_convex_4d_fused_batched"})
                        if self.sparse_convex_planner == "kmeans_convex_4d_fused_batched":
                            if self.sparse_convex_4d_geometry is None:
                                raise RuntimeError("4-D K-means planner requires a predicted geometry pilot")
                            use_4d = global_ordinal in self.sparse_convex_4d_global_layers
                            plan_kwargs.update(
                                geometry=self.sparse_convex_4d_geometry if use_4d else None,
                                selection_geometry_weight=(
                                    self.sparse_convex_4d_selection_weight if use_4d else 0.0
                                ),
                                support_geometry_weight=(
                                    self.sparse_convex_4d_support_weight if use_4d else 0.0
                                ),
                                support_geometry_guard_tolerance=(
                                    self.sparse_convex_4d_guard_tolerance if use_4d else None
                                ),
                            )
                    elif effective_planner != "fixed_topology":
                        plan_kwargs.update(mode=self.sparse_convex_planner,
                                           max_support=self.sparse_convex_max_support,
                                           coefficient_limit=self.sparse_convex_coefficient_limit)
                    reuse_plan = (
                        self.sparse_convex_planner == "kmeans_convex_reuse"
                        or (
                            self.sparse_convex_planner == "ttt_protected_uniform_convex"
                            and os.environ.get("VGGT_TTT_REUSE_PLAN", "0") == "1"
                        )
                    )
                    refit_ordinal = int(os.environ.get("VGGT_TTT_REFIT_ORDINAL", "-1"))
                    force_refit = (
                        reuse_plan and refit_ordinal >= 0
                        and global_ordinal == refit_ordinal
                    )
                    if reuse_plan and self._sparse_convex_reuse_plan is not None and not force_refit:
                        cached_plan = self._sparse_convex_reuse_plan
                        plan = (
                            cached_plan
                            if (cached_plan.preserve_queries == preserve_queries
                                and cached_plan.preserve_destination_keys == preserve_keys)
                            else replace(
                                cached_plan, preserve_queries=preserve_queries,
                                preserve_destination_keys=preserve_keys,
                            )
                        )
                    else:
                        plan = plan_builder(normalized, **plan_kwargs)
                        if reuse_plan:
                            self._sparse_convex_reuse_plan = plan
                    self.sparse_convex_last_plan_stats.append({
                        "block_index": block_idx,
                        "global_ordinal": global_ordinal,
                        "original_tokens": plan.original_tokens,
                        "kept_tokens": plan.kept_tokens,
                        "merged_tokens": plan.original_tokens - plan.kept_tokens,
                        "merge_ratio": 1.0 - plan.keep_ratio,
                        "compact_queries": bool(compact_queries),
                        "physical_keys": bool(preserve_keys),
                        "reused_plan": bool(reuse_plan and global_ordinal > 0 and not force_refit),
                        "refit_plan": bool(force_refit),
                        "attention_query_tokens": int(
                            plan.kept_tokens if compact_queries else plan.original_tokens
                        ),
                        **plan.metadata,
                    })
                elif use_fastvggt:
                    if self.fastvggt_operator_signature:
                        attention = self.inter_frame_blocks[block_idx].attn
                        qkv = attention.qkv(normalized)
                        head_dim = embed_dim // attention.num_heads
                        split = qkv.reshape(
                            batch_size, num_frames * num_tokens, 3,
                            attention.num_heads, head_dim,
                        )
                        query = split[:, :, 0].transpose(1, 2)
                        key = split[:, :, 1].transpose(1, 2)
                        value = split[:, :, 2].transpose(1, 2)
                        if attention.use_qk_norm:
                            query, key = attention.q_norm(query), attention.k_norm(key)
                        plan = build_operator_fastvggt_plan(
                            normalized, query=query, key=key, value=value,
                            qkv_features=qkv, frames=num_frames,
                            tokens_per_frame=num_tokens,
                            patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                            merge_ratio=self.fastvggt_merge_ratio,
                            protect_ratio=self.fastvggt_protect_ratio,
                            probe_count=self.fastvggt_operator_probe_count,
                        )
                    else:
                        plan = build_fastvggt_plan(
                            normalized, frames=num_frames, tokens_per_frame=num_tokens,
                            patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                            merge_ratio=self.fastvggt_merge_ratio,
                            protect_ratio=self.fastvggt_protect_ratio,
                        )
                    self.fastvggt_last_plan_stats.append({
                        "block_index": block_idx,
                        "original_tokens": plan.original_tokens,
                        "kept_tokens": plan.kept_tokens,
                        "merged_tokens": plan.original_tokens - plan.kept_tokens,
                        "merge_ratio": 1.0 - plan.keep_ratio,
                        **plan.metadata,
                    })
                elif use_reference_anchor:
                    plan = build_reference_anchor_plan(
                        normalized, frames=num_frames, tokens_per_frame=num_tokens,
                        patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                        similarity_threshold=self.reference_anchor_similarity_threshold,
                        reference_frame_count=self.reference_anchor_frame_count,
                    )
                    self.reference_anchor_last_plan_stats.append({
                        "block_index": block_idx,
                        "original_tokens": plan.original_tokens,
                        "kept_tokens": plan.kept_tokens,
                        "merged_tokens": plan.original_tokens - plan.kept_tokens,
                        "merge_ratio": 1.0 - plan.keep_ratio,
                    })
                elif use_reference_anchor_cluster:
                    plan = build_reference_anchor_cluster_plan(
                        normalized, frames=num_frames, tokens_per_frame=num_tokens,
                        patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                        high_similarity_threshold=self.reference_anchor_cluster_high_similarity_threshold,
                        medium_similarity_lower=self.reference_anchor_cluster_medium_similarity_lower,
                        cluster_similarity_threshold=self.reference_anchor_cluster_similarity_threshold,
                        residual_fastvggt=self.reference_anchor_cluster_residual_fastvggt,
                        residual_keep_every=self.reference_anchor_cluster_residual_keep_every,
                        residual_grid_position=self.reference_anchor_cluster_residual_grid_position,
                        residual_strong_clusters=self.reference_anchor_cluster_residual_strong_clusters,
                        residual_max_clusters_per_reference=self.reference_anchor_cluster_residual_max_clusters_per_reference,
                        residual_cluster_similarity_threshold=self.reference_anchor_cluster_residual_cluster_similarity_threshold,
                    )
                    self.reference_anchor_cluster_last_plan_stats.append({
                        "block_index": block_idx,
                        "original_tokens": plan.original_tokens,
                        "kept_tokens": plan.kept_tokens,
                        "merged_tokens": plan.original_tokens - plan.kept_tokens,
                        "merge_ratio": 1.0 - plan.keep_ratio,
                        **plan.metadata,
                    })
                elif use_recursive_local:
                    plan = build_recursive_local_2x2_plan(
                        normalized, frames=num_frames, tokens_per_frame=num_tokens,
                        patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                        similarity_threshold=self.recursive_local_similarity_threshold,
                        dominant_corner_only=self.recursive_local_dominant_corner_only,
                    )
                    self.recursive_local_last_plan_stats.append({
                        "block_index": block_idx,
                        "original_tokens": plan.original_tokens,
                        "kept_tokens": plan.kept_tokens,
                        "merged_tokens": plan.original_tokens - plan.kept_tokens,
                        **plan.metadata,
                    })
                elif use_affine_warp:
                    affine_target_merge_ratio = self.affine_warp_target_merge_ratio
                    affine_qkv = (
                        self.inter_frame_blocks[block_idx].attn.qkv(normalized)
                        if (self.affine_warp_qkv_similarity_threshold is not None
                            or affine_target_merge_ratio is not None) else None
                    )
                    affine_importance = None
                    if affine_target_merge_ratio is not None:
                        attention = self.inter_frame_blocks[block_idx].attn
                        head_dim = embed_dim // attention.num_heads
                        split = affine_qkv.reshape(
                            batch_size, num_frames * num_tokens, 3, attention.num_heads, head_dim,
                        )
                        query = split[:, :, 0].transpose(1, 2)
                        key = split[:, :, 1].transpose(1, 2)
                        if attention.use_qk_norm:
                            query, key = attention.q_norm(query), attention.k_norm(key)
                        if self.affine_warp_importance_policy == "camera_qk":
                            anchor_frames = torch.linspace(
                                0, num_frames - 1,
                                min(self.affine_warp_camera_importance_anchors, num_frames),
                                device=tokens.device,
                            ).round().long().unique()
                            camera_ids = anchor_frames * num_tokens
                            camera_query = query.index_select(2, camera_ids)
                            # Max over camera anchors and heads: a token
                            # important to any retained camera view is kept.
                            affine_importance = torch.einsum(
                                "bhad,bhnd->bhan", camera_query.float(), key.float(),
                            ).amax(dim=(1, 2))[0]
                        else:
                            # First-order leave-one-key influence on the
                            # reference frame's complete camera/register bank:
                            # a_qj * ||v_j - y_q||.  This captures both the
                            # attention probability and whether the value is
                            # actually replaceable in the special-token readout.
                            key_fp32 = key[0].float()
                            value_fp32 = split[0, :, 2].transpose(0, 1).float()
                            def influence(query_set: torch.Tensor) -> torch.Tensor:
                                logits = torch.einsum("had,hnd->han", query_set.float(), key_fp32)
                                logits.mul_(head_dim ** -0.5)
                                probability = logits.softmax(dim=-1)
                                output = torch.einsum("han,hnd->had", probability, value_fp32)
                                value_sq = value_fp32.square().sum(dim=-1)
                                output_sq = output.square().sum(dim=-1)
                                cross = torch.einsum("had,hnd->han", output, value_fp32)
                                distance = (
                                    output_sq[:, :, None] + value_sq[:, None, :] - 2.0 * cross
                                ).clamp_min_(0.0).sqrt_()
                                return (probability * distance).sum(dim=(0, 1))

                            reference_importance = influence(query[0, :, :self.patch_token_start])
                            if self.affine_warp_importance_policy == "reference_special_influence":
                                affine_importance = reference_importance
                            else:
                                anchor_frames = torch.linspace(
                                    0, num_frames - 1,
                                    min(self.affine_warp_camera_importance_anchors, num_frames),
                                    device=tokens.device,
                                ).round().long().unique()
                                camera_ids = anchor_frames * num_tokens
                                temporal_importance = influence(query[0].index_select(1, camera_ids))
                                reference_importance = reference_importance / reference_importance.sum().clamp_min(1e-12)
                                temporal_importance = temporal_importance / temporal_importance.sum().clamp_min(1e-12)
                                affine_importance = (
                                    reference_importance
                                    + self.affine_warp_temporal_importance_weight * temporal_importance
                                )
                    plan = build_affine_warp_plan(
                        normalized, frames=num_frames, tokens_per_frame=num_tokens,
                        patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                        similarity_threshold=self.affine_warp_similarity_threshold,
                        anchor_grid=self.affine_warp_anchor_grid,
                        assignment_mode=self.affine_warp_assignment_mode,
                        dense_special_kv=(
                            self.affine_warp_dense_special_kv
                            and (self.affine_warp_dense_special_start_block is None
                                 or block_idx >= self.affine_warp_dense_special_start_block)
                        ),
                        pool_source_content=self.affine_warp_pool_source_content,
                        qkv_features=affine_qkv,
                        qkv_similarity_threshold=self.affine_warp_qkv_similarity_threshold,
                        star_window=self.affine_warp_star_window,
                        target_merge_ratio=affine_target_merge_ratio,
                        token_importance=affine_importance,
                    )
                    self.affine_warp_last_plan_stats.append({
                        "block_index": block_idx,
                        "original_tokens": plan.original_tokens,
                        "kept_tokens": plan.kept_tokens,
                        "merged_tokens": plan.original_tokens - plan.kept_tokens,
                        **plan.metadata,
                    })
                else:
                    plan = build_temporal_plan(
                        normalized, self._temporal_labels, frames=num_frames,
                        tokens_per_frame=num_tokens, patch_token_start=self.patch_token_start,
                        patch_hw=patch_hw,
                        dynamic_unknown_keep_ratio=self.temporal_dynamic_unknown_keep_ratio,
                        unknown_keep_ratio=self.temporal_unknown_keep_ratio,
                        static_keep_ratio=self.temporal_static_keep_ratio,
                        periodic_protect_ratio=self.temporal_periodic_protect_ratio,
                    )
                tokens = self.inter_frame_blocks[block_idx].forward_compacted(tokens, plan, normalized=normalized)
                del normalized, plan
                if self.temporal_merge_enabled:
                    external_labels = self._external_labels_for_forward(
                        frames=num_frames, patch_hw=patch_hw, device=tokens.device,
                    )
                    if external_labels is not None:
                        # Fixed external labels are the experimental variable:
                        # never replace them using this model's features.
                        self._temporal_labels = external_labels
                        if self.temporal_label_recorder is not None:
                            self.temporal_label_recorder.record(block_idx, self._temporal_labels)
                    else:
                        global_indices = [i for i, kind in enumerate(self.inter_frame_attention_types) if kind == "global"]
                        if self.temporal_label_backend in {"deep_track_graph", "adaptive_native", "predictability_topology"}:
                            # Native patch labels describe the immutable
                            # encoder output from this forward.  Recomputing
                            # the same local tracks after every global block
                            # would only add latency; reuse the causal map.
                            if self.temporal_label_recorder is not None:
                                self.temporal_label_recorder.record(block_idx, self._temporal_labels)
                        elif block_idx != global_indices[-1]:
                            self._temporal_labels = temporal_labels(
                                tokens, frames=num_frames, tokens_per_frame=num_tokens,
                                patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                                backend=self.temporal_label_backend,
                            native_image_tokens=native_image_tokens if self.temporal_label_backend in {"deep_track_graph", "adaptive_native"} else None,
                            )
                            if self.temporal_label_recorder is not None:
                                self.temporal_label_recorder.record(block_idx, self._temporal_labels)
                        elif self.temporal_label_recorder is not None:
                            # This diagnostic-only pass mirrors the label function
                            # after the final attention layer; it is never reused.
                            self.temporal_label_recorder.record(
                                block_idx,
                                temporal_labels(
                                    tokens, frames=num_frames, tokens_per_frame=num_tokens,
                                    patch_token_start=self.patch_token_start, patch_hw=patch_hw,
                                    backend=self.temporal_label_backend,
                                    native_image_tokens=native_image_tokens if self.temporal_label_backend in {"deep_track_graph", "adaptive_native"} else None,
                                ),
                            )
            else:
                tokens = self._run_block(self.inter_frame_blocks[block_idx], tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self._run_block(
            self.inter_frame_blocks[block_idx], camera_and_register_tokens, None
        )
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


def _build_patch_embed(patch_size: int, embed_dim: int, use_checkpoint: bool = False) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        use_checkpoint=use_checkpoint,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
