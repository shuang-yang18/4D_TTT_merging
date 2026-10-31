"""Canonical configuration for the released 4D-aware TTT merging method."""
from __future__ import annotations

import os


METHOD_ENV = {
    "VGGT_TTT_ONLINE_FIXED_BASIS": "1",
    "VGGT_TTT_FIXED_BASIS_BUDGET_300": "20000",
    "VGGT_TTT_FIXED_BASIS_SAMPLE_STRIDE": "10",
    "VGGT_TTT_FIXED_BASIS_STEPS": "500",
    "VGGT_TTT_FIXED_BASIS_THRESHOLD_FACTOR": "0.4",
    "VGGT_TTT_REUSE_PLAN": "0",
    "VGGT_TTT_FIXED_TOPOLOGY": "0",
    "VGGT_TTT_LOCAL_SUPPORT_SEARCH": "1",
    "VGGT_TTT_FIT_CHUNK": "4096",
    "VGGT_TTT_STABLE_SLOTS_PER_4X4": "4",
    "VGGT_TTT_LOCAL_TOP1": "0",
    "VGGT_TTT_LOCAL_SOLVER": "projected",
    "VGGT_TRITON_ACTIVE_SET": "0",
    "VGGT_TRITON_MASS_BIAS_ATTENTION": "1",
}


EVAL_FLAGS = (
    "--sparse-convex",
    "--sparse-convex-planner", "ttt_protected_uniform_convex",
    "--sparse-convex-threshold", "0.90",
    "--sparse-convex-max-support", "4",
    "--sparse-convex-ttt-disable-top1-pruning",
    "--sparse-convex-ttt-sensitive-event-ratio", "0.01",
    "--sparse-convex-4d-source", "anchor_ttt",
    "--sparse-convex-4d-anchor-frames", "12",
    "--sparse-convex-4d-flow-backend", "token",
    "--sparse-convex-token-match-dim", "64",
    "--sparse-convex-token-match-projection", "random",
    "--sparse-convex-token-match-radius", "4",
    "--sparse-convex-token-match-chunk-frames", "32",
    "--sparse-convex-token-match-sensitive-cap", "0.20",
    "--sparse-convex-4d-ttt-mode", "strict_adapter",
    "--sparse-convex-4d-strict-ttt-steps", "12",
    "--sparse-convex-4d-strict-ttt-partition-source", "adapter",
    "--sparse-convex-4d-selection-weight", "0.01",
    "--sparse-convex-4d-support-weight", "0.01",
    "--sparse-convex-4d-guard-tolerance", "0",
    # Global ordinals 8/10/11/12 retain full queries; all others compact Q/K/V.
    "--sparse-convex-qkv-global-layers",
    "0,1,2,3,4,5,6,7,9,13,14,15,16,17,18",
    "--sparse-convex-dense-global-layers", "",
)


def apply_environment() -> None:
    """Apply the canonical environment without overriding explicit user values."""
    for name, value in METHOD_ENV.items():
        os.environ.setdefault(name, value)

