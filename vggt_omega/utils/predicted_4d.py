"""Build patch-aligned 4-D descriptors from VGGT-Omega predictions."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .pose_enc import encoding_to_camera


@torch.no_grad()
def predicted_4d_patch_descriptor(
    prediction: dict[str, torch.Tensor],
    patch_hw: tuple[int, int],
    *,
    time_weight: float = .25,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return robustly normalized world XYZ plus frame time for every patch.

    The descriptor order is frame-major and then row-major within a frame,
    matching the flattened patch-token order used by ``Aggregator``.  It is a
    prediction-only pilot: neither dataset depth nor camera ground truth is
    read here.  Absolute monocular scale is removed by a robust scene radius,
    because basis construction only needs relative spatiotemporal proximity.
    """
    if time_weight < 0:
        raise ValueError("time_weight must be nonnegative")
    required = {"depth", "pose_enc", "images"}
    missing = required.difference(prediction)
    if missing:
        raise KeyError(f"prediction lacks required 4-D fields: {sorted(missing)}")
    depth = prediction["depth"]
    if depth.ndim != 5 or depth.shape[0] != 1:
        raise ValueError(f"depth must be [1,T,H,W,1], got {tuple(depth.shape)}")
    depth = depth[0].squeeze(-1).float()
    frames, image_height, image_width = depth.shape
    patch_height, patch_width = patch_hw
    if patch_height < 1 or patch_width < 1:
        raise ValueError("patch grid must be nonempty")
    depth = F.interpolate(
        depth[:, None], size=patch_hw, mode="bilinear", align_corners=False,
    )[:, 0]
    w2c_34, intrinsics = encoding_to_camera(
        prediction["pose_enc"], prediction["images"].shape[-2:],
    )
    w2c = torch.eye(4, device=depth.device, dtype=torch.float32).view(1, 1, 4, 4).repeat(
        1, frames, 1, 1,
    )
    w2c[..., :3, :] = w2c_34.float()
    c2w = torch.linalg.inv(w2c)[0]
    intrinsics = intrinsics[0].float()

    yy = (torch.arange(patch_height, device=depth.device, dtype=torch.float32) + .5)
    xx = (torch.arange(patch_width, device=depth.device, dtype=torch.float32) + .5)
    yy = yy * (image_height / patch_height) - .5
    xx = xx * (image_width / patch_width) - .5
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    z = depth
    x = (grid_x[None] - intrinsics[:, None, None, 0, 2]) / intrinsics[:, None, None, 0, 0] * z
    y = (grid_y[None] - intrinsics[:, None, None, 1, 2]) / intrinsics[:, None, None, 1, 1] * z
    camera = torch.stack((x, y, z, torch.ones_like(z)), dim=-1)
    world = torch.einsum("tij,thwj->thwi", c2w, camera)[..., :3].reshape(-1, 3)
    finite = torch.isfinite(world).all(dim=-1)
    if not bool(finite.all()):
        replacement = (
            world[finite].median(dim=0).values if finite.any()
            else torch.zeros(3, device=world.device)
        )
        world[~finite] = replacement
    centre = world.median(dim=0).values
    radius = (world - centre).norm(dim=-1)
    scale = torch.quantile(radius, .75).clamp_min(1e-4)
    normalized_world = ((world - centre) / scale).clamp(-4, 4)
    frame_time = torch.linspace(-.5, .5, frames, device=world.device, dtype=torch.float32)
    frame_time = frame_time[:, None].expand(frames, patch_height * patch_width).reshape(-1, 1)
    descriptor = torch.cat((normalized_world, time_weight * frame_time), dim=-1).contiguous()
    return descriptor, {
        "frames": float(frames),
        "patches_per_frame": float(patch_height * patch_width),
        "valid_world_ratio": float(finite.float().mean()),
        "world_scale": float(scale),
        "depth_mean": float(depth.mean()),
        "time_weight": float(time_weight),
    }
