# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import warnings

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF


def load_and_preprocess_images(image_path_list, mode="balanced", image_resolution=512, patch_size=16, *, return_metadata=False):
    """Load images for VGGT-Omega inference.

    `balanced` keeps the total token count close to image_resolution**2.
    `max_size` resizes the longest side to image_resolution.
    Both modes first center-crop extreme aspect ratios into [0.5, 2.0].
    Optionally return exact crop/resize/padding metadata for diagnostic maps.
    """
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")
    if mode not in ["balanced", "max_size"]:
        raise ValueError("Mode must be either 'balanced' or 'max_size'")
    if image_resolution <= 0:
        raise ValueError("image_resolution must be positive")
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_resolution % patch_size != 0:
        raise ValueError("image_resolution must be divisible by patch_size")

    images = []
    metadata = []
    shapes = set()
    to_tensor = TF.ToTensor()

    for image_path in image_path_list:
        original = _load_rgb_image(image_path)
        crop_box = _supported_crop_box(original.size)
        image = original.crop(crop_box)
        width, height = image.size
        aspect_ratio = height / max(width, 1)

        if mode == "balanced":
            target_h, target_w = _balanced_target_shape(aspect_ratio, image_resolution, patch_size)
        else:
            target_h, target_w = _max_size_target_shape(aspect_ratio, image_resolution, patch_size)

        image = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        image = to_tensor(image)

        shapes.add((image.shape[1], image.shape[2]))
        images.append(image)
        metadata.append({"original_size": list(original.size), "crop_box": list(crop_box),
                         "resized_size": [target_w, target_h]})

    if len(shapes) > 1:
        warnings.warn(f"Found images with different shapes: {shapes}; padding to a common size.", stacklevel=2)
        images = _pad_images_to_common_size(images, shapes)

    result = torch.stack(images)
    canvas_h, canvas_w = result.shape[-2:]
    for item in metadata:
        resized_w, resized_h = item["resized_size"]
        left, top = (canvas_w - resized_w) // 2, (canvas_h - resized_h) // 2
        item["canvas_size"] = [canvas_w, canvas_h]
        item["content_box"] = [left, top, left + resized_w, top + resized_h]
    return (result, metadata) if return_metadata else result


def _load_rgb_image(image_path):
    with Image.open(image_path) as image:
        if image.mode == "RGBA":
            background = Image.new("RGBA", image.size, (255, 255, 255, 255))
            image = Image.alpha_composite(background, image)
        return image.convert("RGB")


def _crop_to_supported_aspect_ratio(image, min_aspect_ratio=0.5, max_aspect_ratio=2.0):
    return image.crop(_supported_crop_box(image.size, min_aspect_ratio, max_aspect_ratio))


def _supported_crop_box(size, min_aspect_ratio=0.5, max_aspect_ratio=2.0):
    width, height = size
    aspect_ratio = height / max(width, 1)

    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return (left, 0, left + crop_width, height)

    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return (0, top, width, top + crop_height)

    return (0, 0, width, height)


def _balanced_target_shape(aspect_ratio, image_resolution, patch_size):
    token_number = (image_resolution // patch_size) ** 2
    w_patches = np.sqrt(token_number / aspect_ratio)
    h_patches = token_number / w_patches
    w_patches = max(1, int(np.round(w_patches)))
    h_patches = max(1, int(np.round(h_patches)))
    return h_patches * patch_size, w_patches * patch_size


def _max_size_target_shape(aspect_ratio, image_resolution, patch_size):
    if aspect_ratio >= 1.0:
        height = image_resolution
        width = _round_to_patch_multiple(image_resolution / aspect_ratio, patch_size)
    else:
        width = image_resolution
        height = _round_to_patch_multiple(image_resolution * aspect_ratio, patch_size)
    return height, width


def _round_to_patch_multiple(value, patch_size):
    return max(patch_size, int(np.round(float(value) / patch_size)) * patch_size)


def _pad_images_to_common_size(images, shapes):
    max_height = max(shape[0] for shape in shapes)
    max_width = max(shape[1] for shape in shapes)

    padded_images = []
    for image in images:
        h_padding = max_height - image.shape[1]
        w_padding = max_width - image.shape[2]
        if h_padding > 0 or w_padding > 0:
            pad_top = h_padding // 2
            pad_bottom = h_padding - pad_top
            pad_left = w_padding // 2
            pad_right = w_padding - pad_left
            image = torch.nn.functional.pad(
                image,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded_images.append(image)

    return padded_images
