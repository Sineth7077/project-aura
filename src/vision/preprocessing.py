"""Image preprocessing utilities for computer vision pipelines.

Provides reusable image transformation functions used by both Colab
notebooks and production inference code. Includes resizing, normalization,
augmentation, and batch preparation.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def resize_image(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    preserve_aspect_ratio: bool = True,
) -> np.ndarray:
    """Resize an image to target dimensions.

    Args:
        image: Input image as numpy array (H, W, C).
        target_size: (height, width) tuple for output dimensions.
        preserve_aspect_ratio: If True, pad instead of distorting.

    Returns:
        Resized image as numpy array.

    Raises:
        ValueError: If image is not a valid numpy array or has wrong dimensions.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")
    if image.ndim not in (2, 3):
        raise ValueError("Image must be 2D (grayscale) or 3D (RGB/RGBA)")

    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required for image resizing")

    h, w = target_size
    if preserve_aspect_ratio:
        # Calculate scaling to fit within target while preserving aspect ratio
        scale = min(h / image.shape[0], w / image.shape[1])
        new_w = int(image.shape[1] * scale)
        new_h = int(image.shape[0] * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        # Pad to target size
        pad_h = h - new_h
        pad_w = w - new_w
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        if image.ndim == 2:
            padded = np.pad(resized, ((top, bottom), (left, right)), mode="constant")
        else:
            padded = np.pad(resized, ((top, bottom), (left, right), (0, 0)), mode="constant")
        return padded
    else:
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def normalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    to_float: bool = True,
) -> np.ndarray:
    """Normalize image using ImageNet statistics by default.

    Args:
        image: Input image as numpy array (H, W, C) with values in [0, 255] or [0, 1].
        mean: Per-channel mean values.
        std: Per-channel standard deviation values.
        to_float: If True, convert to float32 and scale to [0, 1] first.

    Returns:
        Normalized image as float32 numpy array.
    """
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")

    img = image.astype(np.float32) if to_float else image.astype(np.float32)

    if img.max() > 1.0:
        img = img / 255.0

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    normalized = (img - mean_arr) / std_arr
    return normalized.astype(np.float32)


def prepare_batch(
    images: list[np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Prepare a batch of images for model input.

    Resizes, normalizes, and stacks images into a single NCHW batch tensor.

    Args:
        images: List of input images as numpy arrays.
        target_size: Target (height, width) for each image.

    Returns:
        Batch tensor of shape (N, C, H, W) as float32.
    """
    if not images:
        raise ValueError("Image list cannot be empty")

    processed = []
    for img in images:
        resized = resize_image(img, target_size=target_size)
        normalized = normalize_image(resized)
        # Convert HWC -> CHW
        chw = np.transpose(normalized, (2, 0, 1)) if normalized.ndim == 3 else normalized[np.newaxis, ...]
        processed.append(chw)

    batch = np.stack(processed, axis=0)
    return batch.astype(np.float32)


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Reverse normalization to recover displayable image.

    Args:
        image: Normalized image array (C, H, W) or (H, W, C).
        mean: Per-channel mean used during normalization.
        std: Per-channel std used during normalization.

    Returns:
        Denormalized image with values in [0, 255] as uint8.
    """
    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)

    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        # CHW -> HWC
        img = np.transpose(image, (1, 2, 0))
    else:
        img = image

    img = img * std_arr + mean_arr
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255).astype(np.uint8)
    return img