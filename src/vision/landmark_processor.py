"""Landmark processing utilities for hand gesture recognition.

Provides pure-geometry functions for working with MediaPipe hand landmarks:
normalized coordinate handling, bounding-box computation, finger-extension
detection, distance and angle calculations, and handedness-aware coordinate
flipping (so left-hand landmarks mirror right-hand conventions).

All functions operate on plain Python/numpy data and do not require MediaPipe
or OpenCV at import time, making the module easy to unit-test with synthetic
landmark data.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

# MediaPipe hand landmark indices (0-based, 21 total)
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_DIP = 15
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

FINGER_TIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_PIPS = [THUMB_IP, INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
FINGER_MCPS = [THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]

PALM_LANDMARKS = [WRIST, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]


def validate_landmarks(
    landmarks: list[tuple[float, float, float]],
    require_exact_count: bool = True,
) -> bool:
    """Validate a list of hand landmarks.

    Args:
        landmarks: List of (x, y, z) tuples in normalized [0, 1] space.
        require_exact_count: If True, exactly 21 landmarks are required.

    Returns:
        True if the landmarks are structurally valid.
    """
    if not isinstance(landmarks, (list, tuple)):
        return False
    if require_exact_count and len(landmarks) != 21:
        return False
    if len(landmarks) == 0:
        return False
    for lm in landmarks:
        if not isinstance(lm, (list, tuple)) or len(lm) < 2:
            return False
        x, y = lm[0], lm[1]
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
            return False
        if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
            return False
    return True


def to_numpy(
    landmarks: list[tuple[float, float, float]],
) -> np.ndarray:
    """Convert a list of (x, y, z) landmarks to an (N, 3) float array."""
    if not validate_landmarks(landmarks, require_exact_count=False):
        return np.zeros((0, 3), dtype=np.float64)
    return np.array(
        [[float(lm[0]), float(lm[1]), float(lm[2])] for lm in landmarks],
        dtype=np.float64,
    )


def compute_bounding_box(
    landmarks: list[tuple[float, float, float]],
    image_width: int,
    image_height: int,
    padding: float = 0.0,
) -> list[int]:
    """Compute a pixel-space bounding box from normalized landmarks."""
    if not validate_landmarks(landmarks):
        return [0, 0, 0, 0]
    arr = to_numpy(landmarks)
    xs = arr[:, 0] * image_width
    ys = arr[:, 1] * image_height
    xmin, xmax = float(xs.min()), float(xs.max())
    ymin, ymax = float(ys.min()), float(ys.max())

    if padding > 0.0:
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = (xmax - xmin) * (1.0 + 2.0 * padding)
        h = (ymax - ymin) * (1.0 + 2.0 * padding)
        xmin = cx - w / 2.0
        xmax = cx + w / 2.0
        ymin = cy - h / 2.0
        ymax = cy + h / 2.0

    xmin = max(0, int(xmin))
    ymin = max(0, int(ymin))
    xmax = min(image_width, int(xmax))
    ymax = min(image_height, int(ymax))
    return [xmin, ymin, xmax, ymax]


def palm_center(
    landmarks: list[tuple[float, float, float]],
) -> Optional[tuple[float, float, float]]:
    """Return the average (x, y, z) of the palm-defining landmarks."""
    if not validate_landmarks(landmarks):
        return None
    arr = to_numpy(landmarks)
    palm = arr[PALM_LANDMARKS].mean(axis=0)
    return (float(palm[0]), float(palm[1]), float(palm[2]))


def distance(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )


def distance_2d(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
) -> float:
    """Euclidean distance between two points using only x, y coordinates."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle_between(
    p1: tuple[float, float, float],
    p2: tuple[float, float, float],
    p3: tuple[float, float, float],
) -> float:
    """Return the angle (in degrees) at p2 formed by p1-p2-p3."""
    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    norm1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    norm2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (norm1 * norm2)))
    return math.degrees(math.acos(cos_angle))



def is_finger_extended(
    landmarks: list[tuple[float, float, float]],
    finger_tip: int,
    finger_pip: int,
    finger_mcp: int,
    handedness: str = "Right",
) -> bool:
    """Determine whether a finger is extended using landmark geometry."""
    if not validate_landmarks(landmarks):
        return False

    tip = landmarks[finger_tip]
    pip = landmarks[finger_pip]
    mcp = landmarks[finger_mcp]

    if finger_tip == THUMB_TIP:
        if handedness == "Left":
            return tip[0] < landmarks[THUMB_IP][0]
        return tip[0] > landmarks[THUMB_IP][0]

    return tip[1] < pip[1]


def extended_fingers(
    landmarks: list[tuple[float, float, float]],
    handedness: str = "Right",
) -> list[bool]:
    """Return a list of 5 booleans indicating which fingers are extended.

    Order: [thumb, index, middle, ring, pinky].
    """
    if not validate_landmarks(landmarks):
        return [False, False, False, False, False]

    return [
        is_finger_extended(landmarks, THUMB_TIP, THUMB_IP, THUMB_MCP, handedness),
        is_finger_extended(landmarks, INDEX_TIP, INDEX_PIP, INDEX_MCP, handedness),
        is_finger_extended(landmarks, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP, handedness),
        is_finger_extended(landmarks, RING_TIP, RING_PIP, RING_MCP, handedness),
        is_finger_extended(landmarks, PINKY_TIP, PINKY_DIP, PINKY_MCP, handedness),
    ]


def normalize_handedness(
    landmarks: list[tuple[float, float, float]],
    handedness: str,
) -> list[tuple[float, float, float]]:
    """Mirror left-hand landmarks so they follow right-hand geometry."""
    if not validate_landmarks(landmarks):
        return []
    if handedness == "Left":
        return [(-x, y, z) for (x, y, z) in landmarks]
    return list(landmarks)

