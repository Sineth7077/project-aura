"""Gesture recognition logic for Project Aura 2.0.

Provides hand and body gesture detection utilities. Integrates with
MediaPipe when available, with a lightweight fallback for Colab
environments without heavy ML dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GestureResult:
    """Result of a gesture recognition operation."""

    gesture_name: str
    confidence: float
    bounding_box: Optional[list[int]] = None
    landmarks: Optional[list[tuple[float, float, float]]] = None
    timestamp: float = field(default_factory=lambda: __import__("time").time())

    def to_dict(self) -> dict:
        return {
            "gesture_name": self.gesture_name,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "landmarks": self.landmarks,
            "timestamp": self.timestamp,
        }


class GestureRecognizer:
    """Gesture recognition engine.

    Uses MediaPipe Hands for real hand landmark detection when available.
    Falls back to a simple motion-based detector when MediaPipe is not
    installed (e.g., minimal Colab runtime).
    """

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        self.confidence_threshold = confidence_threshold
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._detector = None
        self._use_mediapipe = False
        self._prev_frame: Optional[np.ndarray] = None

    def _try_load_mediapipe(self) -> bool:
        """Attempt to initialize MediaPipe Hands detector."""
        try:
            import mediapipe as mp

            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._detector = self._mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._use_mediapipe = True
            return True
        except ImportError:
            self._use_mediapipe = False
            return False

    def detect(self, image: np.ndarray) -> list[GestureResult]:
        """Detect gestures in an image.

        Args:
            image: Input image as numpy array (H, W, C) in RGB format.

        Returns:
            List of GestureResult objects for detected gestures.
        """
        if self._detector is None and not self._use_mediapipe:
            self._try_load_mediapipe()

        if self._use_mediapipe and self._detector is not None:
            return self._detect_mediapipe(image)
        else:
            return self._detect_fallback(image)
    def _detect_mediapipe(self, image: np.ndarray) -> list[GestureResult]:
        """Detect gestures using MediaPipe Hands."""
        results = self._detector.process(image)
        gestures: list[GestureResult] = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = [
                    (lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark
                ]
                gesture_name, confidence = self._classify_gesture(landmarks)

                xs = [lm[0] for lm in landmarks]
                ys = [lm[1] for lm in landmarks]
                h, w = image.shape[:2]
                bbox = [
                    int(min(xs) * w),
                    int(min(ys) * h),
                    int(max(xs) * w),
                    int(max(ys) * h),
                ]

                gestures.append(
                    GestureResult(
                        gesture_name=gesture_name,
                        confidence=confidence,
                        bounding_box=bbox,
                        landmarks=landmarks,
                    )
                )

        return gestures

    def _detect_fallback(self, image: np.ndarray) -> list[GestureResult]:
        """Fallback gesture detection using frame differencing.

        Used when MediaPipe is unavailable. Detects gross motion
        as a proxy for gesture activity.
        """
        gray = self._to_grayscale(image)
        results: list[GestureResult] = []

        if self._prev_frame is not None:
            diff = np.abs(gray.astype(np.float32) - self._prev_frame.astype(np.float32))
            motion_score = float(np.mean(diff))

            if motion_score > 10.0:
                gesture_name = "motion_detected"
                confidence = min(motion_score / 50.0, 1.0)
                results.append(
                    GestureResult(
                        gesture_name=gesture_name,
                        confidence=confidence,
                    )
                )

        self._prev_frame = gray
        return results

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale."""
        if image.ndim == 3:
            return np.mean(image, axis=2).astype(np.uint8)
        return image.astype(np.uint8)

    @staticmethod
    def _classify_gesture(
        landmarks: list[tuple[float, float, float]],
    ) -> tuple[str, float]:
        """Classify a hand landmark set into a gesture name.

        Simple heuristic-based classification. In production, this would
        be replaced by a trained gesture classification model.
        """
        if not landmarks or len(landmarks) < 21:
            return "unknown", 0.0

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        middle_tip = landmarks[12]
        ring_tip = landmarks[16]
        pinky_tip = landmarks[20]

        thumb_mcp = landmarks[2]
        index_mcp = landmarks[5]
        middle_mcp = landmarks[9]
        ring_mcp = landmarks[13]
        pinky_mcp = landmarks[17]

        extended = [
            thumb_tip[1] < thumb_mcp[1],
            index_tip[1] < index_mcp[1],
            middle_tip[1] < middle_mcp[1],
            ring_tip[1] < ring_mcp[1],
            pinky_tip[1] < pinky_mcp[1],
        ]
        num_extended = sum(extended)

        if num_extended == 0:
            return "fist", 0.85
        elif num_extended == 5:
            return "open_palm", 0.90
        elif num_extended == 1 and extended[1]:
            return "pointing", 0.80
        elif num_extended == 2 and extended[1] and extended[2]:
            return "peace_sign", 0.85
        elif num_extended == 1 and extended[0]:
            return "thumbs_up", 0.75
        else:
            return "open_palm", 0.60

    def release(self) -> None:
        """Release detector resources."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None
