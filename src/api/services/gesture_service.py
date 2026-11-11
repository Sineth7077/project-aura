"""Gesture recognition service for the API."""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class GestureService:
    """Manages hand tracking and gesture recognition for API requests."""

    def __init__(
        self,
        confidence_threshold: float = 0.7,
        max_num_hands: int = 2,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.max_num_hands = max_num_hands
        self._tracker = None
        self._recognizer = None
        self._smoother = None
        self._lock = threading.Lock()
        self._initialized = False

    def is_available(self) -> bool:
        """Return True if MediaPipe is available."""
        self._ensure_initialized()
        return self._tracker is not None and self._tracker.available

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return
            try:
                from src.vision.hand_tracker import HandTracker
                from src.vision.gesture_recognizer import GestureRecognizer
                from src.vision.gesture_smoother import GestureSmoother

                self._tracker = HandTracker(max_num_hands=self.max_num_hands)
                self._recognizer = GestureRecognizer(
                    confidence_threshold=self.confidence_threshold
                )
                self._smoother = GestureSmoother()
            except Exception:
                self._tracker = None
                self._recognizer = None
                self._smoother = None
            self._initialized = True

    def recognize(self, image: np.ndarray) -> list[dict]:
        """Recognize gestures in an image.

        Returns:
            List of gesture result dicts (empty if no hands detected or
            MediaPipe unavailable).
        """
        self._ensure_initialized()
        if not self.is_available() or self._tracker is None or self._recognizer is None:
            return []

        try:
            tracking_result = self._tracker.detect(image)
            results = []
            for hand in tracking_result.hands:
                if not hand.is_valid():
                    continue
                prediction = self._recognizer.recognize(
                    hand.landmarks, hand.handedness
                )
                results.append(
                    {
                        "gesture": prediction.gesture,
                        "confidence": prediction.confidence,
                        "handedness": hand.handedness,
                        "bounding_box": hand.bounding_box,
                    }
                )
            return results
        except Exception:
            return []

    def release(self) -> None:
        """Release tracker resources."""
        if self._tracker is not None:
            self._tracker.release()
