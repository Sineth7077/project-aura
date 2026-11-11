"""Geometric gesture recognition for hand landmarks.

Implements rule-based recognition of seven gesture classes:

* OPEN_PALM    - all five fingers extended
* FIST         - no fingers extended
* THUMBS_UP    - only the thumb extended
* POINTING     - only the index finger extended
* VICTORY      - index and middle fingers extended (peace sign)
* PINCH        - thumb tip touching index fingertip
* UNKNOWN      - confidence below threshold or ambiguous pose

The recognizer is intentionally deterministic and explainable: every decision
is based on normalized landmark geometry (finger extension, tip distances,
joint angles) and includes a confidence score. It is **not** a neural network
and should not be treated as one.

Designed for use with :class:`src.vision.hand_tracker.HandTrackingResult`
outputs or synthetic landmark data in unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.core.exceptions import GestureRecognitionError
from src.vision.landmark_processor import (
    THUMB_TIP,
    INDEX_TIP,
    distance,
    extended_fingers,
    normalize_handedness,
    validate_landmarks,
)


# Gesture class names (canonical, uppercase)
OPEN_PALM = "OPEN_PALM"
FIST = "FIST"
THUMBS_UP = "THUMBS_UP"
POINTING = "POINTING"
VICTORY = "VICTORY"
PINCH = "PINCH"
UNKNOWN = "UNKNOWN"

ALL_GESTURES = (OPEN_PALM, FIST, THUMBS_UP, POINTING, VICTORY, PINCH, UNKNOWN)

DEFAULT_CONFIDENCE_THRESHOLD = 0.7
PINCH_DISTANCE_THRESHOLD = 0.05


@dataclass
class GesturePrediction:
    """Result of classifying a single hand's gesture.

    Attributes:
        gesture: One of the CANONICAL_GESTURE_NAMES constants.
        confidence: Confidence score in [0, 1]. 0.0 means UNKNOWN.
        handedness: "Left", "Right", or "Unknown".
        landmarks: The 21 normalized landmarks used for classification.
        feature_scores: Intermediate per-feature scores used for debugging.
    """

    gesture: str
    confidence: float
    handedness: str = "Unknown"
    landmarks: Optional[list[tuple[float, float, float]]] = None
    feature_scores: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gesture": self.gesture,
            "confidence": self.confidence,
            "handedness": self.handedness,
            "landmarks": self.landmarks,
            "feature_scores": self.feature_scores,
        }

    def is_confident(self, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> bool:
        return self.confidence >= threshold


class GestureRecognizer:
    """Rule-based geometric gesture recognizer.

    Classifies hand landmarks into one of seven gesture classes using
    normalized geometry. Handles left and right hands by normalizing
    handedness before applying right-handed rules.
    """

    def __init__(
        self,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        pinch_distance_threshold: float = PINCH_DISTANCE_THRESHOLD,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.pinch_distance_threshold = pinch_distance_threshold

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def recognize(
        self,
        landmarks: list[tuple[float, float, float]],
        handedness: str = "Right",
    ) -> GesturePrediction:
        if not validate_landmarks(landmarks):
            raise GestureRecognitionError(
                "Invalid landmark input: expected 21 (x, y, z) normalized landmarks."
            )

        norm_landmarks = normalize_handedness(landmarks, handedness)
        ext = extended_fingers(norm_landmarks, handedness="Right")
        thumb_ext, index_ext, middle_ext, ring_ext, pinky_ext = ext

        feature_scores: dict[str, float] = {}

        pinch_dist = distance(
            norm_landmarks[THUMB_TIP], norm_landmarks[INDEX_TIP]
        )
        pinch_confidence = max(0.0, 1.0 - (pinch_dist / self.pinch_distance_threshold))
        pinch_confidence = min(pinch_confidence, 1.0)
        feature_scores["pinch_distance"] = pinch_dist
        feature_scores["pinch_confidence"] = pinch_confidence

        num_extended = sum(ext)

        gesture: str
        confidence: float

        if pinch_confidence >= 0.75:
            gesture = PINCH
            confidence = pinch_confidence
        elif num_extended == 0:
            gesture = FIST
            confidence = 0.95
        elif num_extended == 5:
            gesture = OPEN_PALM
            confidence = 0.95
        elif num_extended == 1 and thumb_ext and not index_ext:
            gesture = THUMBS_UP
            confidence = 0.90
        elif num_extended == 1 and index_ext and not thumb_ext:
            gesture = POINTING
            confidence = 0.88
        elif num_extended == 2 and index_ext and middle_ext:
            gesture = VICTORY
            confidence = 0.90
        else:
            gesture = UNKNOWN
            confidence = 0.0

        if confidence < self.confidence_threshold and gesture != UNKNOWN:
            gesture = UNKNOWN
            confidence = 0.0

        feature_scores["num_extended"] = float(num_extended)
        feature_scores["finger_extensions"] = float(sum(ext))

        return GesturePrediction(
            gesture=gesture,
            confidence=confidence,
            handedness=handedness,
            landmarks=list(landmarks),
            feature_scores=feature_scores,
        )

    def recognize_many(
        self,
        hands: list[tuple[list[tuple[float, float, float]], str]],
    ) -> list[GesturePrediction]:
        """Classify multiple hands at once.

        Args:
            hands: List of (landmarks, handedness) tuples.

        Returns:
            List of :class:`GesturePrediction` objects. Invalid inputs yield
            UNKNOWN predictions with confidence 0.0 rather than raising.
        """
        predictions: list[GesturePrediction] = []
        for landmarks, handedness in hands:
            try:
                pred = self.recognize(landmarks, handedness)
            except GestureRecognitionError:
                pred = GesturePrediction(
                    gesture=UNKNOWN,
                    confidence=0.0,
                    handedness=handedness,
                    landmarks=list(landmarks) if landmarks else None,
                )
            predictions.append(pred)
        return predictions
