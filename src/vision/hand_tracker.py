"""Hand tracking module using MediaPipe Hands.

Provides a production-ready hand landmark detector that wraps MediaPipe's
Hands solution with graceful fallback, multi-hand support (up to two hands),
left/right handedness identification, bounding-box calculation, and
landmark visualization helpers.

Designed for Google Colab compatibility: MediaPipe is an optional dependency.
When it is not installed, the tracker reports itself as unavailable and callers
can degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class HandLandmarks:
    """Normalized 3D landmarks for a single detected hand.

    Attributes:
        landmarks: List of 21 (x, y, z) tuples in normalized [0, 1] image
            coordinates (z is depth relative to the wrist).
        handedness: "Left", "Right", or "Unknown" when MediaPipe does not
            provide handedness information.
        confidence: Detection confidence score in [0, 1].
        bounding_box: [xmin, ymin, xmax, ymax] in pixel coordinates of the
            original image, or None when unavailable.
    """

    landmarks: list[tuple[float, float, float]]
    handedness: str = "Unknown"
    confidence: float = 0.0
    bounding_box: Optional[list[int]] = None

    @property
    def num_landmarks(self) -> int:
        """Return the number of landmarks detected."""
        return len(self.landmarks)

    def is_valid(self) -> bool:
        """Return True if this hand has a complete 21-landmark set."""
        return self.num_landmarks == 21 and self.confidence > 0.0

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "landmarks": self.landmarks,
            "handedness": self.handedness,
            "confidence": self.confidence,
            "bounding_box": self.bounding_box,
            "num_landmarks": self.num_landmarks,
        }


@dataclass
class HandTrackingResult:
    """Result of a hand-tracking pass over a single image.

    Attributes:
        hands: List of detected :class:`HandLandmarks` objects (up to
            ``max_num_hands`` entries).
        image_shape: (height, width) of the processed image.
        tracker_available: Whether the underlying MediaPipe detector is active.
    """

    hands: list[HandLandmarks] = field(default_factory=list)
    image_shape: tuple[int, int] = (0, 0)
    tracker_available: bool = False

    @property
    def num_hands(self) -> int:
        """Return the number of hands detected."""
        return len(self.hands)

    def get_hand_by_handedness(self, handedness: str) -> Optional[HandLandmarks]:
        """Return the first hand matching the given handedness, or None."""
        for hand in self.hands:
            if hand.handedness == handedness:
                return hand
        return None

    def to_dict(self) -> dict:
        """Serialize to a plain dictionary."""
        return {
            "hands": [h.to_dict() for h in self.hands],
            "num_hands": self.num_hands,
            "image_shape": list(self.image_shape),
            "tracker_available": self.tracker_available,
        }


class HandTracker:
    """MediaPipe-based hand landmark detector.

    Detects up to ``max_num_hands`` hands per frame and returns 21 normalized
    landmarks per hand together with handedness and bounding boxes.

    When MediaPipe is unavailable, :attr:`available` is False and
    :meth:`detect` returns an empty result. This keeps the module importable
    in minimal environments (e.g. Colab CPU-only runtimes without MediaPipe).
    """

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        static_image_mode: bool = False,
    ) -> None:
        """Initialize the hand tracker.

        Args:
            max_num_hands: Maximum number of hands to detect (1 or 2).
            min_detection_confidence: Minimum confidence for hand detection.
            min_tracking_confidence: Minimum confidence for landmark tracking.
            static_image_mode: Whether to treat each image as independent
                (True) or as a video stream (False, enables tracking).
        """
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.static_image_mode = static_image_mode
        self._detector = None
        self._mp_hands = None
        self._mp_drawing = None
        self._mp_drawing_styles = None
        self.available = self._initialize_detector()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #
    def _initialize_detector(self) -> bool:
        """Attempt to initialize MediaPipe Hands.

        Returns:
            True if MediaPipe is available and the detector was created.
        """
        try:
            import mediapipe as mp  # type: ignore

            self._mp_hands = mp.solutions.hands
            self._mp_drawing = mp.solutions.drawing_utils
            self._mp_drawing_styles = mp.solutions.drawing_styles
            self._detector = self._mp_hands.Hands(
                static_image_mode=self.static_image_mode,
                max_num_hands=self.max_num_hands,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            return True
        except ImportError:
            self.available = False
            return False
        except Exception:
            # Any other initialization failure degrades gracefully.
            self.available = False
            return False

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    def detect(self, image: np.ndarray) -> HandTrackingResult:
        """Detect hands and landmarks in an image.

        Args:
            image: RGB image array of shape (H, W, 3). MediaPipe expects
                RGB input.

        Returns:
            A :class:`HandTrackingResult` containing detected hands.
        """
        height, width = image.shape[:2]
        result = HandTrackingResult(
            hands=[],
            image_shape=(height, width),
            tracker_available=self.available,
        )

        if not self.available or self._detector is None:
            return result

        try:
            mp_results = self._detector.process(image)

            if mp_results.multi_hand_landmarks:
                handedness_list = getattr(mp_results, "multi_handedness", None)
                for idx, hand_landmarks in enumerate(mp_results.multi_hand_landmarks):
                    landmarks = self._extract_landmarks(hand_landmarks)
                    handedness = self._extract_handedness(handedness_list, idx)
                    confidence = self._estimate_confidence(mp_results, idx)
                    bounding_box = self._compute_bounding_box(landmarks, width, height)

                    result.hands.append(
                        HandLandmarks(
                            landmarks=landmarks,
                            handedness=handedness,
                            confidence=confidence,
                            bounding_box=bounding_box,
                        )
                    )
        except Exception:
            # Tracking failures should not crash the pipeline.
            pass

        return result

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_landmarks(hand_landmarks) -> list[tuple[float, float, float]]:
        """Extract normalized (x, y, z) landmark tuples from MediaPipe output."""
        landmarks: list[tuple[float, float, float]] = []
        for landmark in hand_landmarks.landmark:
            landmarks.append((float(landmark.x), float(landmark.y), float(landmark.z)))
        return landmarks

    @staticmethod
    def _extract_handedness(multi_handedness, index: int) -> str:
        """Extract handedness label for a specific hand index."""
        if multi_handedness is None or index >= len(multi_handedness):
            return "Unknown"
        try:
            classification = multi_handedness[index].classification[0]
            return str(classification.label)
        except Exception:
            return "Unknown"

    @staticmethod
    def _estimate_confidence(mp_results, index: int) -> float:
        """Estimate detection confidence for a hand.

        MediaPipe Hands does not expose per-hand detection confidence directly
        in all versions, so we return a heuristic based on handedness score
        when available, otherwise a nominal value.
        """
        handedness_list = getattr(mp_results, "multi_handedness", None)
        if handedness_list is not None and index < len(handedness_list):
            try:
                return float(handedness_list[index].classification[0].score)
            except Exception:
                pass
        return 0.8

    @staticmethod
    def _compute_bounding_box(
        landmarks: list[tuple[float, float, float]],
        width: int,
        height: int,
    ) -> list[int]:
        """Compute pixel bounding box from normalized landmarks."""
        xs = [lm[0] for lm in landmarks]
        ys = [lm[1] for lm in landmarks]
        xmin = int(min(xs) * width)
        xmax = int(max(xs) * width)
        ymin = int(min(ys) * height)
        ymax = int(max(ys) * height)
        return [xmin, ymin, xmax, ymax]



    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    def draw_landmarks(
        self,
        image: np.ndarray,
        tracking_result: HandTrackingResult,
    ) -> np.ndarray:
        """Draw MediaPipe hand landmarks and connections onto an image.

        Args:
            image: RGB image array (H, W, 3) to draw on (modified in place
                and also returned).
            tracking_result: Result from :meth:`detect`.

        Returns:
            The image with landmarks drawn.
        """
        if not self.available or self._mp_drawing is None:
            return image

        try:
            import mediapipe as mp  # type: ignore

            annotated = image.copy()
            connections = self._mp_hands.HAND_CONNECTIONS

            for hand in tracking_result.hands:
                if not hand.is_valid():
                    continue
                h, w = annotated.shape[:2]
                # Draw connections
                for start_idx, end_idx in connections:
                    if start_idx >= len(hand.landmarks) or end_idx >= len(
                        hand.landmarks
                    ):
                        continue
                    x1 = int(hand.landmarks[start_idx][0] * w)
                    y1 = int(hand.landmarks[start_idx][1] * h)
                    x2 = int(hand.landmarks[end_idx][0] * w)
                    y2 = int(hand.landmarks[end_idx][1] * h)
                    cv2 = _import_cv2()
                    if cv2 is not None:
                        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # Draw landmark points
                for idx, (lx, ly, _lz) in enumerate(hand.landmarks):
                    cx = int(lx * w)
                    cy = int(ly * h)
                    cv2 = _import_cv2()
                    if cv2 is not None:
                        cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)
                        cv2.putText(
                            annotated,
                            str(idx),
                            (cx + 5, cy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.3,
                            (255, 255, 255),
                            1,
                        )
            return annotated
        except Exception:
            return image

    def release(self) -> None:
        """Release MediaPipe detector resources."""
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
            self._detector = None


def _import_cv2():
    """Lazy-import OpenCV, returning None when unavailable."""
    try:
        import cv2  # type: ignore

        return cv2
    except ImportError:
        return None
