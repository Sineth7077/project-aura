"""Temporal gesture smoothing for stable gesture recognition.

Provides a :class:`GestureSmoother` that accumulates a rolling history of
gesture predictions and applies majority voting, minimum stable-frame
requirements, and configurable cooldowns to prevent jittery gesture changes.

This is essential for turning per-frame MediaPipe predictions into reliable
gesture events that an autonomous agent can act upon.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from src.vision.gesture_recognizer import (
    GesturePrediction,
    UNKNOWN,
)


@dataclass
class SmoothedGesture:
    """Result of temporal smoothing.

    Attributes:
        gesture: The stabilized gesture name.
        confidence: Average confidence over the voting window.
        stable: Whether the gesture has met the minimum stable-frame count.
        transition: True if the gesture changed since the last smoothed output.
        event_suppressed: True if a cooldown suppressed a new event.
        history_length: Number of predictions in the current window.
    """

    gesture: str
    confidence: float
    stable: bool = False
    transition: bool = False
    event_suppressed: bool = False
    history_length: int = 0


class GestureSmoother:
    """Temporal smoother for gesture predictions.

    Maintains a rolling window of recent predictions and emits a stable
    gesture only when a majority of the window agrees and a minimum number of
    consecutive frames support the same gesture. A cooldown period prevents
    duplicate events from firing too quickly.
    """

    def __init__(
        self,
        window_size: int = 5,
        min_stable_frames: int = 3,
        cooldown_frames: int = 10,
    ) -> None:
        """Initialize the smoother.

        Args:
            window_size: Rolling history window size (must be >= 1).
            min_stable_frames: Minimum consecutive frames for stability
                (must be >= 1).
            cooldown_frames: Frames to wait before emitting a new gesture
                event after a change (must be >= 0).
        """
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if min_stable_frames < 1:
            raise ValueError("min_stable_frames must be >= 1")
        if cooldown_frames < 0:
            raise ValueError("cooldown_frames must be >= 0")

        self.window_size = window_size
        self.min_stable_frames = min_stable_frames
        self.cooldown_frames = cooldown_frames

        self._history: deque[GesturePrediction] = deque(maxlen=window_size)
        self._last_gesture: str = UNKNOWN
        self._stable_count: int = 0
        self._cooldown_counter: int = 0
        self._last_emitted_gesture: str = UNKNOWN

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def update(self, prediction: GesturePrediction) -> SmoothedGesture:
        """Feed a new frame prediction and return the smoothed result.

        Args:
            prediction: A :class:`GesturePrediction` from the recognizer.

        Returns:
            A :class:`SmoothedGesture` with the stabilized gesture and
            metadata about transitions and cooldown suppression.
        """
        self._history.append(prediction)
        raw_gesture = prediction.gesture

        # Update cooldown counter.
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # Track consecutive frames with the same raw gesture.
        if raw_gesture == self._last_gesture:
            self._stable_count += 1
        else:
            self._stable_count = 1
            self._last_gesture = raw_gesture

        # Majority vote over the window.
        voted_gesture, voted_confidence = self._majority_vote()

        # Determine stability.
        stable = self._stable_count >= self.min_stable_frames

        # The output gesture is the voted gesture only if stable; otherwise
        # we hold the last emitted gesture to avoid flicker.
        if stable:
            output_gesture = voted_gesture
            output_confidence = voted_confidence
        else:
            output_gesture = self._last_emitted_gesture
            output_confidence = self._last_emitted_confidence()

        # Detect transition (change from last emitted gesture).
        transition = output_gesture != self._last_emitted_gesture

        # Cooldown: suppress new events during cooldown window.
        event_suppressed = False
        if transition:
            if self._cooldown_counter > 0:
                event_suppressed = True
                output_gesture = self._last_emitted_gesture
                output_confidence = self._last_emitted_confidence()
                transition = False
            else:
                # Emit new gesture and start cooldown.
                self._last_emitted_gesture = output_gesture
                self._cooldown_counter = self.cooldown_frames

        return SmoothedGesture(
            gesture=output_gesture,
            confidence=output_confidence,
            stable=stable,
            transition=transition and not event_suppressed,
            event_suppressed=event_suppressed,
            history_length=len(self._history),
        )

    def get_gesture_history(self) -> list[GesturePrediction]:
        """Return a copy of the current rolling history."""
        return list(self._history)

    def reset(self) -> None:
        """Reset all smoothing state."""
        self._history.clear()
        self._last_gesture = UNKNOWN
        self._stable_count = 0
        self._cooldown_counter = 0
        self._last_emitted_gesture = UNKNOWN

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _majority_vote(self) -> tuple[str, float]:
        """Return the most common gesture in the window and average confidence."""
        if not self._history:
            return UNKNOWN, 0.0

        votes: dict[str, int] = {}
        conf_sum: dict[str, float] = {}
        for pred in self._history:
            votes[pred.gesture] = votes.get(pred.gesture, 0) + 1
            conf_sum[pred.gesture] = conf_sum.get(pred.gesture, 0.0) + pred.confidence

        best_gesture = UNKNOWN
        best_votes = -1
        best_confidence = 0.0
        for gesture, count in votes.items():
            avg_conf = conf_sum[gesture] / count
            if count > best_votes or (
                count == best_votes and avg_conf > best_confidence
            ):
                best_gesture = gesture
                best_votes = count
                best_confidence = avg_conf

        return best_gesture, best_confidence

    def _last_emitted_confidence(self) -> float:
        """Return the confidence of the last emitted gesture (approx.)."""
        for pred in reversed(self._history):
            if pred.gesture == self._last_emitted_gesture:
                return pred.confidence
        return 0.0
