"""Model service for lazy model loading.

Provides a singleton-style model service that loads expensive models on
first use (application lifecycle), caches them, and serves inference requests.
Designed for FastAPI's lifespan events.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np


class ModelService:
    """Manages vision model lifecycle for the API.

    Models are loaded lazily on first use and cached. Thread-safe access via
    a lock. Supports graceful degradation when heavy dependencies (PyTorch)
    are unavailable.
    """

    def __init__(
        self,
        model_name: str = "efficientnet_b0",
        num_classes: int = 10,
    ) -> None:
        self.model_name = model_name
        self.num_classes = num_classes
        self._engine = None
        self._lock = threading.Lock()
        self._loaded = False

    def is_loaded(self) -> bool:
        """Return True if the model has been initialized."""
        return self._loaded

    def load_model(self) -> bool:
        """Load the model into memory.

        Returns:
            True if the model was loaded successfully.
        """
        with self._lock:
            if self._loaded:
                return True
            try:
                from src.vision.inference import InferenceEngine

                self._engine = InferenceEngine(model_name=self.model_name)
                self._engine.load_model()
                self._loaded = True
                return True
            except Exception:
                self._loaded = False
                return False

    def predict(self, image: np.ndarray) -> Optional[dict]:
        """Run inference on an image.

        Args:
            image: RGB image array (H, W, 3).

        Returns:
            Prediction dict or None if the model is unavailable.
        """
        if not self._loaded:
            self.load_model()
        if not self._loaded or self._engine is None:
            return None
        try:
            return self._engine.predict(image)
        except Exception:
            return None

    def get_model_info(self) -> dict:
        """Return model metadata."""
        return {
            "model_name": self.model_name,
            "num_classes": self.num_classes,
            "loaded": self._loaded,
            "device": "cpu",
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        with self._lock:
            self._engine = None
            self._loaded = False
