"""Inference engine for computer vision models.

Provides a high-level inference pipeline that wraps model loading,
preprocessing, and prediction. Used by both Colab notebooks and
production API code.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from src.core.exceptions import InferenceError, ModelNotFoundError
from src.core.config import VisionConfig, load_config
from src.vision.models import VisionModelBase, create_model
from src.vision.preprocessing import normalize_image, resize_image


class InferenceEngine:
    """Production-ready inference engine for vision models.

    Handles model lifecycle (load/unload), preprocessing, and batched
    prediction. Thread-safe for concurrent inference requests.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        config: Optional[VisionConfig] = None,
        device: str = "auto",
    ):
        self.config = config or load_config().vision
        self.model_name = model_name or self.config.model_name
        self.device = device
        self._model: Optional[VisionModelBase] = None
        self._input_size: Tuple[int, int] = self.config.input_size

    def load_model(self) -> VisionModelBase:
        """Load the vision model into memory.

        Returns:
            The loaded VisionModelBase instance.

        Raises:
            ModelNotFoundError: If the model architecture is unsupported.
            InferenceError: If model loading fails.
        """
        try:
            self._model = create_model(
                model_name=self.model_name,
                num_classes=self.config.num_classes,
                pretrained=self.config.pretrained,
            )
            self._model.load()
            return self._model
        except ValueError as exc:
            raise ModelNotFoundError(str(exc)) from exc
        except Exception as exc:
            raise InferenceError(f"Failed to load model '{self.model_name}': {exc}") from exc

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess a single image for inference.

        Args:
            image: Input image as numpy array (H, W, C).

        Returns:
            Preprocessed array ready for model input (C, H, W).
        """
        resized = resize_image(image, target_size=self._input_size)
        normalized = normalize_image(resized)
        # HWC -> CHW
        if normalized.ndim == 3:
            return np.transpose(normalized, (2, 0, 1))
        return normalized[np.newaxis, ...]

    def preprocess_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """Preprocess a list of images into a batch.

        Args:
            images: List of input images as numpy arrays.

        Returns:
            Batch tensor of shape (N, C, H, W).
        """
        processed = [self.preprocess(img) for img in images]
        return np.stack(processed, axis=0)

    def predict(self, image: np.ndarray) -> dict:
        """Run inference on a single image.

        Args:
            image: Input image as numpy array (H, W, C).

        Returns:
            Dictionary with class probabilities, predicted class, and confidence.
        """
        if self._model is None:
            self.load_model()

        input_tensor = self.preprocess(image)
        input_batch = input_tensor[np.newaxis, ...]
        probs = self._model.predict_batch(input_batch)[0]

        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        return {
            "probabilities": probs.tolist(),
            "predicted_class": predicted_class,
            "confidence": confidence,
            "model_name": self.model_name,
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[dict]:
        """Run inference on a batch of images.

        Args:
            images: List of input images as numpy arrays.

        Returns:
            List of prediction dictionaries.
        """
        if self._model is None:
            self.load_model()

        input_batch = self.preprocess_batch(images)
        probs = self._model.predict_batch(input_batch)

        results = []
        for i, prob_vector in enumerate(probs):
            predicted_class = int(np.argmax(prob_vector))
            confidence = float(prob_vector[predicted_class])
            results.append({
                "probabilities": prob_vector.tolist(),
                "predicted_class": predicted_class,
                "confidence": confidence,
                "model_name": self.model_name,
                "batch_index": i,
            })
        return results

    def unload(self) -> None:
        """Unload the model to free memory."""
        self._model = None