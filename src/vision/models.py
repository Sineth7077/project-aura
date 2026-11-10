"""Computer vision model definitions for Project Aura 2.0.

Defines model architectures, loading utilities, and wrapper classes
for transfer learning-based vision models. Designed to work with
PyTorch when available, with graceful fallback for CPU-only Colab
environments.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class VisionModelBase:
    """Base class for vision models in Project Aura."""

    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self._model = None
        self._device = "cpu"

    def load(self) -> None:
        """Load the model. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement load()")

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement predict()")

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Run inference on a batch of images."""
        raise NotImplementedError("Subclasses must implement predict_batch()")

    @property
    def is_loaded(self) -> bool:
        """Return True if the model has been loaded."""
        return self._model is not None
class EfficientNetModel(VisionModelBase):
    """EfficientNet-based vision model wrapper.

    Uses torchvision EfficientNet when PyTorch is available.
    Falls back to a lightweight feature extractor when PyTorch is not
    installed (e.g., minimal Colab environments).
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        variant: str = "efficientnet_b0",
    ):
        super().__init__(variant, num_classes, pretrained)
        self.variant = variant

    def load(self) -> None:
        """Load EfficientNet model from torchvision."""
        try:
            import torch
            import torchvision.models as models

            model_fn = getattr(models, self.variant, None)
            if model_fn is None:
                raise ValueError(f"Unknown EfficientNet variant: {self.variant}")

            self._model = model_fn(pretrained=self.pretrained)
            # Replace classifier head
            in_features = self._model.classifier[1].in_features
            self._model.classifier[1] = torch.nn.Linear(in_features, self.num_classes)
            self._model.eval()
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
        except ImportError:
            # PyTorch not available - use lightweight fallback
            self._model = _LightweightVisionModel(self.num_classes)
            self._device = "cpu"

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Run inference on a single image.

        Args:
            image: Preprocessed image array (C, H, W) or (H, W, C).

        Returns:
            Class probability array of shape (num_classes,).
        """
        if not self.is_loaded:
            self.load()

        if isinstance(self._model, _LightweightVisionModel):
            return self._model.predict(image)

        import torch

        img = image
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            img = np.transpose(img, (2, 0, 1))
        img = img[np.newaxis, ...].astype(np.float32)
        tensor = torch.from_numpy(img).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()[0]

    def predict_batch(self, images: np.ndarray) -> np.ndarray:
        """Run inference on a batch of images.

        Args:
            images: Batch array of shape (N, C, H, W).

        Returns:
            Class probabilities of shape (N, num_classes).
        """
        if not self.is_loaded:
            self.load()

        if isinstance(self._model, _LightweightVisionModel):
            return np.stack([self._model.predict(img) for img in images])

        import torch

        tensor = torch.from_numpy(images.astype(np.float32)).to(self._device)
        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.cpu().numpy()


class _LightweightVisionModel:
    """Fallback vision model when PyTorch is unavailable.

    Uses simple statistical feature extraction for demonstration and
    CPU-only Colab environments. Not for production use.
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self._weights = np.random.RandomState(42).randn(num_classes, 3 * 224 * 224)
        self._weights = self._weights / np.linalg.norm(self._weights, axis=1, keepdims=True)

    def predict(self, image: np.ndarray) -> np.ndarray:
        """Generate class probabilities using random projection."""
        if image.ndim == 3:
            flat = image.flatten()
        else:
            flat = image.flatten()

        # Pad or truncate to expected size
        expected_size = self._weights.shape[1]
        if flat.size < expected_size:
            flat = np.pad(flat, (0, expected_size - flat.size))
        else:
            flat = flat[:expected_size]

        logits = self._weights @ flat
        # Softmax
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        return probs


def create_model(
    model_name: str = "efficientnet_b0",
    num_classes: int = 10,
    pretrained: bool = True,
) -> VisionModelBase:
    """Factory function to create a vision model.

    Args:
        model_name: Name of the model architecture.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.

    Returns:
        A VisionModelBase instance.

    Raises:
        ValueError: If model_name is not supported.
    """
    from src.core.constants import SUPPORTED_VISION_MODELS

    if model_name not in SUPPORTED_VISION_MODELS:
        raise ValueError(
            f"Unsupported model: {model_name}. "
            f"Supported: {SUPPORTED_VISION_MODELS}"
        )

    if model_name.startswith("efficientnet"):
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained, variant=model_name)
    else:
        # Default to EfficientNet for other architectures (fallback)
        return EfficientNetModel(num_classes=num_classes, pretrained=pretrained, variant="efficientnet_b0")
