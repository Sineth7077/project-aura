"""YOLO-based object detection module.

Provides a reusable interface for YOLO object detection with configurable
weights, CPU/GPU inference, and integration with agent observations.

This module uses Ultralytics YOLO models for object detection. It supports
local checkpoint paths and handles missing weights gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Import observation schema for integration
try:
    from src.agents.observation import Observation, ObservationType
    _OBSERVATION_AVAILABLE = True
except ImportError:
    _OBSERVATION_AVAILABLE = False


@dataclass
class Detection:
    """A single object detection result."""
    
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2] in normalized coordinates [0, 1]
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox,
        }


@dataclass
class DetectionResult:
    """Complete detection result for an image."""
    
    detections: List[Detection]
    person_count: int
    image_size: tuple[int, int]  # (width, height)
    model_name: str
    inference_time_ms: float
    
    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "detections": [d.to_dict() for d in self.detections],
            "person_count": self.person_count,
            "image_size": self.image_size,
            "model_name": self.model_name,
            "inference_time_ms": self.inference_time_ms,
        }


class ObjectDetector:
    """YOLO-based object detector with configurable weights and inference settings.
    
    This class provides a reusable interface for object detection using Ultralytics
    YOLO models. It supports CPU and CUDA inference, local checkpoint paths, and
    handles missing weights gracefully.
    
    Example:
        detector = ObjectDetector(
            model_path="yolov8n.pt",
            confidence_threshold=0.5,
            device="cpu"
        )
        result = detector.detect(image)
        print(f"Detected {result.person_count} persons")
    """
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.5,
        device: str = "cpu",
        image_size: int = 640,
    ) -> None:
        """Initialize the object detector.
        
        Args:
            model_path: Path to YOLO model weights (local file or Ultralytics model name).
            confidence_threshold: Minimum confidence for detections.
            device: Inference device ("cpu" or "cuda").
            image_size: Inference image size (square).
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.image_size = image_size
        self._model: Optional[Any] = None
        self._is_loaded = False
        self._load_error: Optional[str] = None
    
    def load_model(self) -> bool:
        """Load the YOLO model.
        
        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            self._load_error = "ultralytics package not installed. Install with: pip install ultralytics"
            logger.error(self._load_error)
            return False
        
        try:
            self._model = YOLO(self.model_path)
            self._is_loaded = True
            logger.info(f"Loaded YOLO model from {self.model_path}")
            return True
        except Exception as e:
            self._load_error = f"Failed to load model: {e}"
            logger.error(self._load_error)
            return False
    
    def detect(self, image: Any) -> Optional[DetectionResult]:
        """Run object detection on an image.
        
        Args:
            image: Input image (numpy array or PIL Image).
        
        Returns:
            DetectionResult with detections, or None if model not loaded or inference fails.
        """
        if not self._is_loaded:
            if not self.load_model():
                logger.error("Model not available for detection")
                return None
        
        import time
        import numpy as np
        from PIL import Image
        
        # Convert to numpy array if needed
        if hasattr(image, 'mode'):  # PIL Image
            image = np.array(image)
        
        # Get image size
        if len(image.shape) == 3:
            h, w = image.shape[:2]
        else:
            h, w = image.shape
        image_size = (w, h)
        
        try:
            start_time = time.time()
            
            # Run inference
            results = self._model(
                image,
                conf=self.confidence_threshold,
                device=self.device,
                imgsz=self.image_size,
                verbose=False,
            )
            
            inference_time_ms = (time.time() - start_time) * 1000.0
            
            # Parse results
            detections = []
            person_count = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class label
                        cls_id = int(box.cls[0]) if box.cls is not None else 0
                        label = self._model.names.get(cls_id, "unknown")
                        
                        # Get confidence
                        conf = float(box.conf[0]) if box.conf is not None else 0.0
                        
                        # Get bounding box (normalized coordinates)
                        xyxy = box.xyxy[0].cpu().numpy() if box.xyxy is not None else [0, 0, 0, 0]
                        x1, y1, x2, y2 = xyxy
                        bbox = [
                            float(x1 / w),
                            float(y1 / h),
                            float(x2 / w),
                            float(y2 / h),
                        ]
                        
                        detection = Detection(label=label, confidence=conf, bbox=bbox)
                        detections.append(detection)
                        
                        # Count persons
                        if label.lower() == "person":
                            person_count += 1
            
            return DetectionResult(
                detections=detections,
                person_count=person_count,
                image_size=image_size,
                model_name=self.model_path,
                inference_time_ms=inference_time_ms,
            )
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return None
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._is_loaded
    
    def get_load_error(self) -> Optional[str]:
        """Get the error message if model loading failed."""
        return self._load_error
    
    def get_model_info(self) -> dict[str, Any]:
        """Get model information."""
        return {
            "model_path": self.model_path,
            "is_loaded": self._is_loaded,
            "confidence_threshold": self.confidence_threshold,
            "device": self.device,
            "image_size": self.image_size,
            "load_error": self._load_error,
        }
    
    def to_observation(self, detection_result: DetectionResult, source: str = "yolo_detector") -> Optional[Any]:
        """Convert DetectionResult to agent Observation.
        
        Args:
            detection_result: DetectionResult from detect() method.
            source: Source identifier for the observation.
        
        Returns:
            Observation object if observation schema is available, None otherwise.
        """
        if not _OBSERVATION_AVAILABLE:
            logger.warning("Observation schema not available, cannot convert to Observation")
            return None
        
        obs_data = {
            "detections": [d.to_dict() for d in detection_result.detections],
            "person_count": detection_result.person_count,
            "image_size": detection_result.image_size,
            "model_name": detection_result.model_name,
            "inference_time_ms": detection_result.inference_time_ms,
        }
        
        # Use the highest confidence among detections as the observation confidence
        confidences = [d.confidence for d in detection_result.detections]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return Observation(
            source=source,
            type=ObservationType.OBJECT_DETECTION,
            data=obs_data,
            confidence=avg_confidence,
            metadata={"model": detection_result.model_name, "device": self.device},
        )
