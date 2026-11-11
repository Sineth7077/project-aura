"""Inference routes."""

from __future__ import annotations

import base64
import io
import time

from fastapi import APIRouter, HTTPException
import numpy as np

from src.api.schemas.inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceResult,
)
from src.api.services.model_service import ModelService

router = APIRouter(tags=["inference"])

model_service: ModelService = ModelService()


def _decode_image(image_base64: str, fmt: str) -> np.ndarray:
    """Decode a base64-encoded image into a numpy array."""
    try:
        data = base64.b64decode(image_base64)
        stream = io.BytesIO(data)
        from PIL import Image

        img = Image.open(stream)
        img = img.convert("RGB")
        return np.array(img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")


@router.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest) -> InferenceResponse:
    """Run vision model inference on an image."""
    image = _decode_image(request.image_base64, request.format)

    start_time = time.time()
    prediction = model_service.predict(image)
    processing_ms = (time.time() - start_time) * 1000.0

    if prediction is None:
        raise HTTPException(
            status_code=503,
            detail="Model not available. Ensure dependencies are installed.",
        )

    result = InferenceResult(
        predicted_class=prediction["predicted_class"],
        confidence=prediction["confidence"],
        probabilities=prediction.get("probabilities"),
    )

    return InferenceResponse(
        results=[result],
        model_name=model_service.model_name,
        processing_time_ms=processing_ms,
    )
