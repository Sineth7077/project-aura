"""Inference routes."""

from __future__ import annotations

import base64
import io
import time

from fastapi import APIRouter, HTTPException, Request
import numpy as np

from src.api.schemas.inference import (
    InferenceRequest,
    InferenceResponse,
    InferenceResult,
)
from src.api.services.model_service import ModelService
from src.api.config import load_api_config

router = APIRouter(tags=["inference"])

model_service: ModelService = ModelService()
config = load_api_config()


def _decode_image(image_base64: str, fmt: str) -> np.ndarray:
    """Decode a base64-encoded image into a numpy array."""
    try:
        data = base64.b64decode(image_base64)
        
        # Check image size before decoding
        if len(data) > config.max_image_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image size exceeds maximum allowed size of {config.max_image_size_mb} MB"
            )
        
        stream = io.BytesIO(data)
        from PIL import Image

        img = Image.open(stream)
        
        # Validate format
        if fmt and fmt.lower() not in ["jpeg", "jpg", "png", "webp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format: {fmt}. Supported formats: jpeg, jpg, png, webp"
            )
        
        img = img.convert("RGB")
        return np.array(img)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")


@router.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest, http_request: Request) -> InferenceResponse:
    """Run vision model inference on an image."""
    # Validate request size
    content_length = http_request.headers.get("content-length")
    if content_length and int(content_length) > config.max_image_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Request size exceeds maximum allowed size of {config.max_image_size_mb} MB"
        )
    
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
