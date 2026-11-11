"""Gesture recognition routes."""

from __future__ import annotations

import base64
import io

from fastapi import APIRouter, HTTPException
import numpy as np

from src.api.schemas.gesture import (
    GestureRequest,
    GestureResponse,
    GestureResult,
)
from src.api.services.gesture_service import GestureService

router = APIRouter(tags=["gesture"])

gesture_service: GestureService = GestureService()


def _decode_image(image_base64: str, fmt: str) -> np.ndarray:
    try:
        data = base64.b64decode(image_base64)
        stream = io.BytesIO(data)
        from PIL import Image

        img = Image.open(stream)
        img = img.convert("RGB")
        return np.array(img)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")


@router.post("/gesture", response_model=GestureResponse)
async def recognize_gesture(request: GestureRequest) -> GestureResponse:
    """Recognize gestures in an image."""
    image = _decode_image(request.image_base64, request.format)
    results_data = gesture_service.recognize(image)

    results = [
        GestureResult(
            gesture=r["gesture"],
            confidence=r["confidence"],
            handedness=r["handedness"],
            bounding_box=r.get("bounding_box"),
        )
        for r in results_data
    ]

    return GestureResponse(
        results=results,
        num_hands=len(results),
        available=gesture_service.is_available(),
    )
