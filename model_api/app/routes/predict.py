"""Inference endpoint — receives a base64 image, returns a binary mask PNG."""

import base64
import io
import logging

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from app.model_manager import manager

log = logging.getLogger(__name__)

router = APIRouter()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DIVISOR = 16  # UNet++ with 4 pool layers needs spatial dims divisible by 16

class PredictRequest(BaseModel):
    image_base64: str


def _preprocess(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32) / 255.0  # (H, W, C)
    arr = (arr - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32,
    )
    return np.transpose(arr, (2, 0, 1))  # (C, H, W)


def _pad_to_divisor(array: np.ndarray, divisor: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Pad a (C, H, W) array so H and W are divisible by *divisor*."""
    _, h, w = array.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h or pad_w:
        array = np.pad(
            array,
            ((0, 0), (0, pad_h), (0, pad_w)),
            mode="reflect",
        )
    return array, (h, w)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@router.post("/predict")
def predict(request: PredictRequest) -> Response:
    session = manager.session
    input_name = manager.input_name
    output_name = manager.output_name
    if session is None or input_name is None or output_name is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc

    input_arr = _preprocess(image)
    input_arr, (orig_h, orig_w) = _pad_to_divisor(input_arr, DIVISOR)
    batch = np.expand_dims(input_arr, axis=0).astype(np.float32)  # (1, C, H, W)

    logits = session.run([output_name], {input_name: batch})[0]
    probs = _sigmoid(logits[0, 0, :orig_h, :orig_w])
    mask = probs > 0.5
    mask_uint8 = (mask * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(mask_uint8, mode="L").save(buf, format="PNG")

    return Response(content=buf.getvalue(), media_type="image/png")
