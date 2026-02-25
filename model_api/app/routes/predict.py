"""Inference endpoint — receives a base64 image, returns a binary mask PNG."""

import base64
import io
import logging

import numpy as np
import torch
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms

from app.model_manager import manager

log = logging.getLogger(__name__)

router = APIRouter()

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DIVISOR = 16  # UNet++ with 4 pool layers needs spatial dims divisible by 16

_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class PredictRequest(BaseModel):
    image_base64: str


def _pad_to_divisor(tensor: torch.Tensor, divisor: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad a (C, H, W) tensor so H and W are divisible by *divisor*.

    Returns the padded tensor and the original (H, W) for later cropping.
    """
    _, h, w = tensor.shape
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, (h, w)


@router.post("/predict")
async def predict(request: PredictRequest) -> Response:
    model = manager.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        image_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}") from exc

    input_tensor = _preprocess(image)
    input_tensor, (orig_h, orig_w) = _pad_to_divisor(input_tensor, DIVISOR)
    batch = input_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    batch = batch.to(device)

    with torch.no_grad():
        logits = model(batch)

    mask = (torch.sigmoid(logits[0, 0, :orig_h, :orig_w]) > 0.5).cpu().numpy()
    mask_uint8 = (mask * 255).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(mask_uint8, mode="L").save(buf, format="PNG")

    return Response(content=buf.getvalue(), media_type="image/png")
