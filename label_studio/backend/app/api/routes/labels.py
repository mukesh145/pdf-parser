"""API routes for image upload, listing, and mask submission."""

import base64
import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from app.config import settings
from app.models.label_job import LabelPair, TrainJob
from app.services.storage import (
    clear_uploads,
    copy_pair_to_train_dir,
    extract_zip,
    list_uploaded_images,
    save_upload,
)
from app.services.train_queue import TrainQueue

router = APIRouter()

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "/app/frontend"))

_queue: TrainQueue | None = None


def _get_queue() -> TrainQueue:
    global _queue
    if _queue is None:
        _queue = TrainQueue(settings.redis_url, settings.train_queue_name)
    return _queue


@router.get("/", response_class=HTMLResponse)
async def index():
    """Serve the single-page labelling UI."""
    html = (FRONTEND_DIR / "templates" / "index.html").read_text()
    return HTMLResponse(content=html)


@router.post("/clear")
async def clear_images():
    """Delete all previously uploaded images from the staging area."""
    clear_uploads()
    return {"status": "cleared"}


@router.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """Accept one or more image files, or a single zip archive."""
    saved: List[str] = []
    for f in files:
        data = await f.read()
        if f.filename and f.filename.lower().endswith(".zip"):
            saved.extend(extract_zip(data))
        else:
            name = f.filename or f"image_{uuid.uuid4().hex[:8]}.png"
            save_upload(name, data)
            saved.append(name)
    return {"uploaded": saved}


@router.get("/images")
async def get_images():
    """Return the list of uploaded image filenames."""
    return {"images": list_uploaded_images()}


@router.get("/images/{filename}")
async def serve_image(filename: str):
    """Serve a single uploaded image by name."""
    path = Path(settings.upload_dir) / filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


class MaskPayload(BaseModel):
    image_filename: str
    mask_data_url: str  # data:image/png;base64,...


class SubmitRequest(BaseModel):
    masks: List[MaskPayload]


@router.post("/submit")
async def submit_masks(req: SubmitRequest):
    """Save image+mask pairs to train volume and enqueue a training job."""
    if not req.masks:
        raise HTTPException(status_code=400, detail="No masks provided")

    pairs: List[LabelPair] = []
    for item in req.masks:
        header, encoded = item.mask_data_url.split(",", 1)
        mask_bytes = base64.b64decode(encoded)

        copy_pair_to_train_dir(item.image_filename, mask_bytes)

        mask_name = Path(item.image_filename).stem + "_mask.png"
        pairs.append(
            LabelPair(image_filename=item.image_filename, mask_filename=mask_name)
        )

    job = TrainJob(job_id=uuid.uuid4().hex, pairs=pairs)
    _get_queue().enqueue(job)

    return {"job_id": job.job_id, "pairs": len(pairs)}
