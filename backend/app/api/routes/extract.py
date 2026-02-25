"""API routes for PDF image extraction."""

from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile, File
from fastapi.responses import JSONResponse, Response

from app.config import settings
from app.dependencies import get_extract_images_handler, get_image_storage
from app.interfaces.image_storage import IImageStorage
from app.orchestrators.extract_images_handler import ExtractImagesHandler

router = APIRouter()


@router.post("/", status_code=202, response_model=None)
async def post_extract(
    file: UploadFile = File(...),
    handler: ExtractImagesHandler = Depends(get_extract_images_handler),
):
    """Accept a PDF upload, enqueue an extraction job, and return the job id."""
    pdf_bytes = await file.read()
    job = handler.handle_upload(pdf_bytes)
    return {"job_id": job.job_id}


@router.get("/{job_id}")
def get_extract_result(
    job_id: str,
    image_storage: IImageStorage = Depends(get_image_storage),
) -> Response:
    """Poll endpoint: returns the zip when ready, or 202 while still processing."""
    zip_full_path = str(Path(settings.base_zip_path) / f"{job_id}.zip")

    if not image_storage.exists(zip_full_path):
        return JSONResponse(
            content={"status": "processing", "job_id": job_id},
            status_code=202,
        )

    zip_bytes = image_storage.get_file(zip_full_path)
    return Response(
        content=zip_bytes,
        status_code=200,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )
