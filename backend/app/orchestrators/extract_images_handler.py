"""Orchestrator that handles the API-facing upload and enqueue flow."""

import uuid

from app.interfaces.pdf_storage import IPdfStorage
from app.interfaces.request_queue import IRequestQueue
from app.models.extract_job import ExtractJob


class ExtractImagesHandler:
    """Orchestrate the upload → enqueue lifecycle.

    Used by the API layer. Depends only on abstractions (IPdfStorage,
    IRequestQueue).
    """

    def __init__(
        self,
        pdf_storage: IPdfStorage,
        queue: IRequestQueue,
    ) -> None:
        self._pdf_storage = pdf_storage
        self._queue = queue

    def handle_upload(self, pdf_bytes: bytes) -> ExtractJob:
        """Save the PDF to storage, create an ExtractJob, and enqueue it."""
        job_id = uuid.uuid4().hex
        pdf_rel_path = f"{job_id}.pdf"
        zip_rel_path = f"{job_id}.zip"

        self._pdf_storage.save(pdf_bytes, pdf_rel_path)

        job = ExtractJob(
            job_id=job_id,
            pdf_path=pdf_rel_path,
            zip_output_path=zip_rel_path,
        )
        self._queue.enqueue(job)
        return job
