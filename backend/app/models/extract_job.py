"""Data Transfer Object for an image extraction job."""

from datetime import datetime

from pydantic import BaseModel, Field


class ExtractJob(BaseModel):
    """Describes a single PDF image-extraction job passed through the queue."""

    job_id: str
    pdf_path: str
    zip_output_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
