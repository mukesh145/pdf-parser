"""Data Transfer Object for a completed job result."""

from typing import Optional

from pydantic import BaseModel


class JobResult(BaseModel):
    """Describes the outcome of a processed extraction job."""

    job_id: str
    zip_path: str
    success: bool
    error_message: Optional[str] = None
