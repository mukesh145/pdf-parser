"""Pydantic model for a training job pushed onto the Redis queue."""

from typing import List

from pydantic import BaseModel


class LabelPair(BaseModel):
    """A single image–mask file pair."""

    image_filename: str
    mask_filename: str


class TrainJob(BaseModel):
    """Message pushed to Redis when the user submits labelled masks."""

    job_id: str
    pairs: List[LabelPair]
