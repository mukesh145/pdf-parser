"""Redis-backed queue consumer for training jobs.

Mirror of the producer in label_studio/backend/app/services/train_queue.py
so that both ends agree on message format and queue semantics.
"""

from typing import List, Optional

import redis
from pydantic import BaseModel


class LabelPair(BaseModel):
    """A single image-mask file pair."""

    image_filename: str
    mask_filename: str


class TrainJob(BaseModel):
    """Message pushed to Redis when the user submits labelled masks."""

    job_id: str
    pairs: List[LabelPair]


class TrainQueueConsumer:
    """Poll / ack / nack training jobs from a Redis list."""

    def __init__(self, redis_url: str, queue_name: str) -> None:
        self._queue_name = queue_name
        self._processing_key = f"{queue_name}:processing"
        self._client: redis.Redis = redis.Redis.from_url(
            redis_url, decode_responses=True
        )

    def poll(self) -> Optional[TrainJob]:
        raw: Optional[str] = self._client.lpop(self._queue_name)
        if raw is None:
            return None
        job = TrainJob.model_validate_json(raw)
        self._client.hset(self._processing_key, job.job_id, raw)
        return job

    def ack(self, job_id: str) -> None:
        self._client.hdel(self._processing_key, job_id)

    def nack(self, job_id: str) -> None:
        raw: Optional[str] = self._client.hget(self._processing_key, job_id)
        if raw is not None:
            self._client.rpush(self._queue_name, raw)
            self._client.hdel(self._processing_key, job_id)
