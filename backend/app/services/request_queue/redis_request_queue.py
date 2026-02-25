"""Concrete IRequestQueue implementation backed by Redis."""

from typing import Optional

import redis

from app.interfaces.request_queue import IRequestQueue
from app.models.extract_job import ExtractJob


class RedisRequestQueue(IRequestQueue):
    """Use a Redis list as a FIFO job queue."""

    def __init__(self, redis_url: str, queue_name: str) -> None:
        """Initialise with Redis connection URL and queue key name."""

        self._redis_url = redis_url
        self._queue_name = queue_name
        self._processing_key = f"{queue_name}:processing"
        self._client: redis.Redis = redis.Redis.from_url(
            redis_url, decode_responses=True
        )

    def enqueue(self, job: ExtractJob) -> None:
        """Serialise the job and push it onto the Redis list."""
        self._client.rpush(self._queue_name, job.model_dump_json())

    def poll(self) -> Optional[ExtractJob]:
        """Pop the next job from the Redis list, or return None if empty."""
        raw: Optional[str] = self._client.lpop(self._queue_name)
        if raw is None:
            return None
        job = ExtractJob.model_validate_json(raw)
        self._client.hset(self._processing_key, job.job_id, raw)
        return job

    def ack(self, job_id: str) -> None:
        """Acknowledge that a job has been processed successfully."""
        self._client.hdel(self._processing_key, job_id)

    def nack(self, job_id: str) -> None:
        """Reject a job, making it available for retry."""
        raw: Optional[str] = self._client.hget(self._processing_key, job_id)
        if raw is not None:
            self._client.rpush(self._queue_name, raw)
            self._client.hdel(self._processing_key, job_id)
