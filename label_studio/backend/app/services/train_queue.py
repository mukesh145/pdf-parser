"""Redis-backed queue for training jobs."""

from typing import Optional

import redis

from app.models.label_job import TrainJob


class TrainQueue:
    """Push / poll training jobs via a Redis list."""

    def __init__(self, redis_url: str, queue_name: str) -> None:
        self._queue_name = queue_name
        self._processing_key = f"{queue_name}:processing"
        self._client: redis.Redis = redis.Redis.from_url(
            redis_url, decode_responses=True
        )

    def enqueue(self, job: TrainJob) -> None:
        self._client.rpush(self._queue_name, job.model_dump_json())

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
