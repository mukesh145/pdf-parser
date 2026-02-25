"""Abstract interface for the job request queue."""

from abc import ABC, abstractmethod
from typing import Optional

from app.models.extract_job import ExtractJob


class IRequestQueue(ABC):
    """Contract for enqueueing, polling, acknowledging, and rejecting extraction jobs."""

    @abstractmethod
    def enqueue(self, job: ExtractJob) -> None:
        """Push a job onto the queue."""
        ...

    @abstractmethod
    def poll(self) -> Optional[ExtractJob]:
        """Attempt to claim the next job from the queue. Returns None if empty."""
        ...

    @abstractmethod
    def ack(self, job_id: str) -> None:
        """Acknowledge successful processing of a job."""
        ...

    @abstractmethod
    def nack(self, job_id: str) -> None:
        """Reject a job so it can be retried."""
        ...
