"""Abstract interface for extracted-image storage and zip packaging."""

from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class IImageStorage(ABC):
    """Contract for saving extracted images, zipping them, and retrieving the zip."""

    @abstractmethod
    def save_images(self, job_id: str, page_idx: int, images: List[Image.Image]) -> None:
        """Save a list of PIL images for a given job and page index."""
        ...

    @abstractmethod
    def zip_directory(self, job_id: str, output_path: str) -> None:
        """Compress all images for a job into a zip file at output_path."""
        ...

    @abstractmethod
    def exists(self, path: str) -> bool:
        """Check whether a file exists at the given path."""
        ...

    @abstractmethod
    def get_file(self, path: str) -> bytes:
        """Read and return raw bytes of the file at the given path."""
        ...
