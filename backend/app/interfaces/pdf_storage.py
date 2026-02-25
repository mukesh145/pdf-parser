"""Abstract interface for PDF file storage."""

from abc import ABC, abstractmethod


class IPdfStorage(ABC):
    """Contract for saving, retrieving, and deleting raw PDF files."""

    @abstractmethod
    def save(self, pdf_bytes: bytes, path: str) -> str:
        """Persist PDF bytes at the given path and return the stored path."""
        ...

    @abstractmethod
    def get(self, path: str) -> bytes:
        """Retrieve raw PDF bytes from the given path."""
        ...

    @abstractmethod
    def delete(self, path: str) -> None:
        """Remove the PDF file at the given path."""
        ...
