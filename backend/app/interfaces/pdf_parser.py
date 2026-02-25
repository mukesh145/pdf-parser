"""Abstract interface for PDF parsing operations."""

from abc import ABC, abstractmethod
from typing import List


class IPdfParser(ABC):
    """Contract for splitting a PDF into pages and converting pages to base64."""

    @abstractmethod
    def split_into_pages(self, pdf_bytes: bytes) -> List[bytes]:
        """Split a full PDF into a list of single-page byte sequences."""
        ...

    @abstractmethod
    def page_to_base64(self, page_bytes: bytes) -> str:
        """Convert a single page's bytes to a base64-encoded string."""
        ...
