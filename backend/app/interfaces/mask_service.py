"""Abstract interface for the U-Net segmentation mask service."""

from abc import ABC, abstractmethod


class IMaskService(ABC):
    """Contract for obtaining a segmentation mask for a single PDF page."""

    @abstractmethod
    def get_mask(self, page_base64: str) -> bytes:
        """Send a base64-encoded page image and receive the mask bytes."""
        ...
