"""Abstract interface for mask-based image extraction from a PDF page."""

from abc import ABC, abstractmethod
from typing import List

from PIL import Image


class IImageExtractor(ABC):
    """Contract for extracting images from a PDF page using a segmentation mask."""

    @abstractmethod
    def extract_images(self, page_bytes: bytes, mask: bytes) -> List[Image.Image]:
        """Use the mask to locate and extract images from the page, returning PIL images."""
        ...
