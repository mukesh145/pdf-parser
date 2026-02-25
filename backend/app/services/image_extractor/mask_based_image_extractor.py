"""Concrete IImageExtractor implementation using segmentation masks."""

import io
from typing import List, Tuple

import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from scipy import ndimage

from app.interfaces.image_extractor import IImageExtractor

_MASK_THRESHOLD = 127
_MIN_REGION_AREA = 100


class MaskBasedImageExtractor(IImageExtractor):
    """Extract images from a PDF page by applying a U-Net segmentation mask."""

    def extract_images(self, page_bytes: bytes, mask: bytes) -> List[Image.Image]:
        """Apply the mask to the page, identify image regions, and return cropped PIL images."""

        page_image = self._render_page(page_bytes)
        mask_image = Image.open(io.BytesIO(mask)).convert("L")

        if mask_image.size != page_image.size:
            mask_image = mask_image.resize(page_image.size, Image.NEAREST)

        mask_array = np.array(mask_image) > _MASK_THRESHOLD
        bboxes = self._find_region_bboxes(mask_array)

        return [page_image.crop(bbox) for bbox in bboxes]

    @staticmethod
    def _render_page(page_bytes: bytes) -> Image.Image:
        """Render a single-page PDF to a PIL Image via PyMuPDF."""

        doc = fitz.open(stream=page_bytes, filetype="pdf")
        pix = doc[0].get_pixmap()
        mode = "RGBA" if pix.alpha else "RGB"
        img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
        doc.close()
        return img

    @staticmethod
    def _find_region_bboxes(
        mask: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """Label connected components and return their bounding boxes in PIL format."""

        labeled, num_features = ndimage.label(mask)
        bboxes: List[Tuple[int, int, int, int]] = []

        for i in range(1, num_features + 1):
            ys, xs = np.where(labeled == i)
            x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            if (x1 - x0) * (y1 - y0) >= _MIN_REGION_AREA:
                bboxes.append((x0, y0, x1, y1))

        return bboxes
