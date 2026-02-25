"""Mock IMaskService that returns a random segmentation mask for testing."""

import io

import numpy as np
from PIL import Image, ImageDraw

from app.interfaces.mask_service import IMaskService

MASK_HEIGHT = 256
MASK_WIDTH = 256


class MockMaskService(IMaskService):
    """Return a randomly generated binary mask as PNG bytes."""

    def get_mask(self, page_base64: str) -> bytes:
        mask = Image.new("L", (MASK_WIDTH, MASK_HEIGHT), 0)
        draw = ImageDraw.Draw(mask)

        cx = np.random.randint(MASK_WIDTH // 4, 3 * MASK_WIDTH // 4)
        cy = np.random.randint(MASK_HEIGHT // 4, 3 * MASK_HEIGHT // 4)
        rx = np.random.randint(MASK_WIDTH // 6, MASK_WIDTH // 3)
        ry = np.random.randint(MASK_HEIGHT // 6, MASK_HEIGHT // 3)
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=255)

        buf = io.BytesIO()
        mask.save(buf, format="PNG")
        return buf.getvalue()
