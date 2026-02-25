"""Concrete IMaskService implementation that calls the U-Net API over HTTP."""

import httpx

from app.interfaces.mask_service import IMaskService


class UnetHttpClient(IMaskService):
    """Obtain segmentation masks by calling the remote U-Net HTTP service."""

    def __init__(self, base_url: str) -> None:
        """Initialise with the base URL of the U-Net API."""

        self._base_url = base_url
        self._client = httpx.Client(timeout=60.0)

    def get_mask(self, page_base64: str) -> bytes:
        """POST the base64-encoded page to the U-Net API and return the mask bytes."""

        response = self._client.post(
            f"{self._base_url}/predict",
            json={"image_base64": page_base64},
        )
        response.raise_for_status()
        return response.content
