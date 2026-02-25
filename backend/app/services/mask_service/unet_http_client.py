"""Concrete IMaskService implementation that calls the U-Net API over HTTP."""

import logging
import time

import httpx

from app.interfaces.mask_service import IMaskService

log = logging.getLogger(__name__)


class UnetHttpClient(IMaskService):
    """Obtain segmentation masks by calling the remote U-Net HTTP service."""

    def __init__(
        self,
        base_url: str,
        timeout_sec: float = 60.0,
        retries: int = 3,
        retry_backoff_sec: float = 0.5,
    ) -> None:
        """Initialise with the base URL of the U-Net API."""

        self._base_url = base_url
        self._retries = max(1, retries)
        self._retry_backoff_sec = max(0.0, retry_backoff_sec)
        self._client = httpx.Client(timeout=timeout_sec)

    def get_mask(self, page_base64: str) -> bytes:
        """POST the base64-encoded page to the U-Net API and return the mask bytes."""
        last_exc: Exception | None = None

        for attempt in range(1, self._retries + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/predict",
                    json={"image_base64": page_base64},
                )
                response.raise_for_status()
                return response.content
            except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                if attempt >= self._retries:
                    break
                sleep_sec = self._retry_backoff_sec * attempt
                log.warning(
                    "U-Net request failed (attempt %d/%d): %s. Retrying in %.2fs",
                    attempt,
                    self._retries,
                    exc,
                    sleep_sec,
                )
                time.sleep(sleep_sec)

        assert last_exc is not None
        raise last_exc
