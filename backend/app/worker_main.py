"""Standalone entry point for the image-extractor worker process."""

import logging

from app.config import settings
from app.dependencies import (
    get_image_extractor,
    get_image_storage,
    get_mask_service,
    get_pdf_parser,
    get_pdf_storage,
    get_request_queue,
)
from app.orchestrators.image_extractor_worker import ImageExtractorWorker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def build_worker() -> ImageExtractorWorker:
    return ImageExtractorWorker(
        queue=get_request_queue(),
        pdf_storage=get_pdf_storage(),
        image_storage=get_image_storage(),
        mask_service=get_mask_service(),
        pdf_parser=get_pdf_parser(),
        image_extractor=get_image_extractor(),
        poll_interval_sec=settings.poll_interval_sec,
        max_page_workers=settings.max_page_workers,
    )


if __name__ == "__main__":
    worker = build_worker()
    worker.run()
