"""Worker that consumes extraction jobs from the queue and processes them."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.interfaces.request_queue import IRequestQueue
from app.interfaces.pdf_storage import IPdfStorage
from app.interfaces.image_storage import IImageStorage
from app.interfaces.mask_service import IMaskService
from app.interfaces.pdf_parser import IPdfParser
from app.interfaces.image_extractor import IImageExtractor
from app.models.extract_job import ExtractJob

logger = logging.getLogger(__name__)


class ImageExtractorWorker:
    """Poll the request queue, process PDF extraction jobs, and write results to storage.

    For each job: fetch PDF → split into pages → get U-Net mask per page →
    extract images → save images → zip output directory.
    """

    def __init__(
        self,
        queue: IRequestQueue,
        pdf_storage: IPdfStorage,
        image_storage: IImageStorage,
        mask_service: IMaskService,
        pdf_parser: IPdfParser,
        image_extractor: IImageExtractor,
        poll_interval_sec: int = 3,
        max_page_workers: int = 4,
    ) -> None:
        """Initialise with all injected service abstractions."""

        self._queue = queue
        self._pdf_storage = pdf_storage
        self._image_storage = image_storage
        self._mask_service = mask_service
        self._pdf_parser = pdf_parser
        self._image_extractor = image_extractor
        self._poll_interval_sec = poll_interval_sec
        self._page_executor = ThreadPoolExecutor(max_workers=max_page_workers)

    def run(self) -> None:
        """Start the infinite polling loop, processing jobs as they arrive."""

        logger.info("Worker started, polling for jobs…")
        while True:
            job = self._queue.poll()
            if job is None:
                time.sleep(self._poll_interval_sec)
                continue

            logger.info("Claimed job %s", job.job_id)
            try:
                self._process_job(job)
                self._queue.ack(job.job_id)
                logger.info("Job %s completed successfully", job.job_id)
            except Exception:
                logger.exception("Job %s failed, nacking", job.job_id)
                self._queue.nack(job.job_id)

    def _process_job(self, job: ExtractJob) -> None:
        """Fetch the PDF, process all pages, and create the output zip."""

        pdf_bytes = self._pdf_storage.get(job.pdf_path)
        pages = self._pdf_parser.split_into_pages(pdf_bytes)

        futures = {
            self._page_executor.submit(
                self._process_page, job.job_id, idx, page_bytes
            ): idx
            for idx, page_bytes in enumerate(pages)
        }
        for future in as_completed(futures):
            future.result()

        self._image_storage.zip_directory(job.job_id, job.zip_output_path)

    def _process_page(self, job_id: str, page_idx: int, page_bytes: bytes) -> None:
        """Convert a single page to base64, get its mask, extract images, and save them."""

        page_base64 = self._pdf_parser.page_to_base64(page_bytes)
        mask = self._mask_service.get_mask(page_base64)
        images = self._image_extractor.extract_images(page_bytes, mask)
        self._image_storage.save_images(job_id, page_idx, images)
