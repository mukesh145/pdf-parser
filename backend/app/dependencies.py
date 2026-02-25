"""FastAPI dependency injection wiring.

Constructs concrete implementations and provides them via FastAPI's Depends().
This is the single place where abstractions are bound to concrete classes.
"""

from app.config import settings
from app.interfaces.pdf_storage import IPdfStorage
from app.interfaces.image_storage import IImageStorage
from app.interfaces.request_queue import IRequestQueue
from app.interfaces.mask_service import IMaskService
from app.interfaces.pdf_parser import IPdfParser
from app.interfaces.image_extractor import IImageExtractor
from app.services.pdf_storage.filesystem_pdf_storage import FileSystemPdfStorage
from app.services.image_storage.filesystem_image_storage import FileSystemImageStorage
from app.services.request_queue.redis_request_queue import RedisRequestQueue
from app.services.mask_service.unet_http_client import UnetHttpClient
from app.services.pdf_parser.pymupdf_parser import PyMuPdfParser
from app.services.image_extractor.mask_based_image_extractor import MaskBasedImageExtractor
from app.orchestrators.extract_images_handler import ExtractImagesHandler


def get_pdf_storage() -> IPdfStorage:
    """Provide a concrete IPdfStorage implementation."""
    return FileSystemPdfStorage(base_path=settings.base_pdf_path)


def get_image_storage() -> IImageStorage:
    """Provide a concrete IImageStorage implementation."""
    return FileSystemImageStorage(
        base_images_path=settings.base_images_path,
        base_zip_path=settings.base_zip_path,
    )


def get_request_queue() -> IRequestQueue:
    """Provide a concrete IRequestQueue implementation."""
    return RedisRequestQueue(
        redis_url=settings.redis_url,
        queue_name=settings.redis_queue_name,
    )


def get_mask_service() -> IMaskService:
    """Provide a concrete IMaskService implementation."""
    return UnetHttpClient(
        base_url=settings.unet_base_url,
        timeout_sec=settings.unet_timeout_sec,
        retries=settings.unet_request_retries,
        retry_backoff_sec=settings.unet_retry_backoff_sec,
    )


def get_pdf_parser() -> IPdfParser:
    """Provide a concrete IPdfParser implementation."""
    return PyMuPdfParser()


def get_image_extractor() -> IImageExtractor:
    """Provide a concrete IImageExtractor implementation."""
    return MaskBasedImageExtractor()


def get_extract_images_handler() -> ExtractImagesHandler:
    """Build and return the ExtractImagesHandler with all dependencies wired."""
    return ExtractImagesHandler(
        pdf_storage=get_pdf_storage(),
        queue=get_request_queue(),
    )
