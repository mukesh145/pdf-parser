"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the PDF Image Extractor backend.

    Values are read from environment variables or a .env file.
    """

    # Storage paths
    base_pdf_path: str = "/tmp/pdf_storage"
    base_images_path: str = "/tmp/image_storage"
    base_zip_path: str = "/tmp/zip_storage"

    # Redis
    redis_url: str = "redis://redis:6379/0"
    redis_queue_name: str = "extract_jobs"

    # U-Net API
    unet_base_url: str = "http://localhost:8001"

    # Worker poll interval
    poll_interval_sec: int = 3

    class Config:
        env_file = ".env"


settings = Settings()
