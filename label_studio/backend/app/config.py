"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the label studio backend."""

    upload_dir: str = "/tmp/label_uploads"
    train_image_dir: str = "/data/train/images"
    train_mask_dir: str = "/data/train/masks"

    redis_url: str = "redis://redis:6379/0"
    train_queue_name: str = "train_jobs"

    poll_interval_sec: int = 3

    class Config:
        env_file = ".env"


settings = Settings()
