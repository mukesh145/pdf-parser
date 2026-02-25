"""Data-related configuration."""

from pydantic import BaseModel


class DataConfig(BaseModel):
    """Settings consumed by datasets and data loaders."""

    train_image_dir: str = "/data/train/images"
    train_mask_dir: str = "/data/train/masks"
    window_size: int = 256
    stride: int = 128
    batch_size: int = 8
