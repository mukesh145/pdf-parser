"""Model-related configuration."""

from pydantic import BaseModel


class ModelConfig(BaseModel):
    """Settings consumed by the model factory."""

    model_name: str = "unet_plusplus"
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 64
