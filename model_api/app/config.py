"""Model API configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the model-serving API."""

    mlflow_tracking_uri: str = "http://mlflow:5000"
    model_name: str = "UNetPlusPlus"
    model_stage: str = "Production"
    fallback_model_name: str = "unet_plusplus"
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 64

    # How often (seconds) the background thread checks MLflow for a new model.
    model_poll_interval_sec: int = 30

    host: str = "0.0.0.0"
    port: int = 8001

    class Config:
        env_file = ".env"


settings = Settings()
