"""Master settings that composes all sub-configs from environment variables."""

from pydantic_settings import BaseSettings

from train_pipeline.configs.data_config import DataConfig
from train_pipeline.configs.model_config import ModelConfig
from train_pipeline.configs.training_config import (
    LossConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)


class Settings(BaseSettings):
    """Flat env-var loading that builds segregated config objects."""

    # --- Infrastructure ---
    redis_url: str = "redis://redis:6379/0"
    train_queue_name: str = "train_jobs"
    poll_interval_sec: int = 5

    # --- MLflow / tracking ---
    mlflow_tracking_uri: str = "http://mlflow:5000"
    mlflow_experiment_name: str = "unet-segmentation"
    registered_model_name: str = "UNetPlusPlus"

    # --- Data ---
    train_image_dir: str = "/data/train/images"
    train_mask_dir: str = "/data/train/masks"
    window_size: int = 256
    stride: int = 128
    batch_size: int = 8

    # --- Model ---
    model_name: str = "unet_plusplus"
    in_channels: int = 3
    num_classes: int = 1
    base_channels: int = 64

    # --- Training ---
    num_epochs: int = 10
    save_every: int = 5
    checkpoint_dir: str = "/app/checkpoints"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 5

    # --- Loss ---
    loss_name: str = "dice_bce"
    dice_weight: float = 0.5
    bce_weight: float = 0.5

    class Config:
        env_file = ".env"

    @property
    def data(self) -> DataConfig:
        return DataConfig(
            train_image_dir=self.train_image_dir,
            train_mask_dir=self.train_mask_dir,
            window_size=self.window_size,
            stride=self.stride,
            batch_size=self.batch_size,
        )

    @property
    def model(self) -> ModelConfig:
        return ModelConfig(
            model_name=self.model_name,
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
        )

    @property
    def training(self) -> TrainingConfig:
        return TrainingConfig(
            num_epochs=self.num_epochs,
            save_every=self.save_every,
            checkpoint_dir=self.checkpoint_dir,
            optimizer=OptimizerConfig(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            ),
            scheduler=SchedulerConfig(
                lr_decay_factor=self.lr_decay_factor,
                lr_decay_patience=self.lr_decay_patience,
            ),
            loss=LossConfig(
                loss_name=self.loss_name,
                dice_weight=self.dice_weight,
                bce_weight=self.bce_weight,
            ),
        )


settings = Settings()
