"""Training hyper-parameter configuration."""

from pydantic import BaseModel


class OptimizerConfig(BaseModel):
    """Optimizer settings."""

    name: str = "adam"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5


class SchedulerConfig(BaseModel):
    """LR scheduler settings."""

    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 5


class LossConfig(BaseModel):
    """Loss function settings."""

    loss_name: str = "dice_bce"
    dice_weight: float = 0.5
    bce_weight: float = 0.5


class TrainingConfig(BaseModel):
    """Top-level training settings."""

    num_epochs: int = 10
    save_every: int = 5
    checkpoint_dir: str = "/app/checkpoints"
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    loss: LossConfig = LossConfig()
