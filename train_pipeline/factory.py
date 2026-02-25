"""Factory — reads config and builds fully-constructed objects.

This is the only place that knows about concrete class names.
Everything else depends on abstractions.
"""

import logging

import torch
from torch.utils.data import DataLoader

from train_pipeline.configs.settings import Settings
from train_pipeline.data.segmentation_dataset import SegmentationDataset
from train_pipeline.data.transforms import train_image_transform, train_mask_transform
from train_pipeline.losses.registry import get_loss
from train_pipeline.models.registry import get_model
from train_pipeline.tracking.mlflow_tracker import MLflowTracker

log = logging.getLogger(__name__)


def build_dataset(settings: Settings) -> SegmentationDataset:
    cfg = settings.data
    return SegmentationDataset(
        image_dir=cfg.train_image_dir,
        mask_dir=cfg.train_mask_dir,
        window_size=(cfg.window_size, cfg.window_size),
        stride=(cfg.stride, cfg.stride),
        image_transform=train_image_transform,
        mask_transform=train_mask_transform,
        padding=True,
    )


def build_loader(settings: Settings, dataset: SegmentationDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=settings.data.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )


def build_model(settings: Settings):
    cfg = settings.model
    return get_model(
        cfg.model_name,
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        base_channels=cfg.base_channels,
    )


def build_loss(settings: Settings):
    lcfg = settings.training.loss
    return get_loss(
        lcfg.loss_name,
        dice_weight=lcfg.dice_weight,
        bce_weight=lcfg.bce_weight,
    )


def build_tracker(settings: Settings) -> MLflowTracker:
    return MLflowTracker(
        tracking_uri=settings.mlflow_tracking_uri,
        experiment_name=settings.mlflow_experiment_name,
    )
