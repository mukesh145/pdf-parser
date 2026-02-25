"""Loss registry — register new loss functions with a single decorator."""

from __future__ import annotations

from typing import Dict, Type

from train_pipeline.losses.base import SegmentationLoss

LOSS_REGISTRY: Dict[str, Type[SegmentationLoss]] = {}


def register_loss(name: str):
    """Class decorator that adds a loss to the global registry.

    Usage::

        @register_loss("my_loss")
        class MyLoss(SegmentationLoss):
            ...
    """

    def wrapper(cls: Type[SegmentationLoss]) -> Type[SegmentationLoss]:
        if name in LOSS_REGISTRY:
            raise ValueError(f"Loss '{name}' is already registered")
        LOSS_REGISTRY[name] = cls
        return cls

    return wrapper


def get_loss(name: str, **kwargs) -> SegmentationLoss:
    """Instantiate a registered loss by name."""
    if name not in LOSS_REGISTRY:
        available = ", ".join(LOSS_REGISTRY) or "(none)"
        raise KeyError(f"Unknown loss '{name}'. Available: {available}")
    return LOSS_REGISTRY[name](**kwargs)
