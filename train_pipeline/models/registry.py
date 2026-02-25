"""Model registry — register new architectures with a single decorator."""

from __future__ import annotations

from typing import Dict, Type

from train_pipeline.models.base import SegmentationModel

MODEL_REGISTRY: Dict[str, Type[SegmentationModel]] = {}


def register_model(name: str):
    """Class decorator that adds a model to the global registry.

    Usage::

        @register_model("my_model")
        class MyModel(SegmentationModel):
            ...
    """

    def wrapper(cls: Type[SegmentationModel]) -> Type[SegmentationModel]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model '{name}' is already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return wrapper


def get_model(name: str, **kwargs) -> SegmentationModel:
    """Instantiate a registered model by name."""
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY) or "(none)"
        raise KeyError(f"Unknown model '{name}'. Available: {available}")
    return MODEL_REGISTRY[name](**kwargs)
