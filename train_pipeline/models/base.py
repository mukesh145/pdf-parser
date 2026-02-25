"""Abstract interface for segmentation models."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SegmentationModel(nn.Module, ABC):
    """Contract every segmentation model must satisfy.

    Subclasses accept (in_channels, num_classes, **kwargs) and implement forward.
    """

    @abstractmethod
    def __init__(self, in_channels: int, num_classes: int, **kwargs) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
