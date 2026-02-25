"""Abstract interface for loss functions."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class SegmentationLoss(nn.Module, ABC):
    """Contract every segmentation loss must satisfy."""

    @abstractmethod
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor: ...
