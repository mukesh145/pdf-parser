"""Combined Dice + BCE loss for binary segmentation."""

import torch
import torch.nn as nn

from train_pipeline.losses.base import SegmentationLoss
from train_pipeline.losses.registry import register_loss


@register_loss("dice_bce")
class DiceBCELoss(SegmentationLoss):
    """Weighted sum of Dice loss and Binary Cross-Entropy with logits."""

    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).view(-1)
        targets_flat = targets.view(-1)
        intersection = (probs * targets_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            probs.sum() + targets_flat.sum() + self.smooth
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self._dice_loss(logits, targets)
            + self.bce_weight * self.bce(logits, targets)
        )
