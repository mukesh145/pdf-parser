"""Checkpoint save / load utilities."""

import logging
import os

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class CheckpointManager:
    """Handles saving and tracking the best model checkpoint."""

    def __init__(self, save_dir: str) -> None:
        self._save_dir = save_dir
        self.best_loss: float = float("inf")
        self.best_checkpoint: dict | None = None

    def update(
        self,
        epoch: int,
        avg_loss: float,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
    ) -> bool:
        """Track whether this epoch is the best so far. Returns True if improved."""
        if avg_loss >= self.best_loss:
            return False

        self.best_loss = avg_loss
        self.best_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss,
        }
        return True

    def save_periodic(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: object,
        avg_loss: float,
    ) -> str:
        """Save a periodic checkpoint and return its path."""
        path = os.path.join(self._save_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": avg_loss,
            },
            path,
        )
        log.info("Saved periodic checkpoint: %s", path)
        return path

    def save_best(self) -> str | None:
        """Save the best checkpoint and return its path, or None if no improvement."""
        if self.best_checkpoint is None:
            return None
        path = os.path.join(self._save_dir, "best_model.pth")
        torch.save(self.best_checkpoint, path)
        log.info("Saved best checkpoint (loss=%.4f): %s", self.best_loss, path)
        return path
