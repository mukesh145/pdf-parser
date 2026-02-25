"""Generic training loop — depends only on abstractions."""

import logging
import tempfile
from typing import Dict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from train_pipeline.configs.training_config import TrainingConfig
from train_pipeline.tracking.base import BaseTracker
from train_pipeline.training.checkpoint import CheckpointManager

log = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Trainer:
    """Orchestrates the train loop.

    Receives fully-constructed model, loss, loader, and tracker —
    it never imports or builds concrete implementations itself.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        loader: DataLoader,
        tracker: BaseTracker,
        config: TrainingConfig,
        registered_model_name: str,
    ) -> None:
        self._model = model.to(device)
        self._criterion = criterion.to(device)
        self._loader = loader
        self._tracker = tracker
        self._config = config
        self._registered_model_name = registered_model_name

        self._optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=config.optimizer.learning_rate,
            weight_decay=config.optimizer.weight_decay,
        )
        self._scheduler = ReduceLROnPlateau(
            self._optimizer,
            mode="min",
            factor=config.scheduler.lr_decay_factor,
            patience=config.scheduler.lr_decay_patience,
        )

    def run(self, run_params: Dict[str, object]) -> None:
        """Execute the full training loop with tracking.

        Assumes the caller has already called ``tracker.start_run()``
        and will call ``tracker.end_run()`` when done (or use it as a
        context manager around *this* call).
        """
        self._tracker.log_params(run_params)
        self._train_epochs()
        self._tracker.log_model(self._model, self._registered_model_name)

    def _train_epochs(self) -> None:
        cfg = self._config

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = CheckpointManager(tmp)

            for epoch in range(1, cfg.num_epochs + 1):
                avg_loss = self._train_one_epoch()
                self._scheduler.step(avg_loss)

                self._tracker.log_metrics(
                    {
                        "train_loss": avg_loss,
                        "learning_rate": self._optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )
                log.info(
                    "Epoch %d/%d — loss: %.4f  lr: %.6f",
                    epoch, cfg.num_epochs, avg_loss,
                    self._optimizer.param_groups[0]["lr"],
                )

                ckpt.update(epoch, avg_loss, self._model, self._optimizer, self._scheduler)

                if epoch % cfg.save_every == 0:
                    path = ckpt.save_periodic(
                        epoch, self._model, self._optimizer, self._scheduler, avg_loss,
                    )
                    self._tracker.log_artifact(path, artifact_path="checkpoints")

            self._tracker.log_metrics({"best_loss": ckpt.best_loss}, step=cfg.num_epochs)

            best_path = ckpt.save_best()
            if best_path:
                self._tracker.log_artifact(best_path, artifact_path="checkpoints")

    def _train_one_epoch(self) -> float:
        self._model.train()
        epoch_loss = 0.0
        num_batches = 0

        for images, masks in self._loader:
            images = images.to(device)
            masks = masks.to(device).float()

            self._optimizer.zero_grad()
            outputs = self._model(images)
            loss = self._criterion(outputs, masks)
            loss.backward()
            self._optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        return epoch_loss / max(num_batches, 1)
