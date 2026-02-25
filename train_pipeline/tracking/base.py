"""Abstract interface for experiment trackers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch.nn as nn


class BaseTracker(ABC):
    """Contract every experiment tracker must satisfy.

    Swap MLflow for W&B, Neptune, or a no-op tracker by implementing this.
    """

    @abstractmethod
    def start_run(self, tags: Optional[Dict[str, str]] = None) -> None: ...

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None: ...

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int) -> None: ...

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str) -> None: ...

    @abstractmethod
    def log_model(self, model: nn.Module, registered_model_name: str) -> None: ...

    @abstractmethod
    def end_run(self) -> None: ...

    @abstractmethod
    def get_run_id(self) -> str: ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False
