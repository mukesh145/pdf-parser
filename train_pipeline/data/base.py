"""Abstract interface for datasets."""

from abc import ABC, abstractmethod
from typing import Any, Tuple

from torch.utils.data import Dataset


class BaseSegmentationDataset(Dataset, ABC):
    """Contract every segmentation dataset must satisfy.

    Subclasses must return (image_tensor, mask_tensor) from __getitem__.
    """

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, Any]: ...
