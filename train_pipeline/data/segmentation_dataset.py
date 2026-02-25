"""Sliding-window segmentation dataset for image / mask pairs."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from train_pipeline.data.base import BaseSegmentationDataset

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


class SegmentationDataset(BaseSegmentationDataset):
    """Yields fixed-size patches extracted via a sliding window over full images."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: Optional[str] = None,
        window_size: Tuple[int, int] = (256, 256),
        stride: Optional[Tuple[int, int]] = None,
        image_transform=None,
        mask_transform=None,
        padding: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir) if mask_dir else None
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.padding = padding

        self.window_size = (
            (window_size, window_size) if isinstance(window_size, int) else window_size
        )
        self.stride = (
            self.window_size
            if stride is None
            else ((stride, stride) if isinstance(stride, int) else stride)
        )

        self.image_files = sorted(
            f
            for ext in IMAGE_EXTENSIONS
            for f in list(self.image_dir.glob(f"*{ext}"))
            + list(self.image_dir.glob(f"*{ext.upper()}"))
        )

        if not self.image_files:
            raise ValueError(f"No images found in {image_dir}")

        self.patches: List[dict] = []
        for img_path in self.image_files:
            with Image.open(img_path) as img:
                h, w = img.size[1], img.size[0]
            for coords in self._sliding_window(h, w):
                self.patches.append({"path": img_path, "coords": coords})

    def _sliding_window(self, h: int, w: int) -> List[Tuple[int, int, int, int]]:
        wh, ww = self.window_size
        sh, sw = self.stride

        if h < wh or w < ww:
            return [(0, 0, h, w)] if self.padding else []

        patches = []
        for top in range(0, h - wh + 1, sh):
            if top + wh > h:
                top = h - wh
            for left in range(0, w - ww + 1, sw):
                if left + ww > w:
                    left = w - ww
                patches.append((top, left, top + wh, left + ww))
        return patches

    def _load_mask(self, img_path: Path, h: int, w: int) -> np.ndarray:
        if self.mask_dir is None:
            return np.zeros((h, w), dtype=np.uint8)

        stem = img_path.stem
        for name in [
            f"{stem}_mask.png",
            f"{stem}.png",
            f"{stem}.jpg",
            f"{stem}.jpeg",
        ]:
            mask_path = self.mask_dir / name
            if mask_path.exists():
                mask = np.array(Image.open(mask_path).convert("L"))
                if mask.shape != (h, w):
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                return mask
        return np.zeros((h, w), dtype=np.uint8)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx]
        img = Image.open(patch["path"]).convert("RGB")
        h, w = img.size[1], img.size[0]

        mask = self._load_mask(patch["path"], h, w)
        top, left, bottom, right = patch["coords"]

        img_patch = np.array(img)[top:bottom, left:right]
        mask_patch = mask[top:bottom, left:right]

        ph, pw = img_patch.shape[:2]
        if ph < self.window_size[0] or pw < self.window_size[1]:
            pad_h = max(0, self.window_size[0] - ph)
            pad_w = max(0, self.window_size[1] - pw)
            img_patch = np.pad(img_patch, ((0, pad_h), (0, pad_w), (0, 0)))
            mask_patch = np.pad(mask_patch, ((0, pad_h), (0, pad_w)))

        img_patch = Image.fromarray(img_patch)
        mask_patch = Image.fromarray(mask_patch, mode="L")

        seed = np.random.randint(2147483647)
        if self.image_transform:
            torch.manual_seed(seed)
            img_patch = self.image_transform(img_patch)
        if self.mask_transform:
            torch.manual_seed(seed)
            mask_patch = self.mask_transform(mask_patch)

        if not isinstance(img_patch, torch.Tensor):
            img_patch = TF.to_tensor(img_patch)
        if not isinstance(mask_patch, torch.Tensor):
            mask_patch = torch.from_numpy(np.array(mask_patch)).long()
            if len(mask_patch.shape) == 2:
                mask_patch = mask_patch.unsqueeze(0)

        return img_patch, mask_patch
