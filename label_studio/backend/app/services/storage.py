"""Filesystem helpers for upload staging and train-dir persistence."""

import io
import zipfile
from pathlib import Path
from typing import List

from PIL import Image
import numpy as np

from app.config import settings


def ensure_dirs() -> None:
    """Create required directories if they do not exist."""
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.train_image_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.train_mask_dir).mkdir(parents=True, exist_ok=True)


def save_upload(filename: str, data: bytes) -> Path:
    """Persist a single uploaded file to the staging area."""
    dest = Path(settings.upload_dir) / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def extract_zip(zip_bytes: bytes) -> List[str]:
    """Extract a zip archive into the upload dir; return saved filenames."""
    saved: List[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            if "__MACOSX" in member:
                continue
            name = Path(member).name
            if name.startswith("."):
                continue
            if not _is_image(name):
                continue
            data = zf.read(member)
            save_upload(name, data)
            saved.append(name)
    return saved


def clear_uploads() -> None:
    """Remove all files from the upload staging directory."""
    upload = Path(settings.upload_dir)
    if not upload.exists():
        return
    for f in upload.iterdir():
        if f.is_file():
            f.unlink()


def list_uploaded_images() -> List[str]:
    """Return filenames of all images currently in the upload dir."""
    upload = Path(settings.upload_dir)
    if not upload.exists():
        return []
    return sorted(
        f.name for f in upload.iterdir() if f.is_file() and _is_image(f.name)
    )


def copy_pair_to_train_dir(image_name: str, mask_bytes: bytes) -> None:
    """Copy an image from uploads and its binarised mask into the train volume.

    The raw mask from the UI is an RGBA PNG painted in red.  We convert it
    to a clean white-on-black grayscale image (alpha > 0 → 255, else 0).
    """
    src = Path(settings.upload_dir) / image_name
    if not src.exists():
        raise FileNotFoundError(f"Upload not found: {image_name}")

    dst_img = Path(settings.train_image_dir) / image_name
    dst_img.write_bytes(src.read_bytes())

    raw = Image.open(io.BytesIO(mask_bytes)).convert("RGBA")
    alpha = np.array(raw)[:, :, 3]
    binary = np.where(alpha > 0, 255, 0).astype(np.uint8)
    mask_img = Image.fromarray(binary, mode="L")

    mask_name = Path(image_name).stem + "_mask.png"
    dst_mask = Path(settings.train_mask_dir) / mask_name
    mask_img.save(dst_mask, format="PNG")


def _is_image(name: str) -> bool:
    return Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
