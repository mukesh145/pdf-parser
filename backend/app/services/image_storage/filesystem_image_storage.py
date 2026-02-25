"""Concrete IImageStorage implementation backed by the local filesystem."""

import zipfile
from pathlib import Path
from typing import List

from PIL import Image

from app.interfaces.image_storage import IImageStorage


class FileSystemImageStorage(IImageStorage):
    """Save extracted images to disk and produce zip archives."""

    def __init__(self, base_images_path: str, base_zip_path: str) -> None:
        """Initialise with root directories for images and zip output."""

        self._base_images_path = Path(base_images_path)
        self._base_zip_path = Path(base_zip_path)

    def save_images(self, job_id: str, page_idx: int, images: List[Image.Image]) -> None:
        """Write PIL images to disk under {base_images_path}/{job_id}/page_{page_idx}/."""
        output_dir = self._base_images_path / job_id / f"page_{page_idx}"
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, img in enumerate(images):
            file_path = output_dir / f"img_{idx}.png"
            img.save(file_path, format="PNG")

    def zip_directory(self, job_id: str, output_path: str) -> None:
        """Compress the job's image directory into a zip file at output_path.

        Writes to a temporary file first, then atomically renames to the final
        path so that concurrent readers never see a partially-written zip.
        """
        source_dir = self._base_images_path / job_id
        zip_path = self._base_zip_path / output_path
        zip_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = zip_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(source_dir.rglob("*")):
                if file.is_file():
                    zf.write(file, arcname=file.relative_to(source_dir))

        tmp_path.rename(zip_path)

    def exists(self, path: str) -> bool:
        """Check whether a file exists at the given path on disk."""
        return Path(path).is_file()

    def get_file(self, path: str) -> bytes:
        """Read and return raw bytes of a file from disk."""
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(f"File not found: {target}")
        return target.read_bytes()
