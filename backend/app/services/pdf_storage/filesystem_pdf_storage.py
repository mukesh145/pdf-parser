"""Concrete IPdfStorage implementation backed by the local filesystem."""

from pathlib import Path

from app.interfaces.pdf_storage import IPdfStorage


class FileSystemPdfStorage(IPdfStorage):
    """Store and retrieve PDF files on the local filesystem."""

    def __init__(self, base_path: str) -> None:
        """Initialise with the root directory for PDF storage."""

        self._base_path = Path(base_path)

    def save(self, pdf_bytes: bytes, path: str) -> str:
        """Write PDF bytes to disk at base_path/path and return the full path."""
        full_path = self._base_path / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_bytes(pdf_bytes)
        return str(full_path)

    def get(self, path: str) -> bytes:
        """Read and return PDF bytes from disk."""
        full_path = self._base_path / path
        if not full_path.is_file():
            raise FileNotFoundError(f"PDF not found: {full_path}")
        return full_path.read_bytes()

    def delete(self, path: str) -> None:
        """Remove a PDF file from disk."""
        full_path = self._base_path / path
        if full_path.is_file():
            full_path.unlink()
