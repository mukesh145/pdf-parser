"""Concrete IPdfParser implementation using PyMuPDF (fitz)."""

import base64
from typing import List

import fitz  # PyMuPDF

from app.interfaces.pdf_parser import IPdfParser


class PyMuPdfParser(IPdfParser):
    """Parse and split PDFs using the PyMuPDF library."""

    def split_into_pages(self, pdf_bytes: bytes) -> List[bytes]:
        """Open the PDF with PyMuPDF and return each page as separate bytes."""

        src = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages: List[bytes] = []
        for page_num in range(len(src)):
            dst = fitz.open()
            dst.insert_pdf(src, from_page=page_num, to_page=page_num)
            pages.append(dst.tobytes())
            dst.close()
        src.close()
        return pages

    def page_to_base64(self, page_bytes: bytes) -> str:
        """Render a single page to a PNG and return its base64-encoded string."""

        doc = fitz.open(stream=page_bytes, filetype="pdf")
        page = doc[0]
        pix = page.get_pixmap()
        png_bytes = pix.tobytes(output="png")
        doc.close()
        return base64.b64encode(png_bytes).decode("ascii")
