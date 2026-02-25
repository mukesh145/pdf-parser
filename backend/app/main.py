"""FastAPI application factory and entry point."""

from fastapi import FastAPI

from app.api.routes import extract


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""

    app = FastAPI(
        title="PDF Image Extractor",
        description="Extract images from PDF pages using U-Net segmentation masks.",
        version="0.1.0",
    )

    app.include_router(extract.router, prefix="/extract", tags=["extract"])

    return app


app = create_app()
