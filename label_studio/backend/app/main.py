"""FastAPI application factory and entry point."""

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import labels
from app.services.storage import ensure_dirs

FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "/app/frontend"))


def create_app() -> FastAPI:
    application = FastAPI(
        title="Label Studio",
        description="Upload images, draw segmentation masks, and queue training jobs.",
        version="0.1.0",
    )

    ensure_dirs()

    application.include_router(labels.router, prefix="/api", tags=["labels"])

    application.mount(
        "/static",
        StaticFiles(directory=str(FRONTEND_DIR / "static")),
        name="static",
    )

    return application


app = create_app()
