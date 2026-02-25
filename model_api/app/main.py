"""Model-serving API — loads the production UNet++ from MLflow and serves predictions."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.model_manager import manager
from app.routes.predict import router as predict_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    loaded = manager.load_production_model()
    if not loaded:
        log.warning(
            "Starting with local fallback ONNX model; will hot-swap to MLflow "
            "Production ONNX model when available"
        )
    manager.start_polling()
    yield
    manager.stop_polling()


def create_app() -> FastAPI:
    app = FastAPI(title="UNet++ Segmentation API", lifespan=lifespan)
    app.include_router(predict_router)
    return app


app = create_app()
