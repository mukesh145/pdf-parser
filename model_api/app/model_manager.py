"""Thread-safe model manager with background MLflow polling."""

import logging
import threading

import mlflow.pytorch
import torch
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException

from app.config import settings
from train_pipeline.models import get_model

log = logging.getLogger(__name__)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ModelManager:
    """Holds the current production model and watches MLflow for updates.

    * ``load_production_model()`` — eagerly loads the current Production model.
    * ``start_polling()``         — spins up a daemon thread that checks for
                                    new Production versions every *poll_interval* seconds.
    * ``stop_polling()``          — signals the thread to stop.
    * ``model``                   — property returning the loaded model (or *None*).
    """

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._current_version: str | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    @property
    def model(self) -> torch.nn.Module | None:
        with self._lock:
            return self._model

    @property
    def current_version(self) -> str | None:
        with self._lock:
            return self._current_version

    def _load_model(self, version: str) -> torch.nn.Module:
        model_uri = f"models:/{settings.model_name}/{version}"
        log.info("Loading model from %s …", model_uri)
        model = mlflow.pytorch.load_model(model_uri, map_location=_device)
        model.to(_device)
        model.eval()
        return model

    def _get_production_version(self) -> str | None:
        """Return the version string of the current Production model, or None."""
        try:
            versions = self._client.get_latest_versions(
                settings.model_name, stages=[settings.model_stage],
            )
        except RestException:
            log.warning(
                "Registered model '%s' not found in MLflow yet",
                settings.model_name,
            )
            return None
        if versions:
            return versions[0].version
        return None

    def _load_fallback_model(self) -> torch.nn.Module:
        """Build an untrained model so inference API can still stay online."""
        model = get_model(
            settings.fallback_model_name,
            in_channels=settings.in_channels,
            num_classes=settings.num_classes,
            base_channels=settings.base_channels,
        )
        model.to(_device)
        model.eval()
        return model

    def load_production_model(self) -> bool:
        """Attempt to load the Production model.  Returns True on success."""
        version = self._get_production_version()
        if version is None:
            log.warning(
                "No %s model in stage '%s'; loading fallback '%s' with random weights",
                settings.model_name,
                settings.model_stage,
                settings.fallback_model_name,
            )
            fallback = self._load_fallback_model()
            with self._lock:
                self._model = fallback
                self._current_version = None
            return False
        model = self._load_model(version)
        with self._lock:
            self._model = model
            self._current_version = version
        log.info("Model %s version %s loaded and ready", settings.model_name, version)
        return True

    # ── background polling ──────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=settings.model_poll_interval_sec)
            if self._stop_event.is_set():
                break
            try:
                new_version = self._get_production_version()
                if new_version is None:
                    continue
                with self._lock:
                    current = self._current_version
                if new_version != current:
                    log.info(
                        "New production version detected: %s (was %s). Reloading …",
                        new_version, current,
                    )
                    model = self._load_model(new_version)
                    with self._lock:
                        self._model = model
                        self._current_version = new_version
                    log.info("Hot-swapped to model version %s", new_version)
            except Exception:
                log.exception("Error while polling for model updates")

    def start_polling(self) -> None:
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="model-poller",
        )
        self._poll_thread.start()
        log.info("Model poller started (interval=%ds)", settings.model_poll_interval_sec)

    def stop_polling(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5)
        log.info("Model poller stopped")


manager = ModelManager()
