"""Thread-safe ONNX model manager with background MLflow polling."""

import logging
import threading
from pathlib import Path

import onnxruntime as ort
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient

from app.config import settings

log = logging.getLogger(__name__)


class ModelManager:
    """Holds the active ONNX session and hot-swaps on new MLflow versions."""

    def __init__(self) -> None:
        self._session: ort.InferenceSession | None = None
        self._input_name: str | None = None
        self._output_name: str | None = None
        self._current_version: str | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poll_thread: threading.Thread | None = None
        self._client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)
        self._providers = self._build_providers()

    @staticmethod
    def _build_providers() -> list[str]:
        available = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]

    @property
    def session(self) -> ort.InferenceSession | None:
        with self._lock:
            return self._session

    @property
    def input_name(self) -> str | None:
        with self._lock:
            return self._input_name

    @property
    def output_name(self) -> str | None:
        with self._lock:
            return self._output_name

    @property
    def current_version(self) -> str | None:
        with self._lock:
            return self._current_version

    def _load_session(self, model_path: Path) -> tuple[ort.InferenceSession, str, str]:
        # Conservative threading lowers memory pressure and reduces OOM risk in small containers.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = max(1, settings.onnx_intra_op_num_threads)
        session_options.inter_op_num_threads = max(1, settings.onnx_inter_op_num_threads)

        session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=self._providers,
        )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        return session, input_name, output_name

    def _load_local_fallback_session(self) -> tuple[ort.InferenceSession, str, str]:
        model_path = Path(settings.local_onnx_model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Fallback ONNX model not found at {model_path}. "
                "Generate/export one before starting model-api."
            )
        log.info("Loading local ONNX model from %s", model_path)
        return self._load_session(model_path)

    def _get_production_version(self) -> str | None:
        """Return the version string of the current Production model, or None."""
        try:
            versions = self._client.get_latest_versions(
                settings.model_name,
                stages=[settings.model_stage],
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

    def _load_session_from_mlflow_version(
        self, version: str,
    ) -> tuple[ort.InferenceSession, str, str]:
        log.info("Loading ONNX model for MLflow version %s", version)
        mv = self._client.get_model_version(settings.model_name, version)
        artifact = self._client.download_artifacts(
            run_id=mv.run_id,
            path=settings.mlflow_onnx_artifact_path,
        )
        model_path = Path(artifact)
        return self._load_session(model_path)

    def load_production_model(self) -> bool:
        """Attempt to load Production ONNX model; fallback to local ONNX file."""
        version = self._get_production_version()
        if version is not None:
            try:
                session, input_name, output_name = self._load_session_from_mlflow_version(version)
                with self._lock:
                    self._session = session
                    self._input_name = input_name
                    self._output_name = output_name
                    self._current_version = version
                log.info("Loaded Production ONNX model %s version %s", settings.model_name, version)
                return True
            except Exception:
                log.exception(
                    "Failed to load ONNX artifact '%s' for %s version %s",
                    settings.mlflow_onnx_artifact_path,
                    settings.model_name,
                    version,
                )

        session, input_name, output_name = self._load_local_fallback_session()
        with self._lock:
            self._session = session
            self._input_name = input_name
            self._output_name = output_name
            self._current_version = None
        return False

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
                        "New production version detected: %s (was %s). Reloading ONNX session...",
                        new_version,
                        current,
                    )
                    session, input_name, output_name = self._load_session_from_mlflow_version(new_version)
                    with self._lock:
                        self._session = session
                        self._input_name = input_name
                        self._output_name = output_name
                        self._current_version = new_version
                    log.info("Hot-swapped ONNX model to version %s", new_version)
            except Exception:
                log.exception("Error while polling for ONNX model updates")

    def start_polling(self) -> None:
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="model-poller",
        )
        self._poll_thread.start()
        log.info("Model poller started (interval=%ds)", settings.model_poll_interval_sec)

    def stop_polling(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=5)
        log.info("Model poller stopped")


manager = ModelManager()
