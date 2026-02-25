"""MLflow implementation of the experiment tracker."""

import logging
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import torch.nn as nn

from train_pipeline.tracking.base import BaseTracker

log = logging.getLogger(__name__)


class MLflowTracker(BaseTracker):
    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._run = None

    def start_run(self, tags: Optional[Dict[str, str]] = None) -> None:
        mlflow.set_tracking_uri(self._tracking_uri)
        self._ensure_experiment()
        self._run = mlflow.start_run(tags=tags)

    def _ensure_experiment(self) -> None:
        """Ensure an active experiment exists before starting a run.

        If the configured experiment does not exist, create it.
        If it exists but is deleted, create and switch to a new fallback name.
        """
        exp = mlflow.get_experiment_by_name(self._experiment_name)
        if exp is None:
            mlflow.create_experiment(self._experiment_name)
            mlflow.set_experiment(self._experiment_name)
            return

        if exp.lifecycle_stage == "deleted":
            suffix = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
            fallback_name = f"{self._experiment_name}-recreated-{suffix}"
            mlflow.create_experiment(fallback_name)
            mlflow.set_experiment(fallback_name)
            log.warning(
                "Experiment '%s' is deleted; switched to '%s'",
                self._experiment_name,
                fallback_name,
            )
            return

        mlflow.set_experiment(self._experiment_name)

    def log_params(self, params: Dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(self, model: nn.Module, registered_model_name: str) -> None:
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

    def end_run(self) -> None:
        if self._run is not None:
            mlflow.end_run()
            self._run = None

    def get_run_id(self) -> str:
        run = mlflow.active_run()
        if run is None:
            raise RuntimeError("No active MLflow run")
        return run.info.run_id

    def promote_model(
        self,
        model_name: str,
        run_id: str,
        metric_name: str = "best_loss",
        lower_is_better: bool = True,
        threshold: float = 0.0,
    ) -> bool:
        """Compare the latest run with current Production model and promote if better."""
        client = mlflow.tracking.MlflowClient()

        try:
            all_versions = client.get_latest_versions(model_name, stages=[])
            if not all_versions:
                log.warning("No versions found for model %s", model_name)
                return False

            latest_version = next(
                (v for v in all_versions if v.run_id == run_id), None
            )
            if latest_version is None:
                log.warning("No model version found for run_id %s", run_id)
                return False

            latest_run = client.get_run(run_id)
            latest_metric = latest_run.data.metrics.get(metric_name)
            if latest_metric is None:
                log.warning("Metric '%s' not found in run %s", metric_name, run_id)
                return False

            log.info(
                "Latest %s: %.4f (version %s)",
                metric_name, latest_metric, latest_version.version,
            )

            production_versions = client.get_latest_versions(
                model_name, stages=["Production"]
            )

            if not production_versions:
                log.info(
                    "No Production model yet — promoting version %s",
                    latest_version.version,
                )
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Production",
                )
                return True

            prod_version = production_versions[0]
            prod_run = client.get_run(prod_version.run_id)
            prod_metric = prod_run.data.metrics.get(metric_name)
            if prod_metric is None:
                log.warning("Metric '%s' not found in Production run", metric_name)
                return False

            if lower_is_better:
                improvement = prod_metric - latest_metric
                is_better = latest_metric < prod_metric and improvement >= threshold
            else:
                improvement = latest_metric - prod_metric
                is_better = latest_metric > prod_metric and improvement >= threshold

            log.info(
                "Production %s: %.4f | Candidate: %.4f | Improvement: %.4f",
                metric_name, prod_metric, latest_metric, improvement,
            )

            if is_better:
                client.transition_model_version_stage(
                    name=model_name,
                    version=prod_version.version,
                    stage="Archived",
                )
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Production",
                )
                log.info(
                    "Promoted version %s to Production (archived %s)",
                    latest_version.version, prod_version.version,
                )
                return True

            log.info("Keeping Production version %s", prod_version.version)
            return False

        except Exception:
            log.exception("Error during model promotion")
            return False
