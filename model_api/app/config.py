"""Model API configuration loaded from environment variables."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central configuration for the model-serving API."""

    mlflow_tracking_uri: str = "http://mlflow:5000"
    model_name: str = "UNetPlusPlus"
    model_stage: str = "Production"
    local_onnx_model_path: str = "/app/model/model.onnx"
    mlflow_onnx_artifact_path: str = "model/model.onnx"

    # How often (seconds) the background thread checks MLflow for a new model.
    model_poll_interval_sec: int = 30
    onnx_intra_op_num_threads: int = 1
    onnx_inter_op_num_threads: int = 1

    host: str = "0.0.0.0"
    port: int = 8001

    class Config:
        env_file = ".env"


settings = Settings()
