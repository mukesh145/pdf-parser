"""Export the train_pipeline model architecture to ONNX without training."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from train_pipeline.configs.settings import Settings
from train_pipeline.models import get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the same model as train_pipeline and export it to ONNX.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("model/model.onnx"),
        help="Output ONNX file path (default: model/model.onnx)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Input height for dummy tensor (defaults to WINDOW_SIZE)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Input width for dummy tensor (defaults to WINDOW_SIZE)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings()
    model_cfg = settings.model

    model = get_model(
        model_cfg.model_name,
        in_channels=model_cfg.in_channels,
        num_classes=model_cfg.num_classes,
        base_channels=model_cfg.base_channels,
    )
    model.eval()

    height = args.height or settings.window_size
    width = args.width or settings.window_size
    dummy_input = torch.randn(1, model_cfg.in_channels, height, width, dtype=torch.float32)

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        },
    )

    print(f"ONNX model saved to {output_path}")
    print(
        "Architecture:",
        f"name={model_cfg.model_name}",
        f"in_channels={model_cfg.in_channels}",
        f"num_classes={model_cfg.num_classes}",
        f"base_channels={model_cfg.base_channels}",
    )


if __name__ == "__main__":
    main()
