#!/usr/bin/env python3
"""Run local AKCMamba-YOLO detection on an Ultralytics sample image."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ULTRALYTICS_REPO = ROOT / "ultralytics"
DEFAULT_CFG = ULTRALYTICS_REPO / "ultralytics" / "cfg" / "models" / "v8" / "yolov8-akcmamba.yaml"
DEFAULT_SOURCE = ULTRALYTICS_REPO / "ultralytics" / "assets" / "bus.jpg"
DEFAULT_PROJECT = ROOT / "runs" / "akcmamba_detection"

# 中文注释：在导入 Ultralytics/Matplotlib 之前固定配置目录，避免直接运行时写入用户 Library 或 home 目录。
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics_config"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

# 中文注释：优先使用本地修改后的 Ultralytics 仓库，确保加载的是 AKCMamba 复现模块。
sys.path.insert(0, str(ULTRALYTICS_REPO))

import cv2  # noqa: E402
from ultralytics import YOLO, settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run AKCMamba-YOLO local object detection.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Input image path.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="AKCMamba-YOLO model YAML path.")
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path(os.environ["AKCMAMBA_WEIGHTS"]) if os.environ.get("AKCMAMBA_WEIGHTS") else None,
        help="Optional trained AKCMamba .pt weights. Defaults to AKCMAMBA_WEIGHTS if set.",
    )
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output project directory.")
    parser.add_argument("--name", default="bus", help="Output run name.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=20, help="Maximum detections to draw.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, mps, cuda:0.")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> tuple[YOLO, bool]:
    """Load trained AKCMamba weights if present, otherwise instantiate the AKCMamba YAML."""
    if args.weights:
        if not args.weights.exists():
            raise FileNotFoundError(f"AKCMamba weights not found: {args.weights}")
        return YOLO(str(args.weights)), True
    if not args.cfg.exists():
        raise FileNotFoundError(f"AKCMamba config not found: {args.cfg}")
    # 中文注释：当前官方仓库未提供训练权重；无权重时只验证 AKCMamba 结构推理链路。
    return YOLO(str(args.cfg)), False


def annotate_untrained_warning(image, has_weights: bool):
    """Add a visible note when no trained AKCMamba weights are loaded."""
    if has_weights:
        return image
    # 中文注释：随机初始化模型的框不具备检测语义，输出图中直接标明这一点。
    note = "AKCMamba architecture smoke test: no trained weights loaded"
    cv2.rectangle(image, (10, 10), (10 + min(760, image.shape[1] - 20), 48), (0, 0, 0), thickness=-1)
    cv2.putText(image, note, (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    return image


def main() -> None:
    """Run prediction and save a result image."""
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Source image not found: {args.source}")

    # 中文注释：将 Ultralytics 的运行目录固定到项目目录下，避免写入用户配置目录。
    settings.update({"runs_dir": str(ROOT / "runs"), "datasets_dir": str(ROOT / "datasets")})

    model, has_weights = load_model(args)
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        device=args.device,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        save=True,
        verbose=True,
    )

    result = results[0]
    plotted = result.plot()
    plotted = annotate_untrained_warning(plotted, has_weights)
    output_path = Path(result.save_dir) / f"{args.source.stem}_akcmamba_result.jpg"
    cv2.imwrite(str(output_path), plotted)

    box_count = 0 if result.boxes is None else len(result.boxes)
    print(f"model_source={'weights' if has_weights else 'yaml-random-init'}")
    print(f"source={args.source}")
    print(f"boxes={box_count}")
    print(f"result_image={output_path}")


if __name__ == "__main__":
    main()
