"""运行 SASS-AKCMamba-YOLO 的最小可行性验证。

中文说明：
本脚本只验证模型结构、前向推理、预测、COCO8 小训练链路是否可运行。
它不用于报告性能结论，默认使用很小的 imgsz 和 1 个 epoch，适合 M1 芯片快速烟测。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
ULTRALYTICS_DIR = ROOT / "ultralytics"
DEFAULT_CFG = ULTRALYTICS_DIR / "ultralytics/cfg/models/v8/yolov8-sass-akcmamba.yaml"
DEFAULT_PROJECT = ROOT / "runs/sass_akcmamba_feasibility"
DEFAULT_SOURCE = ULTRALYTICS_DIR / "ultralytics/assets/bus.jpg"

# 中文注释：从项目根目录运行时，外层 ultralytics 目录会被识别成 namespace，
# 因此需要显式把真正的源码根目录放到 import 搜索路径最前面。
sys.path.insert(0, str(ULTRALYTICS_DIR))

from ultralytics import YOLO


def default_device() -> str:
    """根据当前机器选择默认设备。"""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="SASS-AKCMamba-YOLO feasibility smoke test.")
    parser.add_argument("--cfg", type=Path, default=DEFAULT_CFG, help="SASS-AKCMamba model YAML.")
    parser.add_argument("--data", type=str, default="coco8.yaml", help="Ultralytics dataset YAML.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="Image used for prediction smoke test.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Output project directory.")
    parser.add_argument("--imgsz", type=int, default=128, help="Small image size for quick feasibility runs.")
    parser.add_argument("--epochs", type=int, default=1, help="Tiny training epochs.")
    parser.add_argument("--batch", type=int, default=1, help="Tiny training batch size.")
    parser.add_argument("--device", type=str, default=default_device(), help="Device, e.g. mps/cpu/0.")
    parser.add_argument("--skip-train", action="store_true", help="Only run build/forward/predict checks.")
    return parser.parse_args()


def main() -> None:
    """执行模型构建、前向、预测和可选小训练。"""
    args = parse_args()
    model = YOLO(str(args.cfg))

    # 中文注释：随机张量前向用于验证网络拓扑和 Detect head 输出，不依赖数据集。
    model.model.eval()
    with torch.no_grad():
        _ = model.model(torch.zeros(1, 3, args.imgsz, args.imgsz))
    print("forward smoke test ok")

    # 中文注释：未训练权重预测只验证图片预处理、模型推理、结果保存链路。
    model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        device=args.device,
        project=str(args.project),
        name="script_predict_random_init",
        exist_ok=True,
        save=True,
        verbose=False,
    )
    print("random-init prediction smoke test ok")

    if args.skip_train:
        return

    # 中文注释：COCO8 一轮训练只验证检测训练、损失、权重保存和最终验证链路。
    train_result = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(args.project),
        name=f"script_{args.data.replace('.yaml', '')}_{args.epochs}epoch_img{args.imgsz}",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    print(f"tiny train smoke test ok: {train_result.save_dir}")


if __name__ == "__main__":
    main()
