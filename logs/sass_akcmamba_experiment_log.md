# SASS-AKCMamba-YOLO 可行性实验日志

## 实验目标

本实验只做可行性验证，不做性能结论。目标是在现有 AKCMamba-YOLO 复现基础上，搭建一个可运行的尺度感知稀疏扫描模型，使规划中的 SASS 模块能够完成模型构建、前向推理、预测和小数据集训练链路。

## 初始环境

- 工作路径：项目根目录
- 代码路径：`ultralytics/`
- Python：项目本地虚拟环境
- PyTorch：`2.8.0`
- MPS：可用
- 当前日期：`2026-07-01`

## 执行记录

### 2026-07-01 18:24:31 CST

- 检查现有仓库状态。
- 发现 `ultralytics` 工作树已有 AKCMamba 复现改动：
  - `ultralytics/nn/modules/block.py`
  - `ultralytics/nn/modules/__init__.py`
  - `ultralytics/nn/tasks.py`
  - `ultralytics/cfg/models/v8/yolov8-akcmamba.yaml`
- 决定在现有改动基础上继续实现 SASS，不回退已有内容。

### 2026-07-01 18:30 CST

- 在 `ultralytics/nn/modules/block.py` 中新增：
  - `SASSAKSS2D`
  - `C3SASSAKCMamba`
  - `C4SASSAKCMamba`
- `SASSAKSS2D` 支持四种扫描模式：
  - `none`：只保留 AKConv 局部适配，不做 SSM 扫描。
  - `atrous`：先按空间步长下采样，再做四方向扫描，最后插值回原尺寸。
  - `window`：按非重叠窗口做局部四方向扫描。
  - `full`：保持原 AKSS2D 的四方向全局扫描。
- 在 `ultralytics/nn/modules/__init__.py` 和 `ultralytics/nn/tasks.py` 中注册新模块。
- 新增配置 `ultralytics/cfg/models/v8/yolov8-sass-akcmamba.yaml`：
  - P3：`none`
  - P4：`atrous`
  - P5：`full`
- 已通过导入检查和 YOLO 模型构建检查。

### 2026-07-01 18:40 CST

- 完成随机张量前向验证：
  - 命令：使用 `YOLO('ultralytics/cfg/models/v8/yolov8-sass-akcmamba.yaml')` 构建模型后，输入 `torch.zeros(1, 3, 128, 128)`。
  - 结果：前向通过，输出检测张量。
- 完成未训练权重预测链路验证：
  - 输入图片：`ultralytics/ultralytics/assets/bus.jpg`
  - 输出目录：`runs/sass_akcmamba_feasibility/predict_random_init`
- 完成 COCO8 一轮小训练：
  - 数据集：`coco8.yaml`
  - 设备：MPS
  - 图片尺寸：128
  - batch：1
  - epoch：1
  - 输出目录：`runs/sass_akcmamba_feasibility/coco8_1epoch_img128`
  - 权重：
    - `weights/best.pt`
    - `weights/last.pt`
  - 说明：这是随机初始化 1 epoch 烟测，指标不用于性能分析。
- 完成加载训练权重后的预测链路验证：
  - 权重：`runs/sass_akcmamba_feasibility/coco8_1epoch_img128/weights/best.pt`
  - 输出目录：`runs/sass_akcmamba_feasibility/predict_after_coco8_1epoch`
  - 由于只训练 1 epoch 且尺寸很小，没有检出框；该现象只说明性能未训练充分，不影响可运行性结论。
- 新增复现实验脚本：
  - `run_sass_akcmamba_feasibility.py`
  - 已通过 `--skip-train --imgsz 128` 烟测。

## 当前可行性结论

- SASS-AKCMamba 模块可以被 Ultralytics YAML 正常解析。
- 模型可以完成随机张量前向。
- 模型可以完成图片预测并保存可视化结果。
- 模型可以在 COCO8 上完成检测训练、权重保存和验证链路。
- 目前只完成“能用”的可行性验证，尚未做性能验证、消融实验或正式数据集训练。
- `results.csv` 中当前 mAP 为 0，这是 `imgsz=128、batch=1、epoch=1、随机初始化` 的烟测设置导致，只表示模型没有被充分训练，不作为方法效果判断。
