# SASS-AKCMamba 项目总览

本项目在 AKCMamba-YOLO 基础上实现尺度感知稀疏扫描原型。当前目标是先验证模型和训练链路可运行，再在 NVIDIA GPU 环境中开展正式训练、测速和消融。

## 运行入口

- `run_sass_akcmamba_feasibility.py`：SASS-AKCMamba 最小可行性脚本，支持模型构建、随机前向、预测和 COCO8 小训练。
- `run_akcmamba_detection.py`：AKCMamba-YOLO 基线推理脚本，可加载外部训练权重；无权重时只验证结构和推理链路。

## 核心代码

主要工作代码位于 `ultralytics/`。其中：

- `ultralytics/ultralytics/nn/modules/block.py`：包含 AKConv、AKSS2D、C3/C4AKCMamba，以及新增的 `SASSAKSS2D`、`C3SASSAKCMamba` 和 `C4SASSAKCMamba`。
- `ultralytics/ultralytics/nn/modules/__init__.py`：导出新增模块。
- `ultralytics/ultralytics/nn/tasks.py`：注册 YAML 模型解析所需模块。
- `ultralytics/ultralytics/cfg/models/v8/yolov8-akcmamba.yaml`：AKCMamba-YOLO 基线配置。
- `ultralytics/ultralytics/cfg/models/v8/yolov8-sass-akcmamba.yaml`：SASS-AKCMamba 原型配置。

## 尺度策略

`SASSAKSS2D` 当前支持四种模式：

- `none`：保留 AKConv 局部分支，不执行 SSM 扫描。
- `atrous`：按空间步长下采样后执行四方向扫描，再插值回原尺寸。
- `window`：在非重叠局部窗口中执行四方向扫描。
- `full`：执行完整四方向全局扫描。

默认 YAML 在 P3 使用 `none`、P4 使用 `atrous`、P5 使用 `full`。

## 资料与日志

- `papers/README.md`：相关论文目录和官方链接。PDF 不进入 Git 历史。
- `logs/sass_akcmamba_experiment_log.md`：初始实现和可行性验证记录。
- `AGENTS.md`：编码 AI 的项目上下文、修改原则和验证要求。

## 本地生成内容

以下目录不提交仓库：

- `datasets/`：本地数据集。
- `runs/`：训练、验证和预测输出。
- `.venv/`：本地虚拟环境。
- `.ultralytics_config/`、`.matplotlib/`：工具配置和缓存。

## 当前状态

- 已完成模型 YAML 解析、随机前向、图片预测保存和 COCO8 一轮训练烟测。
- COCO8 烟测只证明训练链路可用，不作为性能依据。
- 尚未获得有效检测权重，也尚未完成正式数据集训练、消融或真实延迟测试。
