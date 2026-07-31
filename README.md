# SASS-AKCMamba

面向轻量实时目标检测的尺度感知稀疏选择扫描 AKCMamba-YOLO 原型。

本项目基于 Ultralytics YOLO 和 AKCMamba-YOLO，探索一个具体问题：在多尺度检测网络中，不同分辨率特征是否需要相同强度的二维状态空间扫描。当前原型采用固定的尺度策略：

- P3：保留 AKConv 局部建模，不执行 SSM 扫描。
- P4：先按空间步长稀疏采样，再执行四方向扫描。
- P5：执行完整四方向全局扫描。

目前只完成可行性验证，能够构建模型、执行前向传播、保存预测结果并在 COCO8 上完成一轮训练。仓库没有可用于正式检测的预训练权重，也没有性能提升结论。

## 核心文件

- `ultralytics/ultralytics/nn/modules/block.py`：AKConv、AKSS2D 和 SASS-AKCMamba 模块实现。
- `ultralytics/ultralytics/nn/modules/__init__.py`：新增模块导出。
- `ultralytics/ultralytics/nn/tasks.py`：新增模块的 YAML 解析注册。
- `ultralytics/ultralytics/cfg/models/v8/yolov8-akcmamba.yaml`：AKCMamba-YOLO 基线配置。
- `ultralytics/ultralytics/cfg/models/v8/yolov8-sass-akcmamba.yaml`：SASS-AKCMamba 原型配置。
- `run_sass_akcmamba_feasibility.py`：模型构建、前向、预测和 COCO8 小训练烟测。
- `run_akcmamba_detection.py`：AKCMamba 基线推理与权重加载检查。
- `PROJECT_OVERVIEW.md`：项目结构和当前状态总览。
- `AGENTS.md`：供编码 AI 使用的项目约束和验证要求。

## 环境安装

建议使用 Python 3.10 或 3.11，以及与当前 CUDA 环境匹配的 PyTorch。进入代码目录后以可编辑模式安装：

```bash
cd ultralytics
python -m pip install -e .
```

在 NVIDIA GPU 环境中，请先按照 PyTorch 官方说明安装匹配 CUDA 的 `torch` 和 `torchvision`。

## 可行性验证

仅检查模型构建、前向和随机初始化预测：

```bash
python run_sass_akcmamba_feasibility.py --skip-train --imgsz 128
```

额外执行 COCO8 一轮小训练：

```bash
python run_sass_akcmamba_feasibility.py --epochs 1 --imgsz 128 --batch 1
```

在 CUDA 设备上可显式指定：

```bash
python run_sass_akcmamba_feasibility.py --device 0
```

上述命令只验证代码链路，不用于评价 mAP、FPS 或延迟。

## 当前限制

- 尚未在 VOC、VisDrone 或 COCO 上进行充分训练。
- 尚未完成全尺度 AKCMamba、公平参数量基线和各扫描模式的消融实验。
- 当前 `atrous` 模式通过下采样、扫描和插值实现，真实延迟收益必须在目标 GPU 上实测。
- 当前没有正式预训练权重，随机初始化或 COCO8 一轮权重通常无法产生可靠检测框。

## 上游项目

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [AKCMamba-YOLO 官方代码](https://github.com/xlllchen/AKCMamba_YOLO)
- [AKCMamba-YOLO CVPR 2026 论文](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_AKCMamba-YOLO_Selective_State_Space_Models_For_Real-Time_Object_Detection_CVPR_2026_paper.html)

仓库中的 Ultralytics 源码快照基于上游提交 `8b55b87`，AKCMamba 对照代码基于官方仓库提交 `2e6ed60`。SASS 模块是本仓库的研究原型，并非上述上游项目的官方实现。

## 许可证

本仓库基于 Ultralytics 代码修改，按 [AGPL-3.0](LICENSE) 发布。使用和分发时还应遵守相关上游项目、数据集和论文材料各自的许可证与条款。
