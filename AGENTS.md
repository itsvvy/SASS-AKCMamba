# AGENTS.md

本文档供 Codex 等编码 AI 在本仓库中工作时读取。

## 项目目标

项目研究尺度感知稀疏选择扫描在 AKCMamba-YOLO 中的可行性。核心不是简单增加 Mamba 模块，而是减少多尺度检测网络中的冗余扫描，并保留局部细节。

当前固定原型策略：P3 使用 `none`，P4 使用 `atrous`，P5 使用 `full`。`window` 模式已经实现，但尚未纳入默认模型配置。

## 关键代码

- `ultralytics/ultralytics/nn/modules/block.py`：核心模块实现。
- `ultralytics/ultralytics/nn/modules/__init__.py`：模块导出。
- `ultralytics/ultralytics/nn/tasks.py`：YAML 模型解析注册。
- `ultralytics/ultralytics/cfg/models/v8/yolov8-sass-akcmamba.yaml`：默认 SASS 模型。
- `ultralytics/ultralytics/cfg/models/v8/yolov8-akcmamba.yaml`：AKCMamba 基线。

## 修改原则

- 保持 Ultralytics 现有接口和 YAML 构建方式，不绕过 `YOLO` 模型入口。
- 不把 COCO8 烟测结果描述为模型性能。
- 不声称精度、FPS、FLOPs 或延迟改善，除非存在可复查的公平实验记录。
- 扫描位置、步长或模式发生变化时，同步更新 YAML、文档和消融计划。
- 对算法核心或不直观的兼容处理使用简短中文注释，避免逐行解释。
- 不提交数据集、训练输出、缓存、第三方论文 PDF 或大模型权重。

## 最小验证

修改模型代码后至少执行：

```bash
python run_sass_akcmamba_feasibility.py --skip-train --imgsz 128
```

需要检查训练链路时执行：

```bash
python run_sass_akcmamba_feasibility.py --epochs 1 --imgsz 128 --batch 1
```

正式实验应固定数据划分、输入尺寸、batch、epoch、优化器、随机种子、设备和测速方法，并记录 mAP50、mAP50-95、APs、参数量、FLOPs、端到端延迟、FPS 和显存。

## 输出约定

- 本地数据放在 `datasets/`，不提交 Git。
- 训练、验证和预测输出放在 `runs/`，不提交 Git。
- 正式日志放在 `logs/`，应记录命令、提交版本、环境和结论边界。
- 权重通过独立发布附件或模型存储管理，不直接进入 Git 历史。

## 当前状态

当前仅证明模型可以构建、前向、预测和完成 COCO8 一轮训练。没有正式预训练权重，没有性能验证，也没有完整消融结果。
