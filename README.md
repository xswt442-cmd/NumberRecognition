# NumberRecognition

## 项目说明
本仓库基于 NumberRecognition 原始仓库 fork 而来，用于代码复现、结构优化与实验测试。

## 功能概览
- MNIST 原始 idx 数据解析为按类别分目录的 PNG 数据集
- 基于 PyTorch 的 MLP（全连接网络）训练
- 使用训练后的权重进行推理并输出准确率

## 快速开始
```bash
pip install -r requirements.txt
python parse_train_images_labels.py
python parse_t10k_images_labels.py
python model_train.py
python model_inference.py
```

## 重构后代码结构
```text
NumberRecognition/
├─ src/
│  └─ number_recognition/
│     ├─ data/
│     │  ├─ prepare_train.py
│     │  └─ prepare_test.py
│     ├─ models/
│     │  └─ mlp.py
│     ├─ train.py
│     └─ inference.py
├─ model.py
├─ model_train.py
├─ model_inference.py
├─ parse_train_images_labels.py
└─ parse_t10k_images_labels.py
```

说明：根目录脚本目前作为兼容入口保留，内部调用 src 下的结构化模块。

## 模块化运行方式（可选）
```bash
python -m src.number_recognition.data.prepare_train
python -m src.number_recognition.data.prepare_test
python -m src.number_recognition.train
python -m src.number_recognition.inference
```

## 运行环境
- Python 3.12（本地测试环境）
- PyTorch
- torchvision
- Pillow
- numpy

## 数据与结果
- 原始数据目录：MNIST_data/
- 解析后训练集：mnist_train/
- 解析后测试集：mnist_test/
- 模型权重：mnist.pth

历史基线结果示例：测试准确率约 97.77%。

## 致谢
- MNIST 数据集：Yann LeCun 团队
- 深度学习框架：PyTorch

## 许可证
本项目遵循仓库内 LICENSE 文件约定。