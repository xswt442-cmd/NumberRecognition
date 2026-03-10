# NumberRecognition

## 项目说明
本仓库基于 NumberRecognition 原始仓库 fork 而来，用于代码复现、结构优化与实验测试。

**mainwithup 分支：** CNN 改进版本，使用卷积神经网络替代原有的 MLP 模型。

## 功能概览
- MNIST 原始 idx 数据解析为按类别分目录的 PNG 数据集
- 基于 PyTorch 的卷积神经网络（CNN）训练（支持切换到 MLP）
- 使用训练后的权重进行推理并输出准确率

## 模型架构（CNN）
```
Input(1x28x28)
  ↓
Conv2d(1→6, 5x5) + ReLU + MaxPool(2x2)
  ↓ (6x14x14)
Conv2d(6→16, 5x5) + ReLU + MaxPool(2x2)
  ↓ (16x5x5)
Flatten
  ↓
FC(400→120) + ReLU
  ↓
FC(120→84) + ReLU
  ↓
FC(84→10)
  ↓
Output(10)
```

## 快速开始
```bash
pip install -r requirements.txt
python parse_train_images_labels.py
python parse_t10k_images_labels.py
python model_train.py
python model_inference.py
```

## 模型选择与自定义运行
默认使用 CNN 模型训练和推理。如需使用 MLP（原始模型），可按如下方式修改：

**Python 脚本中调用：**
```python
# 训练时选择模型
from src.number_recognition.train import train
train(model_type="cnn")    # CNN（推荐，默认）
train(model_type="mlp")    # MLP（原始）

# 推理时选择模型
from src.number_recognition.inference import infer
infer(model_type="cnn")    # CNN
infer(model_type="mlp")    # MLP
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
│     │  ├─ mlp.py          （原有的全连接网络）
│     │  ├─ cnn.py          （新增的卷积神经网络）
│     │  └─ __init__.py
│     ├─ train.py           （支持模型选择）
│     └─ inference.py       （支持模型选择）
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
- 解析后训练集：mnist_train/（60,000张图片）
- 解析后测试集：mnist_test/（10,000张图片）
- 模型权重：mnist.pth

### 模型性能对比

| 模型 | 网络结构 | 参数量 | 训练集大小 | 准确率 | 性能提升 |
|-----|--------|------|---------|------|--------|
| MLP | 784→256→10 | ~200K | 60,000 | 97.81% | 基线 |
| CNN | Conv(1→6)→Conv(6→16)→FC(120)→FC(84)→FC(10) | ~36K | 60,000 | **98.86%** | +1.05% |

**关键观察：**
- CNN 参数量仅为 MLP 的 18%，却获得了 1.05% 的准确率提升
- 卷积层有效地捕捉了图像的空间特征，提升了分类性能
- 训练收敛更快，loss 从 Epoch 1 的 2.31 快速下降

### 历史实验结果

**MLP 模型（mainwithcn）:**
- 网络配置：784→256→10
- 测试准确率：9781/10000 = **97.81%**
- 训练轮数：10 epochs
- 训练特点：loss 从 2.3043 下降到 0.0058

**CNN 模型（mainwithup）:**
- 网络配置：Conv(1→6, 5×5) + ReLU + MaxPool → Conv(6→16, 5×5) + ReLU + MaxPool → FC(120) → FC(84) → FC(10)
- 测试准确率：9886/10000 = **98.86%**
- 训练轮数：10 epochs
- 训练特点：loss 从 2.3091 下降到 0.0632，收敛更快更稳定
- **性能提升：+1.05%，参数减少 82%**

## 致谢
- MNIST 数据集：Yann LeCun 团队
- 深度学习框架：PyTorch

## 许可证
本项目遵循仓库内 LICENSE 文件约定。
