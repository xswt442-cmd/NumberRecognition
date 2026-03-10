import torch  # 导入PyTorch库
from torch import nn  # 从PyTorch导入神经网络模块

# 定义神经网络类,继承自nn.Module
class Network(nn.Module):
    def __init__(self):
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.layer1 = nn.Linear(784, 256)  # 定义第一个全连接层(输入784维,输出256维)
        self.layer2 = nn.Linear(256, 10)  # 定义第二个全连接层(输入256维,输出10维,对应0-9十个数字)

    def forward(self, x):
        x = x.view(-1, 28*28)  # 将输入张量展平为28*28=784维的一维向量
        x = self.layer1(x)  # 通过第一个全连接层
        x = torch.relu(x)  # 应用ReLU激活函数
        return self.layer2(x)  # 通过第二个全连接层并返回结果(10个类别的得分)
