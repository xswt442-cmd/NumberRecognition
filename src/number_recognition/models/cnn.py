import torch
from torch import nn


class CNN(nn.Module):
    """卷积神经网络 - MNIST 分类
    
    结构：
    - Conv 5x5, 6 channels + ReLU + MaxPool 2x2
    - Conv 5x5, 16 channels + ReLU + MaxPool 2x2  
    - Flatten
    - FC 120 + ReLU
    - FC 84 + ReLU
    - FC 10 (输出)
    """

    def __init__(self):
        super().__init__()
        # 卷积层
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 1x28x28 -> 6x28x28
        self.pool1 = nn.MaxPool2d(2, 2)  # 6x28x28 -> 6x14x14
        
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # 6x14x14 -> 16x10x10
        self.pool2 = nn.MaxPool2d(2, 2)  # 16x10x10 -> 16x5x5
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Conv1 + ReLU + Pool
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        # Conv2 + ReLU + Pool
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC层
        x = self.fc1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = torch.relu(x)
        
        x = self.fc3(x)
        return x
