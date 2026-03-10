import torch
from torch import nn
from torch import optim
from model import Network
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # 定义图像转换:转换为灰度图并转换为张量
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 从指定目录加载训练数据集并应用转换
    train_dataset = datasets.ImageFolder(root='./mnist_train', transform=transform)
    # 从指定目录加载测试数据集并应用转换
    test_dataset = datasets.ImageFolder(root='./mnist_test', transform=transform)
    # 打印训练数据集的大小
    print("train_dataset length: ", len(train_dataset))
    # 打印测试数据集的大小
    print("test_dataset length: ", len(test_dataset))

    # 创建训练数据加载器,批大小为64,启用数据打乱
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 打印训练数据加载器中的批次数量
    print("train_loader length: ", len(train_loader))

    # 遍历训练数据加载器的前几个批次
    for batch_idx, (data, label) in enumerate(train_loader):
        # 取消下面两行注释可在3个批次后停止
        # if batch_idx == 3:
        #     break
        # 打印批次索引
        print("batch_idx: ", batch_idx)
        # 打印数据张量的形状
        print("data.shape: ", data.shape)
        # 打印标签张量的形状
        print("label.shape: ", label.shape)
        # 打印标签
        print(label)

    # 初始化神经网络模型
    model = Network()
    # 使用Adam优化器初始化模型参数
    optimizer = optim.Adam(model.parameters())
    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 训练模型10个epoch(轮次)
    for epoch in range(10):
        # 遍历训练数据加载器中的所有批次
        for batch_idx, (data, label) in enumerate(train_loader):
            # 前向传播:计算模型输出
            output = model(data)
            # 计算损失值
            loss = criterion(output, label)
            # 反向传播:计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 清零梯度,为下一次迭代做准备
            optimizer.zero_grad()
            # 每100个批次打印一次损失值
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/10 "
                      f"| Batch {batch_idx}/{len(train_loader)} "
                      f"| Loss: {loss.item():.4f}")

    # 将训练好的模型参数保存到文件
    torch.save(model.state_dict(), 'mnist.pth')
