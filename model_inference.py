from model import Network  # 导入自定义的神经网络模型类
from torchvision import transforms  # 导入torchvision图像转换工具
from torchvision import datasets  # 导入torchvision数据集工具
import torch  # 导入PyTorch库

if __name__ == '__main__':
    # 定义图像转换:转换为灰度图并转换为张量
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor()
    ])

    # 从指定目录加载测试数据集并应用转换
    test_dataset = datasets.ImageFolder(root='./mnist_test', transform=transform)
    # 打印测试数据集的大小
    print("test_dataset length: ", len(test_dataset))

    # 初始化神经网络模型
    model = Network()
    # 从保存的文件加载模型参数
    model.load_state_dict(torch.load('mnist.pth'))

    right = 0  # 初始化正确分类的图像计数器

    # 遍历测试数据集
    for i, (x, y) in enumerate(test_dataset):
        output = model(x.unsqueeze(0))  # 前向传播:添加批次维度并计算模型输出
        predict = output.argmax(1).item()  # 获取得分最高的索引作为预测标签
        if predict == y:
            right += 1  # 如果预测正确,计数器加1
        else:
            img_path = test_dataset.samples[i][0]  # 获取错误分类图像的路径
            # 打印错误分类的详细信息
            print(f"wrong case: predict = {predict} actual = {y} img_path = {img_path}")

    sample_num = len(test_dataset)  # 获取测试数据集的总样本数
    acc = right * 1.0 / sample_num  # 计算准确率(正确数/总数)
    # 打印测试准确率
    print("test accuracy = %d / %d = %.31f" % (right, sample_num, acc))
