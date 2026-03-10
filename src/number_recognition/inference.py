import torch
from torchvision import datasets
from torchvision import transforms

from .models import Network, CNN


def infer(
    test_root: str = "./mnist_test",
    model_path: str = "./mnist.pth",
    model_type: str = "cnn",
) -> None:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    test_dataset = datasets.ImageFolder(root=test_root, transform=transform)
    print("test_dataset length: ", len(test_dataset))

    # 选择模型类型
    if model_type.lower() == "cnn":
        model = CNN()
        print(f"使用 CNN 模型进行推理")
    else:
        model = Network()
        print(f"使用 MLP 模型进行推理")
    
    model.load_state_dict(torch.load(model_path))

    right = 0

    for i, (x, y) in enumerate(test_dataset):
        output = model(x.unsqueeze(0))
        predict = output.argmax(1).item()
        if predict == y:
            right += 1
        else:
            img_path = test_dataset.samples[i][0]
            print(f"wrong case: predict = {predict} actual = {y} img_path = {img_path}")

    sample_num = len(test_dataset)
    acc = right * 1.0 / sample_num
    print("test accuracy = %d / %d = %.31f" % (right, sample_num, acc))


if __name__ == "__main__":
    infer()
