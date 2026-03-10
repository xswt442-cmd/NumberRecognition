import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from .models import Network


def train(
    train_root: str = "./mnist_train",
    test_root: str = "./mnist_test",
    model_path: str = "./mnist.pth",
    batch_size: int = 64,
    epochs: int = 10,
) -> None:
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root=train_root, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_root, transform=transform)

    print("train_dataset length: ", len(train_dataset))
    print("test_dataset length: ", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("train_loader length: ", len(train_loader))

    for batch_idx, (data, label) in enumerate(train_loader):
        print("batch_idx: ", batch_idx)
        print("data.shape: ", data.shape)
        print("label.shape: ", label.shape)
        print(label)

    model = Network()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_idx, (data, label) in enumerate(train_loader):
            output = model(data)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_idx % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{epochs} "
                    f"| Batch {batch_idx}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

    torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    train()
