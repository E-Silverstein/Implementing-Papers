import numpy as np
import torch
import torch.nn as nn
from torch.nn import (
    BatchNorm2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Hyperparameters
num_classes = 10  # Using CIFAR-10 dataset instead of ImageNet for demonstration
batch_size = 64
learning_rate = 0.005
device = torch.device("mps")  # Running on Apple Silicon


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = Sequential(
            Conv2d(3, 96, 11, stride=4), BatchNorm2d(96), ReLU(), MaxPool2d(3, 2)
        )
        self.layer2 = Sequential(
            Conv2d(96, 256, 5, padding=2), BatchNorm2d(256), ReLU(), MaxPool2d(3, 2)
        )
        self.layer3 = Sequential(
            Conv2d(256, 384, 3, padding=1), BatchNorm2d(384), ReLU()
        )
        self.layer4 = Sequential(
            Conv2d(384, 384, 3, padding=1), BatchNorm2d(384), ReLU()
        )
        self.layer5 = Sequential(
            Conv2d(384, 256, 3, padding=1), BatchNorm2d(256), ReLU(), MaxPool2d(3, 2)
        )

        self.layer6 = Sequential(
            Flatten(), Dropout(0.5), Linear(256 * 6 * 6, 4096), ReLU()
        )

        self.layer7 = Sequential(Dropout(0.5), Linear(4096, 4096), ReLU())

        self.layer8 = Sequential(Linear(4096, 10))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        return x


def get_data_loaders(batch_size):
    transform = transforms.Compose(
        [
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225]),
            ),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


model = AlexNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(), lr=learning_rate, weight_decay=0.005, momentum=0.9
)

train_loader, test_loader = get_data_loaders(batch_size)

total_step = len(train_loader)

for epoch in range(5):
    for i, (images, labels) in tqdm(enumerate(train_loader)):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(
                "\nEpoch [{}/{}], Step [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, 5, i + 1, total_step, loss.item()
                )
            )
    # Validation accuracy
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print("Validation Accuracy {} %".format(100 * correct / total))

# Save the model checkpoint
torch.save(model.state_dict(), "alex_net.ckpt")

# Test the model
model.eval()
correct = 0
total = 0
for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
print("Test Accuracy {} %".format(100 * correct / total))
