# Based on Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks

import torch
from torch.nn.modules.conv import Conv2d
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import cv2
from imutils import build_montages

############### Hyper Params ###############
output_path = "./dcgan_output"
epochs = 20
batch_size = 128
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


################## Models ##################
class Generator(nn.Module):
    def __init__(self, inputDim=100, outputChannels=1, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = nn.ConvTranspose2d(
            in_channels=inputDim,
            out_channels=128,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False,
        )
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(num_features=128)

        self.conv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(num_features=64)

        self.conv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(num_features=32)

        self.conv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=outputChannels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False,
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.bn3(x)

        x = self.conv4(x)
        out = self.tanh(x)

        return out


class Discriminator(nn.Module):
    def __init__(self, depth, alpha=0.2, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.conv1 = Conv2d(
            in_channels=depth, out_channels=32, kernel_size=4, stride=2, padding=1
        )
        self.leakyRelu1 = nn.LeakyReLU(alpha, inplace=True)

        self.conv2 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1
        )
        self.leakyRelu2 = nn.LeakyReLU(alpha, inplace=True)

        self.fc1 = nn.Linear(in_features=3136, out_features=512)
        self.leakyRelu3 = nn.LeakyReLU(alpha, inplace=True)

        self.fc2 = nn.Linear(in_features=512, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyRelu1(x)

        x = self.conv2(x)
        x = self.leakyRelu2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.leakyRelu3(x)

        x = self.fc2(x)
        output = self.sigmoid(x)

        return output


############### Training ###############


def weight_init(model):
    classname = model.__class__.__name__

    if classname.find("conv") != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find("bn") != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)  # bias to 0


train_data = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
    ),
    download=True,
)

train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

steps_per_epoch = len(train_loader.dataset) // batch_size

print("Building Generator")
gen = Generator(inputDim=100, outputChannels=1)
gen.apply(weight_init)
gen.to(device)

print("Building Discriminator")
discrim = Discriminator(depth=1)
discrim.apply(weight_init)
discrim.to(device)

gen_opt = torch.optim.Adam(
    gen.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002 / epochs
)
discrim_opt = torch.optim.Adam(
    discrim.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.0002 / epochs
)

loss = nn.BCELoss()

print("Starting Training")

benchmark_noise = torch.randn(256, 100, 1, 1, device=device)

realLabel = 1
fakeLabel = 0

for epoch in range(epochs):
    print("Starting epoch {} of {}...".format(epoch + 1, epochs))

    epoch_loss_g = 0
    epoch_loss_d = 0

    for x in train_loader:
        discrim.zero_grad()

        images = x[0]
        images = images.to(device)

        bs = images.size(0)
        labels = torch.full((bs,), realLabel, dtype=torch.float, device=device)

        output = discrim(images).view(-1)
        real_error = loss(output, labels)
        real_error.backward()

        noise = torch.randn(bs, 100, 1, 1, device=device)

        fake = gen(noise)
        labels.fill_(fakeLabel)

        output = discrim(fake.detach()).view(-1)
        fake_error = loss(output, labels)
        fake_error.backward()

        discrim_error = real_error + fake_error
        discrim_opt.step()

        gen.zero_grad()

        labels.fill_(realLabel)
        output = discrim(fake).view(-1)

        gen_error = loss(output, labels)
        gen_error.backward()

        gen_opt.step()

        epoch_loss_g += gen_error
        epoch_loss_d += discrim_error

    print(
        "Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(
            epoch_loss_g / steps_per_epoch, epoch_loss_d / steps_per_epoch
        )
    )

    if (epoch + 1) % 2 == 0:
        gen.eval()
        images = gen(benchmark_noise)
        images = images.detach().cpu().numpy().transpose((0, 2, 3, 1))
        images = ((images * 127.5) + 127.5).astype("uint8")
        images = np.repeat(images, 3, axis=-1)
        vis = build_montages(images, (28, 28), (16, 16))[0]

        p = os.path.join(output_path, "epoch_{}.png".format(str(epoch + 1).zfill(4)))
        cv2.imwrite(p, vis)

        gen.train()
