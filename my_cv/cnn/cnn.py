import numpy as np
from conv import Conv3x3
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.ToTensor()

train_ds = datasets.MNIST(
    root="~/.pytorch/MNIST_data",
    train=True,
    download=True,
    transform=transform
)

image_tensor, label = train_ds[0]

image = image_tensor.squeeze().numpy()  # now shape is (28,28)

conv = Conv3x3(8)
output = conv.forward(image)

print(output.shape)  # → (26, 26, 8)
