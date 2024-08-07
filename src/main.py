from models.Generator import Generator
from models.ClassicalDiscriminator import Discriminator as Cdiscriminator
from models.QuantumDiscriminator import Discriminator as Qdiscriminator

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Hyperparameters
lr = 0.001
num_epochs = 50
seed = 111

# Set seed
torch.manual_seed(seed)

# Use cuda if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using cuda:', torch.cuda.get_device_name(device=device))
else:
    device = torch.device("cpu")
    print('Using cpu: aborded')
    exit()

# Load models
generator = Generator().to(device=device)
classicalDiscriminator = Cdiscriminator.to(device=device)
#quantumDiscriminator = Qdiscriminator.to(device)

# Load data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
train_set = torchvision.datasets.MNIST(
    root="../data", train=True, download=True, transform=transform
)
batch_size = 32
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True
)
real_samples, mnist_labels = next(iter(train_loader))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])