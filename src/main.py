from models import Generator, Cdiscriminator, Qdiscriminator
from features import get_data_loader
from visualization import show_sample_data

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Hyperparameters
d_lr = 0.001
g_lr = 0.001
num_epochs = 50
seed = 111
batch_size = 32

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
train_loader = get_data_loader(batch_size=batch_size)

# Plot some training samples
real_samples, mnist_labels = next(iter(train_loader))
show_sample_data(real_samples, sample_size=16)

# Set up optimizers
optimizer_discriminator = torch.optim.Adam(classicalDiscriminator.parameters(), lr=d_lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=g_lr)