from models import Generator, Cdiscriminator, Qdiscriminator
from features import get_data_loader, get_device, get_checkpoint, train_model
from visualization import show_sample_data 

import torch

# Hyperparameters
d_lr = 0.001
g_lr = 0.001
num_epochs = 50
seed = 111
batch_size = 32

# Set seed
torch.manual_seed(seed)

# Use cuda if available
device = get_device()

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

# Load checkpoint
start_epoch = get_checkpoint(classicalDiscriminator, generator, optimizer_generator, optimizer_discriminator)

# Train model
train_model(device, 
            generator, 
            classicalDiscriminator, 
            optimizer_generator,
            optimizer_discriminator, 
            train_loader, 
            num_epochs, 
            batch_size, 
            start_epoch)