from models import Generator, Cdiscriminator, Qdiscriminator
from features import get_data_loader, get_device, get_checkpoint, train_model
from visualization import show_sample_data 

import torch
import os

# Hyperparameters
d_lr = 0.001
g_lr = 0.001
num_epochs = 50
batch_size = 32
seed = 111

checkpoint_folder = os.path.abspath('checkpoints')

# Set seed
torch.manual_seed(seed)

# Use cuda if available
device = get_device()

# Load models
generator = Generator().to(device=device)
classical_discriminator = Cdiscriminator.to(device=device)
#quantumDiscriminator = Qdiscriminator.to(device=device)

# Load data
train_loader = get_data_loader(batch_size=batch_size)

# Plot some training samples
real_samples, mnist_labels = next(iter(train_loader))
show_sample_data(real_samples, sample_size=16)

# Set up optimizers
optimizer_classical_discriminator = torch.optim.Adam(classical_discriminator.parameters(), lr=d_lr)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=g_lr)

# Load checkpoint
start_epoch, loss_values = get_checkpoint(checkpoint_folder=checkpoint_folder, 
                                         classical_discriminator=classical_discriminator, 
                                         generator=generator, 
                                         optimizer_generator=optimizer_generator, 
                                         optimizer_classical_discriminator=optimizer_classical_discriminator)

# Train model
train_model(device=device, 
            num_epochs=num_epochs,
            train_loader=train_loader,
            generator=generator, 
            classical_discriminator=classical_discriminator, 
            optimizer_generator=optimizer_generator,
            optimizer_classical_discriminator=optimizer_classical_discriminator, 
            loss_function=torch.nn.BCELoss(), 
            checkpoint_folder=checkpoint_folder,  
            start_epoch=start_epoch,
            loss_values=loss_values,
            checkpoint_interval=5) 