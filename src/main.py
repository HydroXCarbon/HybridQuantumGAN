from features import get_data_loader, get_device, get_checkpoint, train_model, get_model
from visualization import show_sample_data, generate_sample
from models import Generator, ClassicalDiscriminator, QuantumDiscriminator

import torch
import os
import matplotlib.pyplot as plt

# Hyperparameters (can have only 1 generator)
models = {'generator':{'learning_rate':0.0002, 'model_class':Generator}, 
          'classical_discriminator':{'learning_rate':0.00015, 'model_class':ClassicalDiscriminator},
          'quantum_discriminator':{'learning_rate':0.00015, 'model_class':QuantumDiscriminator}
        }
num_epochs = 50
batch_size = 32

# Configuration Settings
seed = 111
checkpoint_interval = 5
training_mode = 'alternating'  # training mode 'alternating' or 'commbined'
show_sample = True
load_checkpoint = True
training = False
generate_data = True

# Set up folders path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, '..', 'data')
checkpoint_folder = os.path.join(script_dir, 'checkpoints')

# Create required folders
os.makedirs(checkpoint_folder, exist_ok=True)
os.makedirs(data_folder, exist_ok=True)

# Set seed
if seed is not None:
  torch.manual_seed(seed)

# Use cuda if available
device = get_device()

# Load models
model_list, optimizer_list = get_model(models, device)

# Load data
train_loader = get_data_loader(batch_size=batch_size, data_folder=data_folder)

# Plot some training samples
if show_sample:
  real_samples, labels = next(iter(train_loader))
  show_sample_data(real_samples, sample_size=16)

# Set up loss function
loss_function = torch.nn.BCELoss()

# Load checkpoint
start_epoch = 0
loss_values = None
if load_checkpoint:
  start_epoch, loss_values = get_checkpoint(checkpoint_folder=checkpoint_folder, 
                                            model_list=model_list,
                                            optimizer_list=optimizer_list)

# Train model
if training:
  train_model(device=device, 
              num_epochs=num_epochs,
              train_loader=train_loader,
              model_list=model_list,
              optimizer_list=optimizer_list,
              loss_function=loss_function,
              checkpoint_folder=checkpoint_folder,
              start_epoch=start_epoch,
              loss_values=loss_values,
              checkpoint_interval=checkpoint_interval,
              training_mode=training_mode) 

# Generate sample
if generate_data:
  generate_sample(model_list[0], device, batch_size=16)

# Wait for user to close the plot
plt.ioff()
plt.show()