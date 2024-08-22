from features import get_data_loader, get_device, get_checkpoint, train_model, get_model
from visualization import show_sample_data, generate_sample

import torch
import os
import yaml
import matplotlib.pyplot as plt

# Load the configuration file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
Hyperparameter = config['Hyperparameter']
Configuration = config['Configuration']

# Hyperparameters (can have only 1 generator)
model_selector = Hyperparameter['model_selector']
models = Hyperparameter['models']
num_epochs = Hyperparameter['num_epochs']
batch_size = Hyperparameter['batch_size']

# Configuration Settings
seed = Configuration['seed']
checkpoint_interval = Configuration['checkpoint_interval']
training_mode = Configuration['training_mode']
show_sample = Configuration['show_sample']
load_checkpoint = Configuration['load_checkpoint']
training = Configuration['training']
generate_data = Configuration['generate_data']

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
  #algorithm_globals.random_seed = seed

# Use cuda if available
device = get_device()

# Load models
model_list, optimizer_list = get_model(models, model_selector, device)

# Load data
train_loader = get_data_loader(batch_size=batch_size, data_folder=data_folder)

# Plot some training samples
if show_sample:
  real_samples, labels = next(iter(train_loader))
  show_sample_data(real_samples, sample_size=16)

# Load checkpoint
start_epoch, loss_values = 0, None
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