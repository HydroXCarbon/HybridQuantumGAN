from features import get_data_loader, get_device, get_checkpoint, train_model, get_model
from visualization import show_sample_data, generate_sample

import torch
import os
import yaml
import matplotlib.pyplot as plt
import wandb

# Load the configuration file
with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
Hyperparameter = config['Hyperparameter']
Configuration = config['Configuration']

# Start wandb logging
if Configuration['log_wandb']:
  wandb_config={
        "architecture": "HQGAN",
        "epochs": Hyperparameter['num_epochs'],
        "batch_size": Hyperparameter['batch_size'],
        "training_mode": Configuration['training_mode'],
        "seed": Configuration['seed'],
  }
  wandb.init(
    project="HybridQuantumGAN",
    config=wandb_config
  )
  # Override local configuration with wandb config (sweep mode)
  wandb_config = wandb.config
  for key, value in wandb_config.items():
    # Update Hyperparameter and Configuration
    if key in ['epochs', 'batch_size', 'training_mode']:
      if key in Hyperparameter:
        Hyperparameter[key] = value
      elif key in Configuration:
        Configuration[key] = value
    # Update model and optimizer learning rate
    if  key.endswith('learning_rate'):
      model_name = key.replace('_learning_rate', '')
      if model_name in Hyperparameter['models']:
        Hyperparameter['models'][model_name]['learning_rate'] = value
else:
  print("wandb logging is disabled.")

# Hyperparameters
model_selector = Hyperparameter['model_selector']
models = Hyperparameter['models']
num_epochs = Hyperparameter['num_epochs']
batch_size = Hyperparameter['batch_size']

# Configuration Settings
seed = Configuration['seed']
save_sample_interval = Configuration['save_sample_interval']
checkpoint_interval = Configuration['checkpoint_interval']
training_mode = Configuration['training_mode']
show_sample = Configuration['show_sample']
load_checkpoint = Configuration['load_checkpoint']
training = Configuration['training']
show_training_process = Configuration['show_training_process']
show_training_evolution = Configuration['show_training_evolution']
generate_data = Configuration['generate_data']
log_wandb = Configuration['log_wandb']

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
  show_sample_data(real_samples, title='Real Sample', sample_size=16)

# Load checkpoint
start_epoch, loss_values = 0, None
if load_checkpoint:
  start_epoch, loss_values = get_checkpoint(checkpoint_folder=checkpoint_folder, 
                                            model_list=model_list,
                                            optimizer_list=optimizer_list)

# Update wandb config with models and optimizers
if log_wandb:
    wandb.config.update({
      "num_models": len(model_list),
      "models": model_list,
      "optimizer": optimizer_list,
    })

# Train model
if training and num_epochs != start_epoch:
  train_model(device=device, 
              num_epochs=num_epochs,
              train_loader=train_loader,
              model_list=model_list,
              optimizer_list=optimizer_list,
              checkpoint_folder=checkpoint_folder,
              start_epoch=start_epoch,
              loss_values=loss_values,
              checkpoint_interval=checkpoint_interval,
              save_sample_interval=save_sample_interval,
              training_mode=training_mode,
              log_wandb=log_wandb,
              show_training_process=show_training_process,
              show_training_evolution=show_training_evolution) 

# Generate sample
if generate_data:
  generated_sample = generate_sample(model_list[0], device, sample_size=16)
  show_sample_data(generated_sample, title='Generated Sample')

# Wait for user to close the plot
plt.ioff()
plt.show()