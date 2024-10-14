from features import get_data_loader, get_device, get_checkpoint, train_model, get_model, load_configuration, load_hyperparameters
from visualization import show_sample_data, generate_sample
from colorama import Fore, Style

import os
import yaml
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import wandb

def main():
  # Load the configuration file
  with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)
  Hyperparameter = config['Hyperparameter']
  Configuration = config['Configuration']

  # Start wandb logging
  if Configuration['log_wandb']:
    wandb_config={
          "architecture": "HQGAN",
          "epochs": Hyperparameter['epochs'],
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
      elif key.endswith('learning_rate'):
        model_name = key.replace('_learning_rate', '')
        if model_name in Hyperparameter['models']:
          Hyperparameter['models'][model_name]['learning_rate'] = value
      
      # Disable some visualization if using wandb sweep mode
      if wandb.run.sweep_id is not None:
        Configuration.update({
          'show_training_process': False,
          'show_training_evolution': False,
          'show_sample': False,
          'generate_data': False
        })
  else:
    print(Fore.YELLOW + "wandb logging is disabled." + Style.RESET_ALL)

  #Load Hyperparameters
  model_selector, models, epochs, batch_size = load_hyperparameters(Hyperparameter)

  # Load Configuration Settings
  (device, save_sample_interval, checkpoint_interval, training_mode, 
  show_training_sample, load_checkpoint, training, world_size, 
  show_training_process, calculate_FID_score, calculate_FID_interval, 
  show_training_evolution, generate_data, log_wandb) = load_configuration(Configuration)

  # Set up folders path
  script_dir = os.path.dirname(os.path.abspath(__file__))
  data_folder = os.path.join(script_dir, '..', 'data')
  checkpoint_folder = os.path.join(script_dir, 'checkpoints')

  # Create required folders
  os.makedirs(checkpoint_folder, exist_ok=True)
  os.makedirs(data_folder, exist_ok=True)

  # Use cuda if available
  device = get_device(device)

  # Load models
  model_list, optimizer_list = get_model(models, model_selector)

  # Load data
  train_loader = get_data_loader(batch_size=batch_size, data_folder=data_folder)

  # Plot some training samples
  if show_training_sample:
    real_samples, labels = next(iter(train_loader))
    show_sample_data(real_samples, title='Real Sample', sample_size=16)

  # Load checkpoint
  start_epoch, loss_values, fid_score = 0, None, []
  if load_checkpoint:
    start_epoch, loss_values, fid_score = get_checkpoint(checkpoint_folder=checkpoint_folder, 
                                              model_list=model_list,
                                              optimizer_list=optimizer_list,
                                              device=device)

  # Update wandb config with models and optimizers
  if log_wandb:
      wandb.config.update({
        "num_models": len(model_list),
        "models": model_list,
        "optimizer": optimizer_list,
      })

  # Train model using multiprocessing
  #training = False
  if training and epochs != start_epoch:
    mp.spawn(
          train_model,
          args=(world_size, device, epochs, train_loader, model_list, optimizer_list, checkpoint_folder, log_wandb, show_training_process, show_training_evolution, calculate_FID_score, calculate_FID_interval, save_sample_interval, start_epoch, checkpoint_interval, training_mode, loss_values, fid_score),
          nprocs=world_size,
          join=True
    )
  
  # Generate sample
  if generate_data:
    generated_sample = generate_sample(model_list[0], device, sample_size=16)
    show_sample_data(generated_sample, title='Generated Sample')

  # Keep the plot open
  plt.ion()
  plt.show(block=True)

if __name__ == '__main__':
    main()