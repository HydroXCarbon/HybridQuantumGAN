from features import get_data_loader, get_device, get_checkpoint, train_model, get_model, load_configuration, load_hyperparameters, init_wandb, load_run_id
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

  wandb_instant = None
  run_id = None

  # Set up folders path
  script_dir = os.path.dirname(os.path.abspath(__file__))
  data_folder = os.path.join(script_dir, '..', 'data')
  checkpoint_folder = os.path.join(script_dir, 'checkpoints')
  checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint.pth')

  # Create required folders
  os.makedirs(checkpoint_folder, exist_ok=True)
  os.makedirs(data_folder, exist_ok=True)

  # Load run_id from checkpoint
  if Configuration['load_checkpoint']:
    run_id = load_run_id(checkpoint_path)
  if run_id:
    print(Fore.GREEN + "Previous Run ID:" + Style.RESET_ALL + f" {run_id}")

  # Start wandb logging
  wandb_instant = init_wandb(Hyperparameter, Configuration, run_id, Configuration['log_wandb'])

  #Load Hyperparameters
  model_selector, models, epochs, batch_size = load_hyperparameters(Hyperparameter)

  # Load Configuration Settings
  (device, save_sample_interval, checkpoint_interval, training_mode, 
  show_training_sample, load_checkpoint, training, world_size, 
  show_training_process, calculate_FID_score, calculate_FID_interval, 
  show_training_evolution, generate_data, divergent_threshold) = load_configuration(Configuration)

  # Use cuda if available
  device = get_device(device)

  # Load models
  model_list, optimizer_list = get_model(models, model_selector)

  # Load checkpoint
  start_epoch, loss_values, fid_score = 0, None, []
  if load_checkpoint: 
    start_epoch, loss_values, fid_score, batch_size = get_checkpoint(checkpoint_path=checkpoint_path, 
                                                        model_list=model_list,
                                                        optimizer_list=optimizer_list)
    
  # Load data
  train_loader = get_data_loader(batch_size=batch_size, data_folder=data_folder)

  # Plot some training samples
  if show_training_sample:
    real_samples, labels = next(iter(train_loader))
    show_sample_data(real_samples, title='Real Sample', sample_size=16)

  # Update wandb config with models and optimizers
  wandb.config.update({
    "num_models": len(model_list),
    "models": model_list,
    "optimizer": optimizer_list,
  })

  # Train model using multiprocessing
  if training and epochs != start_epoch:
    mp.spawn(
          train_model,
          args=(world_size,
                device, 
                epochs,
                train_loader, 
                model_list, 
                optimizer_list, 
                checkpoint_path, 
                show_training_process, 
                show_training_evolution, 
                calculate_FID_score, 
                calculate_FID_interval, 
                wandb_instant,
                divergent_threshold,
                save_sample_interval, 
                start_epoch, 
                checkpoint_interval, 
                training_mode, 
                loss_values, 
                fid_score),
          nprocs=world_size,
          join=True
    )
  
  # Finish wandb logging
  if wandb_instant:
    wandb_instant.finish()
  
  # Generate sample
  if generate_data:
    generated_sample = generate_sample(model_list[0], device, sample_size=16)
    show_sample_data(generated_sample, title='Generated Sample')

  # Keep the plot open
  plt.ion()
  plt.show(block=True)

if __name__ == '__main__':
    main()