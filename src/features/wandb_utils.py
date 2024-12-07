import wandb
from colorama import Fore, Style

def init_wandb(Hyperparameter, Configuration, run_id, log_wandb):
  project_name = Configuration['wandb']['project']
  entity_name = Configuration['wandb']['entity']
  
  wandb_config={
          "epochs": Hyperparameter['epochs'],
          "batch_size": Hyperparameter['batch_size'],
          "seed": Configuration['seed'],
  }
  
  # Initialize wandb
  wandb_instant = None
  if log_wandb:
    wandb_instant = wandb.init(
      project=project_name,
      entity=entity_name,
      config=wandb_config,
      resume="allow",
      id=run_id
    )
  else:
    wandb.init(mode="disabled")
    print(Fore.YELLOW + "wandb logging is disabled." + Style.RESET_ALL)
  
  # Disable some visualization if using wandb (sweep mode)
  if wandb.run and wandb.run.sweep_id is not None:
    if Configuration['load_checkpoint']:
      print(Fore.RED + "Load checkpoint is disabled in sweep mode." + Style.RESET_ALL)

    Configuration.update({
      'show_training_process': False,
      'show_training_evolution': False,
      'show_sample': False,
      'generate_data': False,
      #'load_checkpoint': False
    })

    # Override local configuration with wandb config (sweep mode)
    sweep_wandb_config = wandb.config
    learning_rate_multiplier = 1
    learning_rate = 1e-4
    for key, value in sweep_wandb_config.items():
      # Update Hyperparameter and Configuration
      print(key, value)
      if key in ['epochs', 'batch_size', 'seed', 'model_selector']:
        if key in Hyperparameter:
          Hyperparameter[key] = value
        elif key in Configuration:
          Configuration[key] = value
      # Update model and optimizer learning rate
      elif key == 'learning_rate':
        learning_rate = value
      elif key == 'learning_rate_multiplier':
        learning_rate_multiplier = value

    for model_name in Hyperparameter['models']:
      if model_name.endswith('generator'):
        Hyperparameter['models'][model_name]['learning_rate'] = learning_rate * learning_rate_multiplier
      elif model_name.endswith('discriminator'):
        Hyperparameter['models'][model_name]['learning_rate'] = learning_rate

  return wandb_instant