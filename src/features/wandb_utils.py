import torch
import wandb

# Initialize wandb API
def init_wandb(project_name, Hyperparameter, Configuration, run_id):
  wandb_config={
          "epochs": Hyperparameter['epochs'],
          "batch_size": Hyperparameter['batch_size'],
          "seed": Configuration['seed'],
  }
  # Initialize wandb with or without run_id based on its presence
  if run_id:
    wandb_instant = wandb.init(
      project=project_name,
      config=wandb_config,
      group="DDP",
      resume="allow",
      id=run_id
    )
  else:
    wandb_instant = wandb.init(
      project=project_name,
      config=wandb_config,
      group="DDP",
      resume="allow"
    )
  
  # Disable some visualization if using wandb (sweep mode)
  if wandb.run and wandb.run.sweep_id is not None:
    print('test')
    Configuration.update({
      'show_training_process': False,
      'show_training_evolution': False,
      'show_sample': False,
      'generate_data': False
    })

    # Override local configuration with wandb config (sweep mode)
    sweep_wandb_config = wandb.config
    for key, value in sweep_wandb_config.items():
      # Update Hyperparameter and Configuration
      print(key, value)
      if key in ['epochs', 'batch_size', 'seed']:
        if key in Hyperparameter:
          Hyperparameter[key] = value
        elif key in Configuration:
          Configuration[key] = value
      # Update model and optimizer learning rate
      elif key.endswith('learning_rate'):
        model_name = key.replace('_learning_rate', '')
        if model_name in Hyperparameter['models']:
          Hyperparameter['models'][model_name]['learning_rate'] = value

  return wandb_instant