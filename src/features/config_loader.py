import torch
import math

def load_hyperparameters(Hyperparameter):
  model_selector = Hyperparameter['model_selector']
  models = Hyperparameter['models']
  epochs = Hyperparameter['epochs']
  batch_size = Hyperparameter['batch_size']
  return model_selector, models, epochs, batch_size

def load_configuration(Configuration):
  device = Configuration['device']
  seed = Configuration['seed']
  save_sample_interval = Configuration['save_sample_interval']
  checkpoint_interval = Configuration['checkpoint_interval']
  training_mode = Configuration['training_mode']
  show_training_sample = Configuration['show_training_sample']
  load_checkpoint = Configuration['load_checkpoint']
  training = Configuration['training']
  world_size = Configuration['world_size']
  if device == 'cuda':
    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        world_size = available_gpus
  show_training_process = Configuration['show_training_process']
  calculate_FID_score = Configuration['calculate_FID_score']
  calculate_FID_interval = Configuration['calculate_FID_interval']
  show_training_evolution = Configuration['show_training_evolution']
  generate_data = Configuration['generate_data']
  sample_point_threshold = Configuration['sample_point_threshold']
  slope_threshold = math.tan(Configuration['slope_threshold'])
  epoch_threshold = Configuration['epoch_threshold']
  loss_threshold = Configuration['loss_threshold']
  
  return (device, save_sample_interval, checkpoint_interval, training_mode, 
    show_training_sample, load_checkpoint, training, world_size, 
    show_training_process, calculate_FID_score, calculate_FID_interval, 
    show_training_evolution, generate_data, sample_point_threshold, slope_threshold, 
    seed, epoch_threshold, loss_threshold)