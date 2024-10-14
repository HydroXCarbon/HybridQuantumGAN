from visualization import PlotTrainingProgress, PlotEvolution
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics.image.fid import FrechetInceptionDistance

from tqdm import tqdm
from colorama import Fore, Style

import torch
import os
import torch.distributed as dist
import wandb

class LossValues:
  def __init__(self):
    self.generator_loss_values = {}
    self.discriminator_loss_values = {}
    self.entropy_values = {}

def setup(rank, world_size, device):
    # Set up distributed training
    backend = 'gloo' if device.type == 'cpu' else 'nccl'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_discriminator(discriminator, optimizer_discriminator, all_samples, all_samples_labels, retain_graph):
  discriminator.zero_grad()
  optimizer_discriminator.zero_grad()
  output_discriminator = discriminator(all_samples)
  loss_discriminator = discriminator.module.loss_function(output_discriminator, all_samples_labels)
  loss_discriminator.backward(retain_graph=retain_graph)
  optimizer_discriminator.step()
  return loss_discriminator.cpu().detach().numpy()

def train_generator(generator, discriminator_list, optimizer_generator, latent_space_samples, real_samples_labels, training_mode, epoch, num_discriminators):
  generator.zero_grad()
  generated_samples = generator(latent_space_samples)
  
  if training_mode == 'combined':
    combined_output = torch.zeros_like(discriminator_list[0](generated_samples))
    for discriminator in discriminator_list:
      combined_output += discriminator(generated_samples)
    combined_output /= len(discriminator_list)
    loss_generator = generator.module.loss_function(combined_output, real_samples_labels)
  elif training_mode == 'alternating':
    discriminator = discriminator_list[epoch % num_discriminators]
    output_discriminator_generated = discriminator(generated_samples)
    loss_generator = generator.module.loss_function(output_discriminator_generated, real_samples_labels)
  elif training_mode == 'continuous':
    for discriminator in discriminator_list:
      output_discriminator_generated = discriminator(generated_samples)
      loss_generator = generator.module.loss_function(output_discriminator_generated, real_samples_labels)
  else:
    raise ValueError(f"Training mode {training_mode} not supported")

  loss_generator.backward()
  optimizer_generator.step()
  return loss_generator.cpu().detach().numpy()

def denormalize_and_convert_uint8(images):
  images = (images * 0.5 + 0.5) * 255.0  
  images = images.clamp(0, 255) 
  images = images.to(torch.uint8) 
  return images

def train_model(rank, 
                world_size,
                device, 
                epochs, 
                train_loader, 
                model_list,
                optimizer_list, 
                checkpoint_folder, 
                log_wandb,
                show_training_process,
                show_training_evolution,
                calculate_FID_score,
                calculate_FID_interval,
                save_sample_interval=1,
                start_epoch=0,
                checkpoint_interval=5,
                training_mode='alternating',
                loss_values=None):
  from features import save_checkpoint
  from visualization import generate_sample

  # Initialize DistributedDataParallel
  setup(rank, world_size, device)

  if loss_values is None:
    loss_values = LossValues()

  # Fix bug Optimizer.name disappears
  for optimizer, model in zip(optimizer_list, model_list):
    optimizer.name = 'optimizer_' + model.name

  # Initialize FID metric
  if calculate_FID_score:
    fid = FrechetInceptionDistance(feature=64).to(device)

  # Setup models and optimizers
  generator, discriminator_list = model_list[0], model_list[1:]
  optimizer_generator, optimizer_discriminator_list = optimizer_list[0], optimizer_list[1:]

  # Move models to device and wrap with DistributedDataParallel
  generator = generator.to(rank)
  generator = DDP(generator, device_ids=[rank])

  for i in range(len(discriminator_list)):
    discriminator_list[i] = discriminator_list[i].to(rank)
    discriminator_list[i] = DDP(discriminator_list[i], device_ids=[rank])

  num_discriminators = len(discriminator_list)
  total_batches = len(train_loader)
  batch_size = train_loader.batch_size
  generated_samples_list = []
  fid_score = []

  # Create instance for plotting
  if rank == 0:
    # Training loop
    print(Fore.GREEN + "Start training: " + Style.RESET_ALL + f'Epoch {start_epoch}')

    plot_progress = PlotTrainingProgress()
    if show_training_evolution:
      plot_evolution = PlotEvolution(epochs=epochs-start_epoch)

  # Initialize progress bar
  progress_bar_epoch = tqdm(total=epochs-start_epoch, desc=f"Process {rank}: Model Progress", unit="epoch", leave=True, position=rank*2)
  progress_bar_batch = tqdm(total=total_batches, desc=f"Process {rank}: Training Epoch {start_epoch}", unit="batch", leave=False, position=(rank*2)+1)

  for epoch_i, epoch in enumerate(range(start_epoch, epochs)):
    # Clear progress bar at the beginning of each epoch
    progress_bar_batch.reset()
    # Clear FID metrics at the beginning of each epoch
    if calculate_FID_score:
      fid.reset()

    for batch_i, (real_samples, mnist_labels) in enumerate(train_loader):
      real_samples = real_samples.to(rank)
      real_samples_labels = torch.ones((batch_size, 1)).to(rank)
      latent_space_samples = torch.randn((batch_size, 100)).to(rank)
      generated_samples = generator(latent_space_samples)
      generated_samples_labels = torch.zeros((batch_size, 1)).to(rank)
      all_samples = torch.cat((real_samples, generated_samples))
      all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

      # Training the discriminators
      for i, (discriminator, optimizer_discriminator) in enumerate(zip(discriminator_list, optimizer_discriminator_list)):
        retain_graph = i < num_discriminators - 1
        loss_discriminator = train_discriminator(discriminator, optimizer_discriminator, all_samples, all_samples_labels, retain_graph)

        if discriminator.module.name not in loss_values.discriminator_loss_values:
          loss_values.discriminator_loss_values[discriminator.module.name] = []
        loss_values.discriminator_loss_values[discriminator.module.name].append(loss_discriminator)
        
        if log_wandb:
          wandb.log({f"{discriminator.module.name}_loss": loss_discriminator})

      # Training the generator
      loss_generator = train_generator(generator, discriminator_list, optimizer_generator, latent_space_samples, real_samples_labels, training_mode, epoch, num_discriminators)
          
      if generator.module.name not in loss_values.generator_loss_values:
        loss_values.generator_loss_values[generator.module.name] = []
      loss_values.generator_loss_values[generator.module.name].append(loss_generator)
      
      if log_wandb:
        wandb.log({f"{generator.module.name}_loss": loss_generator})

      # Update the progress bar
      progress_bar_batch.update()

      # Denormalize and convert real and generated samples to uint8
      real_samples_uint8 = denormalize_and_convert_uint8(real_samples).repeat(1, 3, 1, 1)
      generated_samples_uint8 = denormalize_and_convert_uint8(generated_samples).repeat(1, 3, 1, 1)

      # Accumulate FID (real and generated samples) for this batch
      if calculate_FID_score and epoch % calculate_FID_interval == 0:
        fid.update(real_samples_uint8, real=True)
        fid.update(generated_samples_uint8, real=False)

    # Calculate FID score
    if calculate_FID_score:
      if epoch % calculate_FID_interval == 0:
        fid_score.append(fid.compute().cpu().detach().numpy())
      else:
        fid_score.append(None)
      if log_wandb:
          wandb.log({"FID": fid_score[-1]})

    # Update the progress bar
    progress_bar_epoch.update()

    if rank == 0:
      
      # Plot progress
      if show_training_process:
        plot_progress.plot(epoch, loss_values, fid_score)

      # Save sample data at the specified interval
      if (epoch + 1) % save_sample_interval == 0:
        generated_samples = generate_sample(generator, device, 1)
        generated_samples_list.append(generated_samples)

      # Save checkpoint at the specified interval
      if (epoch + 1) % checkpoint_interval == 0 :
        save_checkpoint(epoch, checkpoint_folder, model_list, optimizer_list, loss_values)    

      # Plot the evolution of the generator
      if show_training_evolution:
        plot_evolution.plot(generated_samples_list[-1], epoch, epoch_i)

  # Close the progress bar
  progress_bar_batch.close()
  progress_bar_epoch.close()

  if rank == 0:

    # Finish training
    print(Fore.GREEN + 'Training finished' + Style.RESET_ALL)

    # Finish the wandb run
    if log_wandb:
      wandb.finish()

    # Save final checkpoint
    save_checkpoint(epochs, checkpoint_folder, model_list, optimizer_list, loss_values)

  cleanup()