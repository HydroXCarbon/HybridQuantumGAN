from visualization import PlotTrainingProgress, PlotEvolution
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from colorama import Fore, Style
from .DDP_utils import setup, cleanup
from .module_utils import move_model_and_optimizer_to_device, train_discriminator, train_generator, denormalize_and_convert_uint8

import torch
import os

class LossValues:
  def __init__(self):
    self.generator_loss_values = {}
    self.discriminator_loss_values = {}
    self.entropy_values = {}

def train_model(rank, 
                world_size,
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
                save_sample_interval=1,
                start_epoch=0,
                checkpoint_interval=5,
                training_mode='alternating',
                loss_values=None,
                fid_score=[]):
  from features import save_checkpoint
  from visualization import generate_sample

  # Set up DistributedDataParallel
  setup(rank, world_size, device)

  if loss_values is None:
    loss_values = LossValues()

  # Fix bug Optimizer.name disappears when using DDP
  for optimizer, model in zip(optimizer_list, model_list):
    optimizer.name = 'optimizer_' + model.name

  # Initialize FID metric
  if calculate_FID_score:
    fid = FrechetInceptionDistance(feature=64).to(rank)

  parent_process_id = os.getppid()
  process_id = os.getpid()

  # Setup models and optimizers
  generator, discriminator_list, optimizer_generator, optimizer_discriminator_list = move_model_and_optimizer_to_device(model_list, optimizer_list, rank, device)
  print(f"Process {rank} ({parent_process_id})({process_id}): is running on {next(generator.parameters()).device}")

  num_discriminators = len(discriminator_list)
  total_batches = len(train_loader)
  batch_size = train_loader.batch_size
  generated_samples_list = []

  # Barrier
  torch.distributed.barrier()

  # Training loop
  print(f"Process {rank} ({parent_process_id})({process_id}): " + Fore.GREEN + "Start training: " + Style.RESET_ALL + f'Epoch {start_epoch}')

  # Create instance for plotting
  plot_progress = PlotTrainingProgress()
  if show_training_evolution:
    plot_evolution = PlotEvolution(epochs=epochs-start_epoch)

  # Initialize progress bar
  progress_bar_epoch = tqdm(total=epochs, desc=f"Process {rank} ({parent_process_id})({process_id}): Model Progress", unit="epoch", leave=True, position=rank, initial=start_epoch)
  progress_bar_batch = tqdm(total=total_batches, desc=f"Process {rank} ({parent_process_id})({process_id}): Training Epoch {start_epoch}", unit="batch", leave=False, position=(rank*2)+world_size)

  # Training loop (Epoch)
  for epoch_i, epoch in enumerate(range(start_epoch, epochs)):
    # Clear progress bar at the beginning of each epoch
    progress_bar_batch.reset()
    progress_bar_batch.set_description(f"Process {rank} ({parent_process_id})({process_id}): Training Epoch {epoch}")
    
    # Clear FID metrics at the beginning of each epoch
    if calculate_FID_score:
      fid.reset()
      
    # Training loop (Batch)
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
        
        if wandb_instant:
          wandb_instant.log({f"{discriminator.module.name}_loss": loss_discriminator})

      # Training the generator
      loss_generator = train_generator(generator, discriminator_list, optimizer_generator, latent_space_samples, real_samples_labels, training_mode, epoch, num_discriminators)
          
      if generator.module.name not in loss_values.generator_loss_values:
        loss_values.generator_loss_values[generator.module.name] = []
      loss_values.generator_loss_values[generator.module.name].append(loss_generator)
      
      if wandb_instant:
        wandb_instant.log({f"{generator.module.name}_loss": loss_generator})

      # Update the progress bar
      progress_bar_batch.update()

      # Accumulate FID (real and generated samples) for this batch
      if calculate_FID_score and epoch % calculate_FID_interval == 0:
        # Denormalize and convert real and generated samples to uint8
        real_samples_uint8 = denormalize_and_convert_uint8(real_samples).repeat(1, 3, 1, 1)
        generated_samples_uint8 = denormalize_and_convert_uint8(generated_samples).repeat(1, 3, 1, 1)

        fid.update(real_samples_uint8, real=True)
        fid.update(generated_samples_uint8, real=False)

    # Calculate FID score
    if calculate_FID_score and epoch % calculate_FID_interval == 0:
        fid_value = fid.compute().cpu().detach().numpy().item()
        fid_score.append([fid_value, epoch])
        if wandb_instant:
          wandb_instant.log({"FID": fid_value, "Epoch": epoch})

    # Update the progress bar
    progress_bar_epoch.update()

    if rank == 0:
      
      # Plot progress
      if show_training_process:
        plot_progress.plot(epoch, loss_values, fid_score)

      # Save sample data at the specified interval
      if epoch % save_sample_interval == 0:
        generated_samples = generate_sample(generator, device, 1)
        generated_samples_list.append(generated_samples)

      # Save checkpoint at the specified interval
      if epoch % checkpoint_interval == 0 :
        save_checkpoint(epoch, batch_size, checkpoint_path, model_list, optimizer_list, loss_values, fid_score, wandb_instant)    

      # Plot the evolution of the generator
      if show_training_evolution:
        plot_evolution.plot(generated_samples_list[-1], epoch, epoch_i)

  # Close the progress bar
  progress_bar_batch.close()
  progress_bar_epoch.close()

  # Finish training
  print(f"Process {rank} ({parent_process_id})({process_id}):" + Fore.GREEN + 'Training finished' + Style.RESET_ALL)

  if rank == 0:

    # Save final checkpoint
    save_checkpoint(epochs, batch_size, checkpoint_path, model_list, optimizer_list, loss_values, fid_score, wandb_instant, finish=True)

  cleanup()