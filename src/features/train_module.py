from visualization import PlotTrainingProgress, PlotEvolution, show_sample_data
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm
from features import get_data_loader
from colorama import Fore, Style
from .DDP_utils import setup, cleanup
from .module_utils import move_model_and_optimizer_to_device, train_discriminator, train_generator, denormalize_and_convert_uint8

import torch.distributed as dist
import torch
import os

class LossValues:
  def __init__(self):
    self.generator_loss_values = {}
    self.discriminator_loss_values = {}
    self.entropy_values = {}

def train_model(rank, 
                world_size,
                seed,
                device, 
                epochs, 
                batch_size,
                model_list,
                optimizer_list, 
                checkpoint_path, 
                data_folder,
                show_training_process,
                show_training_evolution,
                show_training_sample,
                calculate_FID_score,
                calculate_FID_interval,
                wandb_instant,
                sample_point_threshold,
                epoch_threshold,
                loss_threshold,
                slope_threshold,
                save_sample_interval=1,
                start_epoch=0,
                checkpoint_interval=5,
                training_mode='alternating',
                loss_values=None,
                fid_score=[]):
  from features import save_checkpoint
  from visualization import generate_sample

  # Set up DistributedDataParallel
  if device != 'cpu':
    device = torch.device(f"cuda:{rank}")
  setup(rank, world_size, device)

  # Set unique seed for each process
  if seed is not None:
    torch.manual_seed(seed + rank)

  if loss_values is None:
    loss_values = LossValues()

  # Fix bug Optimizer.name disappears when using DDP
  for optimizer, model in zip(optimizer_list, model_list):
    optimizer.name = 'optimizer_' + model.name

  # Initialize FID metric
  if calculate_FID_score:
    fid = FrechetInceptionDistance(feature=64).to(rank)

  # Get PID
  process_id = os.getpid()
  parent_process_id = os.getppid()
  grandparent_process_id = None
  try:
    with open(f"/proc/{parent_process_id}/stat") as f:
        stat_info = f.read().split()
        grandparent_process_id = stat_info[3]
  except FileNotFoundError:
      print("Unable to retrieve grandparent process ID. The process may have terminated.")
  
  # Load data
  train_loader = get_data_loader(batch_size=batch_size, data_folder=data_folder, rank=rank, world_size=world_size)

  # Plot some training samples
  if show_training_sample:
    real_samples, labels = next(iter(train_loader))
    show_sample_data(real_samples, title='Real Sample', sample_size=16)

  # Setup models and optimizers
  generator, discriminator_list, optimizer_generator, optimizer_discriminator_list = move_model_and_optimizer_to_device(model_list, optimizer_list, rank, device)
  print(f"Process {rank} ({grandparent_process_id})({parent_process_id})({process_id}): is running on {next(generator.parameters()).device}")

  num_discriminators = len(discriminator_list)
  total_batches = len(train_loader)
  steps_per_epoch = total_batches * batch_size
  generated_samples_list = []

  # Set up a global flag for termination
  divergence_flag = torch.tensor([0], dtype=torch.int).to(device)

  # Barrier
  dist.barrier()

  # Training loop
  print(f"Process {rank} ({grandparent_process_id})({parent_process_id})({process_id}): " + Fore.GREEN + "Start training: " + Style.RESET_ALL + f'Epoch {start_epoch}')

  # Create instance for plotting
  plot_progress = PlotTrainingProgress()
  if show_training_evolution:
    plot_evolution = PlotEvolution(epochs=epochs-start_epoch)

  # Initialize progress bar
  progress_bar_epoch = tqdm(total=epochs, desc=f"Process {rank} ({grandparent_process_id})({parent_process_id})({process_id}): Model Progress", unit="epoch", leave=True, position=rank, initial=start_epoch)
  progress_bar_batch = tqdm(total=total_batches, desc=f"Process {rank} ({grandparent_process_id})({parent_process_id})({process_id}): Training Epoch {start_epoch}", unit="batch", leave=False, position=(rank*2)+world_size)

  # Training loop (Epoch)
  for epoch_i, epoch in enumerate(range(start_epoch, epochs)):
    # Clear progress bar at the beginning of each epoch
    progress_bar_batch.reset()
    progress_bar_batch.set_description(f"Process {rank} ({grandparent_process_id})({parent_process_id})({process_id}): Training Epoch {epoch}")
    
    # Set epoch for train_loader
    train_loader.sampler.set_epoch(epoch)

    # Clear FID metrics at the beginning of each epoch
    if calculate_FID_score and epoch % calculate_FID_interval == 0:
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
        loss_discriminator_all = [torch.zeros_like(loss_discriminator) for _ in range(world_size)]

        dist.all_gather(loss_discriminator_all, loss_discriminator)
        loss_discriminator_all = sum([d.item() for d in loss_discriminator_all]) / len(loss_discriminator_all)

        if discriminator.module.name not in loss_values.discriminator_loss_values:
          loss_values.discriminator_loss_values[discriminator.module.name] = []
        loss_values.discriminator_loss_values[discriminator.module.name].append(loss_discriminator_all)
        
        if wandb_instant and rank ==0:
          wandb_instant.log({f"{discriminator.module.name}_loss": loss_discriminator_all, 'batch step': epoch_i * steps_per_epoch + batch_i * batch_size})

      # Training the generator
      loss_generator = train_generator(generator, discriminator_list, optimizer_generator, latent_space_samples, real_samples_labels, training_mode, epoch, num_discriminators)
      loss_generator_all = [torch.zeros_like(loss_generator) for _ in range(world_size)]

      dist.all_gather(loss_generator_all, loss_generator)
      loss_generator_all = sum([d.item() for d in loss_generator_all]) / len(loss_generator_all)
      
      if generator.module.name not in loss_values.generator_loss_values:
        loss_values.generator_loss_values[generator.module.name] = []
      loss_values.generator_loss_values[generator.module.name].append(loss_generator_all)
      
      if wandb_instant and rank ==0:
        wandb_instant.log({f"{generator.module.name}_loss": loss_generator_all, 'batch step': epoch_i * steps_per_epoch + batch_i * batch_size})

      # Update the progress bar
      progress_bar_batch.update()

      # Accumulate FID (real and generated samples) for this batch
      if calculate_FID_score and epoch % calculate_FID_interval == 0:
        # Denormalize and convert real and generated samples to uint8
        real_samples_uint8 = denormalize_and_convert_uint8(real_samples).repeat(1, 3, 1, 1)
        generated_samples_uint8 = denormalize_and_convert_uint8(generated_samples).repeat(1, 3, 1, 1)

        # Update FID metric with the combined samples
        fid.update(real_samples_uint8, real=True)
        fid.update(generated_samples_uint8, real=False)
    
    # Calculate FID score
    if calculate_FID_score and epoch % calculate_FID_interval == 0:
      fid_value_tensor = torch.tensor([fid.compute().item()], device=device)

      fid_value_all = [torch.zeros_like(fid_value_tensor) for _ in range(world_size)]
      dist.all_gather(fid_value_all, fid_value_tensor)
      
      fid_value_all = sum([d.item() for d in fid_value_all]) / len(fid_value_all)
      fid_score.append([fid_value_all, epoch])

      if wandb_instant and rank == 0:
        wandb_instant.log({"FID": fid_value_all, "Epoch": epoch})

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

      # Check divergent
      if len(fid_score) >= sample_point_threshold:
        recent_scores = [score[0] for score in fid_score[-sample_point_threshold:]]
        slopes = [
          recent_scores[i] - recent_scores[i - 1]
          for i in range(1, len(recent_scores))
        ]
        average_slope = sum(slopes) / len(slopes)

        divergent_counter = sum(
          1 for i in range(1, len(recent_scores)) if recent_scores[i] > recent_scores[i - 1]
        )
        if (divergent_counter == sample_point_threshold - 1 and average_slope > slope_threshold) or (fid_score[0] < fid_score[-1]):
          divergence_flag.fill_(1)

      if epoch > epoch_threshold:
        for discriminator in discriminator_list:
          if loss_values.discriminator_loss_values[discriminator.module.name][-1] < loss_threshold:
            divergence_flag.fill_(1)
    
    # Barrier
    dist.barrier()

    dist.all_reduce(divergence_flag, op=dist.ReduceOp.SUM)

    if divergence_flag.item() > 0:
      print(f"Process {rank} ({grandparent_process_id})({parent_process_id})({process_id}): Divergence detected. Stopping training.")
      return
        
  # Close the progress bar
  progress_bar_batch.close()
  progress_bar_epoch.close()

  # Finish training
  print(f"Process {rank} ({grandparent_process_id})({parent_process_id})({process_id}):" + Fore.GREEN + 'Training finished' + Style.RESET_ALL)

  if rank == 0:

    # Save final checkpoint
    save_checkpoint(epochs, batch_size, checkpoint_path, model_list, optimizer_list, loss_values, fid_score, wandb_instant, finish=True)

  cleanup()