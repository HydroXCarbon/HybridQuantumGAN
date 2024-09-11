from visualization import PlotTrainingProgress, PlotEvolution
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from colorama import Fore, Style

import torch
import torch.distributed as dist
import wandb

class LossValues:
  def __init__(self):
    self.generator_loss_values = {}
    self.discriminator_loss_values = {}
    self.entropy_values = {}

def train_discriminator(discriminator, optimizer_discriminator, all_samples, all_samples_labels, retain_graph):
  discriminator.zero_grad()
  optimizer_discriminator.zero_grad()
  output_discriminator = discriminator(all_samples)
  loss_discriminator = discriminator.module.loss_function(output_discriminator, all_samples_labels)
  loss_discriminator.backward(retain_graph=retain_graph)
  optimizer_discriminator.step()
  return loss_discriminator

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
  return loss_generator

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
                save_sample_interval=1,
                start_epoch=0,
                checkpoint_interval=5,
                training_mode='alternating',
                loss_values=None):
  from features import save_checkpoint
  from visualization import generate_sample

  # Initialize DistributedDataParallel
  dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)

  if loss_values is None:
    loss_values = LossValues()

  # Fix bug Optimizer.name disappears
  for optimizer, model in zip(optimizer_list, model_list):
    optimizer.name = 'optimizer_' + model.name

  # Setup models
  generator = model_list[0]
  discriminator_list = [discriminator for discriminator in model_list[1:]]

  # Wrap models with DistributedDataParallel
  generator = DDP(generator)
  discriminator_list = [DDP(discriminator) for discriminator in discriminator_list]

  optimizer_generator = optimizer_list[0]
  optimizer_discriminator_list = optimizer_list[1:]

  num_discriminators = len(discriminator_list)
  batch_size = train_loader.batch_size
  generated_samples_list = []
  total_batches = len(train_loader)
    
  # Create instance for plotting
  if rank == 0:
    # Training loop
    print(Fore.GREEN + "Start training: " + Style.RESET_ALL + f'Epoch {start_epoch}')
    
    plot_progress = PlotTrainingProgress()
    if show_training_evolution:
      plot_evolution = PlotEvolution(epochs=epochs-start_epoch)

  # Initialize progress bar
  progress_bar_epoch = tqdm(total=epochs-start_epoch, desc=f"Model Progress", unit="epoch", leave=True)
  
  for epoch_i, epoch in enumerate(range(start_epoch, epochs)):

    # Initialize batch progress bar
    progress_bar_batch = tqdm(total=total_batches, desc=f"Training Epoch {epoch}", unit="batch", leave=False)
      
    for batch_i, (real_samples, mnist_labels) in enumerate(train_loader):
      real_samples = real_samples.to(device=device)
      real_samples_labels = torch.ones((batch_size, 1)).to(device=device)
      latent_space_samples = torch.randn((batch_size, 100)).to(device=device)
      generated_samples = generator(latent_space_samples)
      generated_samples_labels = torch.zeros((batch_size, 1)).to(device=device)
      all_samples = torch.cat((real_samples, generated_samples))
      all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

      # Training the discriminators
      for i, (discriminator, optimizer_discriminator) in enumerate(zip(discriminator_list, optimizer_discriminator_list)):
        retain_graph = i < num_discriminators - 1
        loss_discriminator = train_discriminator(discriminator, optimizer_discriminator, all_samples, all_samples_labels, retain_graph)
        loss_discriminator = loss_discriminator.cpu().detach().numpy()
        
        if discriminator.module.name not in loss_values.discriminator_loss_values:
          loss_values.discriminator_loss_values[discriminator.module.name] = []
        loss_values.discriminator_loss_values[discriminator.module.name].append(loss_discriminator)
        
        if log_wandb:
          wandb.log({f"{discriminator.module.name}_loss": loss_discriminator})

      # Training the generator
      loss_generator = train_generator(generator, discriminator_list, optimizer_generator, latent_space_samples, real_samples_labels, training_mode, epoch, num_discriminators)
      generator_loss = loss_generator.cpu().detach().numpy()
          
      if generator.module.name not in loss_values.generator_loss_values:
        loss_values.generator_loss_values[generator.module.name] = []
      loss_values.generator_loss_values[generator.module.name].append(generator_loss)
      
      if log_wandb:
        wandb.log({f"{generator.module.name}_loss": generator_loss})

      # Update the progress bar
      progress_bar_batch.update()

    # Update and Close the progress bar
    progress_bar_data = {'loss_G': f"{loss_values.generator_loss_values[generator.module.name][-1]:.5f}"}
    for i, discriminator in enumerate(loss_values.discriminator_loss_values.keys()):
      progress_bar_data[f'loss d_{i}'] = f"{loss_values.discriminator_loss_values[discriminator][-1]:.5f}"
    progress_bar_batch.set_postfix(progress_bar_data)
    progress_bar_batch.close()
    progress_bar_epoch.update()

    if rank == 0:
      
      # Plot progress
      if show_training_process:
        plot_progress.plot(epoch, loss_values)

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
  progress_bar_epoch.close()

  if rank == 0:

    # Finish training
    print(Fore.GREEN + 'Training finished' + Style.RESET_ALL)

    # Finish the wandb run
    if log_wandb:
      wandb.finish()

    # Save final checkpoint
    save_checkpoint(epochs, checkpoint_folder, model_list, optimizer_list, loss_values)
