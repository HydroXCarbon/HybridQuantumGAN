from visualization import PlotTrainingProgress

import os
import torch
import matplotlib.pyplot as plt

class LossValues:
  def __init__(self):
    self.generator_loss_values = {}
    self.discriminator_loss_values = {}
    self.entropy_values = {}

def train_model(device, 
                num_epochs, 
                train_loader, 
                model_list,
                optimizer_list,
                loss_function, 
                checkpoint_folder, 
                start_epoch=0,
                checkpoint_interval=5,
                training_mode='alternating',
                loss_values=None):

  if loss_values is None:
    loss_values = LossValues()
  print(f'Start training at epoch {start_epoch}')

  # Setup models
  generator = model_list[0]
  optimizer_generator = optimizer_list[0]
  discriminator_list = model_list[1:]
  optimizer_discriminator_list = optimizer_list[1:]

  # Create thread for plotting (due to long time to plot)
  plot_progress = PlotTrainingProgress()

  for epoch in range(start_epoch, num_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
      # Data for training the discriminator
      real_samples = real_samples.to(device=device)
      real_samples_labels = torch.ones((train_loader.batch_size, 1)).to(device=device)
      latent_space_samples = torch.randn((train_loader.batch_size, 100)).to(device=device)
      generated_samples = generator(latent_space_samples)
      generated_samples_labels = torch.zeros((train_loader.batch_size, 1)).to(device=device)
      all_samples = torch.cat((real_samples, generated_samples))
      all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))

      # Training the discriminator
      for discriminator, optimizer_discriminator in zip(discriminator_list, optimizer_discriminator_list):
        discriminator.zero_grad()
        output_discriminator = discriminator(all_samples)
        loss_discriminator = loss_function(output_discriminator, all_samples_labels)
        loss_discriminator.backward(retain_graph=True)
        optimizer_discriminator.step()

        # Store discriminator loss for plotting
        if discriminator.name not in loss_values.discriminator_loss_values:
            loss_values.discriminator_loss_values[discriminator.name] = []
        loss_values.discriminator_loss_values[discriminator.name].append(loss_discriminator.cpu().detach().numpy())

      # Data for training the generator
      latent_space_samples = torch.randn((train_loader.batch_size, 100)).to(device=device)

      # Training the generator
      generator.zero_grad()
      generated_samples = generator(latent_space_samples)
      if training_mode == 'combined':
        for discriminator in discriminator_list:
          output_discriminator_generated = discriminator(generated_samples)
          if combined_output is None:
              combined_output = output_discriminator_generated
          else:
              combined_output += output_discriminator_generated
        combined_output /= len(discriminator_list)
        loss_generator = loss_function(combined_output, real_samples_labels)
        loss_generator.backward()
        optimizer_generator.step()
      elif training_mode == 'alternating':
        for discriminator in discriminator_list:
          output_discriminator_generated = discriminator(generated_samples)
          loss_generator = loss_function(output_discriminator_generated, real_samples_labels)
          loss_generator.backward()
          optimizer_generator.step()
      else:
        raise ValueError(f"Training mode {training_mode} not supported")

      # Store generator loss for plotting
      if generator.name not in loss_values.generator_loss_values:
        loss_values.generator_loss_values[generator.name] = []
      loss_values.generator_loss_values[generator.name].append(loss_generator.cpu().detach().numpy())
        
    # Show loss
    plot_progress.plot(epoch, loss_values)
    print(f"Training [{100 * epoch / num_epochs}%], Epoch: {epoch}, Loss G.: {loss_values.generator_loss_values[generator.name][-1]}")

    # Save checkpoint at the specified interval
    if (epoch + 1) % checkpoint_interval == 0:
      checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint.pth')
      checkpoint = {
          'epoch': epoch,
          'loss_values': loss_values
      }

      for model, optimizer in zip(model_list, optimizer_list):
        checkpoint[f'{model.name}_state_dict'] = model.state_dict()
        checkpoint[f'{optimizer.name}_state_dict'] = optimizer.state_dict()

      # Save the checkpoint
      torch.save(checkpoint, checkpoint_path)
      print(f'Checkpoint saved at epoch {epoch}')

