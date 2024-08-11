import os
import torch
from visualization import plot_training_progress

class LossValues:
    def __init__(self):
        self.generator_loss_values = []
        self.classical_discriminator_loss_values = []
        self.entropy_values = []

def train_model(device, 
                num_epochs, 
                train_loader, 
                generator, 
                classical_discriminator, 
                optimizer_generator, 
                optimizer_classical_discriminator, 
                loss_function, 
                checkpoint_folder, 
                start_epoch=0,
                checkpoint_interval=5,
                loss_values=None):
  
  if loss_values is None:
        loss_values = LossValues()

  for epoch in range(start_epoch, num_epochs):
    for n, (real_samples, mnist_labels) in enumerate(train_loader):
      # Data for training the discriminator
      real_samples = real_samples.to(device=device)
      real_samples_labels = torch.ones((train_loader.batch_size, 1)).to(
          device=device
      )
      latent_space_samples = torch.randn((train_loader.batch_size, 100)).to(
          device=device
      )
      generated_samples = generator(latent_space_samples)
      generated_samples_labels = torch.zeros((train_loader.batch_size, 1)).to(
          device=device
      )
      all_samples = torch.cat((real_samples, generated_samples))
      all_samples_labels = torch.cat(
          (real_samples_labels, generated_samples_labels)
      )

      # Training the classical discriminator
      classical_discriminator.zero_grad()
      output_classical_discriminator = classical_discriminator(all_samples)
      loss_classical_discriminator = loss_function(
          output_classical_discriminator, all_samples_labels
      )
      loss_classical_discriminator.backward()
      optimizer_classical_discriminator.step()

      # Data for training the generator
      latent_space_samples = torch.randn((train_loader.batch_size, 100)).to(
          device=device
      )

      # Training the generator
      generator.zero_grad()
      generated_samples = generator(latent_space_samples)
      output_classical_discriminator_generated = classical_discriminator(generated_samples)
      loss_generator = loss_function(
          output_classical_discriminator_generated, real_samples_labels
      )
      loss_generator.backward()
      optimizer_generator.step()

      # Store loss for plotting
      loss_values.generator_loss_values.append(loss_generator.cpu().detach().numpy())
      loss_values.classical_discriminator_loss_values.append(loss_classical_discriminator.cpu().detach().numpy())
      loss_values.entropy_values.append(loss_generator.cpu().detach().numpy() + loss_classical_discriminator.cpu().detach().numpy())
        
    # Show loss
    plot_training_progress(loss_values.generator_loss_values, loss_values.classical_discriminator_loss_values, loss_values.entropy_values)
    print(f"Epoch: {epoch} Loss D.: {loss_classical_discriminator} Loss G.: {loss_generator}")

    # Save checkpoint at the specified interval
    if (epoch + 1) % checkpoint_interval == 0:
        
      checkpoint_path = os.path.join(checkpoint_folder, f'checkpoint.pth')
      torch.save({
        'epoch': epoch + 1,
        'generator_state_dict': generator.state_dict(),
        'classical_discriminator_state_dict': classical_discriminator.state_dict(),
        'optimizer_generator_state_dict': optimizer_generator.state_dict(),
        'optimizer_classical_discriminator_state_dict': optimizer_classical_discriminator.state_dict(),
        'loss_values': loss_values
      }, checkpoint_path)
      print(f'Checkpoint saved at {checkpoint_path}')

