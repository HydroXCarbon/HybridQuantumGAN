import os
import torch

def train_model(device, 
                batch_size, 
                num_epochs, 
                train_loader, 
                generator, 
                classical_discriminator, 
                optimizer_generator, 
                optimizer_classical_discriminator, 
                loss_function, 
                checkpoint_folder, 
                start_epoch=0):
  
  generator_loss_values = []
  classical_discriminator_loss_values = []
  entropy_values = []

  for epoch in range(start_epoch, num_epochs):
      for n, (real_samples, mnist_labels) in enumerate(train_loader):
          # Data for training the discriminator
          real_samples = real_samples.to(device=device)
          real_samples_labels = torch.ones((batch_size, 1)).to(
              device=device
          )
          latent_space_samples = torch.randn((batch_size, 100)).to(
              device=device
          )
          generated_samples = generator(latent_space_samples)
          generated_samples_labels = torch.zeros((batch_size, 1)).to(
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
          latent_space_samples = torch.randn((batch_size, 100)).to(
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
          generator_loss_values.append(loss_generator.cpu().detach().numpy())
          classical_discriminator_loss_values.append(loss_classical_discriminator.cpu().detach().numpy())
          entropy_values.append(loss_generator.cpu().detach().numpy() + loss_classical_discriminator.cpu().detach().numpy())
          
      # Show loss
      plot_training_progress(generator_loss_values, classical_discriminator_loss_values, entropy_values)
      print(f"Epoch: {epoch} Loss D.: {loss_classical_discriminator} Loss G.: {loss_generator}")

      # Save Checkpoints for every 10 epoch
      if epoch % 5 == 0:
          
          # Saving the discriminator's state
          discriminator_path = os.path.join(checkpoint_folder, f'discriminator.pth')
          torch.save({
              'epoch': epoch,
              'model_state_dict': classical_discriminator.state_dict(),
              'optimizer_state_dict': optimizer_classical_discriminator.state_dict(),
              'loss': loss_classical_discriminator.item(),
          }, discriminator_path)

          # Saving the generator's state
          generator_path = os.path.join(checkpoint_folder, f'generator.pth')
          torch.save({
              'epoch': epoch,
              'model_state_dict': generator.state_dict(),
              'optimizer_state_dict': optimizer_generator.state_dict(),
              'loss': loss_generator.item(),
          }, generator_path)

