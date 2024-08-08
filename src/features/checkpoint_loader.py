import os
import torch

def get_checkpoint(generator, classical_discriminator, optimizer_generator, optimizer_classical_discriminator):
  checkpoint_dir='../checkpoints'
  classical_discriminator_checkpoint_path = os.path.join(checkpoint_dir, 'classical_discriminator.pth')
  generator_checkpoint_path = os.path.join(checkpoint_dir, 'generator.pth')
  start_epoch = 0

  if os.path.exists(classical_discriminator_checkpoint_path):
      classical_discriminator_checkpoint = torch.load(classical_discriminator_checkpoint_path)
      classical_discriminator.load_state_dict(classical_discriminator_checkpoint['model_state_dict'])
      optimizer_classical_discriminator.load_state_dict(classical_discriminator_checkpoint['optimizer_state_dict'])
      start_epoch = classical_discriminator_checkpoint['epoch'] + 1
      print(f"Resuming training from epoch {start_epoch}")

  if os.path.exists(generator_checkpoint_path):
      generator_checkpoint = torch.load(generator_checkpoint_path)
      generator.load_state_dict(generator_checkpoint['model_state_dict'])
      optimizer_generator.load_state_dict(generator_checkpoint['optimizer_state_dict'])

  return start_epoch