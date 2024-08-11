import os
import torch

def get_checkpoint(checkpoint_folder, generator, classical_discriminator, optimizer_generator, optimizer_classical_discriminator):
	checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint.pth')
	start_epoch = 0

	if os.path.exists(checkpoint_path):
		checkpoint = torch.load(checkpoint_path)

		# Load classical discriminator checkpoint
		classical_discriminator.load_state_dict(checkpoint['classical_discriminator_state_dict'])
		optimizer_classical_discriminator.load_state_dict(checkpoint['optimizer_classical_discriminator_state_dict'])

		# Load generator checkpoint
		generator.load_state_dict(checkpoint['generator_state_dict'])
		optimizer_generator.load_state_dict(checkpoint['optimizer_generator_state_dict'])

		# Load loss value
		loss_values = checkpoint['loss_value']

		# Load epoch
		start_epoch = checkpoint['epoch'] + 1
		print(f"Resuming training from epoch {start_epoch}")
		
	return start_epoch, loss_values