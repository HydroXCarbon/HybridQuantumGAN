import os
import torch

def get_checkpoint(checkpoint_folder, model_list, optimizer_list):
	os.makedirs(checkpoint_folder, exist_ok=True)
	checkpoint_path = os.path.join(checkpoint_folder, 'checkpoint.pth')
	start_epoch = 0
	loss_values = None

	if os.path.exists(checkpoint_path):
		checkpoint = torch.load(checkpoint_path)

		for model, optimizer in zip(model_list, optimizer_list):
			model.load_state_dict(checkpoint[f'{model.name}_state_dict'])
			optimizer.load_state_dict(checkpoint[f'{optimizer.name}_state_dict'])

		# Load loss value
		loss_values = checkpoint['loss_values']

		# Load epoch
		start_epoch = checkpoint['epoch'] + 1
		print(f"Loading checkpoint from epoch {start_epoch - 1}")
		
	return start_epoch, loss_values