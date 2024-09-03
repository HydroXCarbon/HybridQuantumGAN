from os import path
from colorama import Fore, Style
from torch import save, load

def save_checkpoint(epoch, checkpoint_folder, model_list, optimizer_list, loss_values):
  checkpoint_path = path.join(checkpoint_folder, f'checkpoint.pth')
  checkpoint = {
    'epoch': epoch,
    'loss_values': loss_values
  }
  for model, optimizer in zip(model_list, optimizer_list):
    checkpoint[f'{model.name}_state_dict'] = model.state_dict()
    checkpoint[f'{optimizer.name}_state_dict'] = optimizer.state_dict()

  # Save the checkpoint
  save(checkpoint, checkpoint_path)

def get_checkpoint(checkpoint_folder, model_list, optimizer_list):
	checkpoint_path = path.join(checkpoint_folder, 'checkpoint.pth')
	start_epoch = 0
	loss_values = None

	if path.exists(checkpoint_path):
		checkpoint = load(checkpoint_path)

		for model, optimizer in zip(model_list, optimizer_list):
			model.load_state_dict(checkpoint[f'{model.name}_state_dict'])
			optimizer.load_state_dict(checkpoint[f'{optimizer.name}_state_dict'])

		# Load loss value
		loss_values = checkpoint['loss_values']

		# Load epoch
		start_epoch = checkpoint['epoch'] + 1
		print(Fore.GREEN + "Loading checkpoint: " + Style.RESET_ALL, end='')
		print(f"epoch {start_epoch - 1}")
		
	return start_epoch, loss_values