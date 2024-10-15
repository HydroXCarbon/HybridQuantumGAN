from os import path
from colorama import Fore, Style

import torch
import os

def save_checkpoint(epoch, checkpoint_path, model_list, optimizer_list, loss_values, fid_score, wandb_instant=None):
	
	checkpoint = {
		'epoch': epoch,
		'loss_values': loss_values,
		'fid_score': fid_score,
		'run_id': wandb_instant.id if wandb_instant else None
	}
	for model, optimizer in zip(model_list, optimizer_list):
		checkpoint[f'{model.name}_state_dict'] = model.state_dict()
		checkpoint[f'{optimizer.name}_state_dict'] = optimizer.state_dict()

	# Save the checkpoint
	torch.save(checkpoint, checkpoint_path)

	# Save the checkpoint to wandb
	if wandb_instant:
		wandb_instant.save(checkpoint_path, base_path=os.path.dirname(checkpoint_path))

def load_run_id(checkpoint_path):
	run_id = None
	try:
		if path.exists(checkpoint_path):
			checkpoint = torch.load(checkpoint_path)
			run_id = checkpoint['run_id']
	except Exception as e:
		print(Fore.RED + "Error:" + Style.RESET_ALL + " loading run_id from checkpoint.")
		exit(1)
	return run_id

def get_checkpoint(checkpoint_path, model_list, optimizer_list):
	start_epoch = 0
	loss_values = None
	fid_score = []

	if path.exists(checkpoint_path):
		checkpoint = torch.load(checkpoint_path)

		for model in model_list:
			model.load_state_dict(checkpoint[f'{model.name}_state_dict'])
		for optimizer in optimizer_list:
			optimizer.load_state_dict(checkpoint[f'{optimizer.name}_state_dict'])

		# Load loss value
		loss_values = checkpoint['loss_values']

		# Load FID score
		fid_score = checkpoint['fid_score']

		# Load epoch
		start_epoch = checkpoint['epoch'] + 1
		print(Fore.GREEN + "Loading checkpoint: " + Style.RESET_ALL, end='')
		print(f"Epoch {start_epoch - 1}")
		
	return start_epoch, loss_values, fid_score