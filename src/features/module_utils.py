from torch.nn.parallel import DistributedDataParallel as DDP

import torch

def move_optimizer_to_device(optimizer, device):
  for state in optimizer.state.values():
    if isinstance(state, torch.Tensor):
      state.data = state.data.to(device)
      if state._grad is not None:
        state._grad = state._grad.to(device)
    elif isinstance(state, dict):
      for key, value in state.items():
        if isinstance(value, torch.Tensor):
          state[key] = value.to(device)

def move_model_and_optimizer_to_device(model_list, optimizer_list, rank, device):
  # Move models to device and wrap with DistributedDataParallel
  temp_device = 'cpu' if device == 'cpu' else rank

  generator, discriminator_list = model_list[0], model_list[1:]
  optimizer_generator, optimizer_discriminator_list = optimizer_list[0], optimizer_list[1:]

  generator = generator.to(temp_device)
  if device == 'cpu':
    generator = DDP(generator)
  else:
    generator = DDP(generator, device_ids=[rank])

  for i in range(len(discriminator_list)):
    discriminator_list[i] = discriminator_list[i].to(temp_device)
    if device == 'cpu':
      discriminator_list[i] = DDP(discriminator_list[i])
    else:
      discriminator_list[i] = DDP(discriminator_list[i], device_ids=[rank])

  # Move optimizers to the correct device
  move_optimizer_to_device(optimizer_generator, temp_device)
  for optimizer_discriminator in optimizer_discriminator_list:
    move_optimizer_to_device(optimizer_discriminator, temp_device)

  return generator, discriminator_list, optimizer_generator, optimizer_discriminator_list

def train_discriminator(discriminator, optimizer_discriminator, all_samples, all_samples_labels, retain_graph):
  discriminator.zero_grad()
  optimizer_discriminator.zero_grad()
  output_discriminator = discriminator(all_samples)
  loss_discriminator = discriminator.module.loss_function(output_discriminator, all_samples_labels)
  loss_discriminator.backward(retain_graph=retain_graph)
  optimizer_discriminator.step()
  return loss_discriminator.cpu().detach().numpy()

def train_generator(generator, discriminator_list, optimizer_generator, latent_space_samples, real_samples_labels, training_mode, epoch, num_discriminators):
  generator.zero_grad()
  generated_samples = generator(latent_space_samples)
  
  if training_mode == 'combined':
    outputs = []
    for discriminator in discriminator_list:
      outputs.append(discriminator(generated_samples))
    combined_output = torch.mean(torch.stack(outputs), dim=0)
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
  return loss_generator.cpu().detach().numpy()

def denormalize_and_convert_uint8(images):
  images = (images * 0.5 + 0.5) * 255.0  
  images = images.clamp(0, 255) 
  images = images.to(torch.uint8) 
  return images