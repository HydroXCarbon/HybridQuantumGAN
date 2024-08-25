import torch
import matplotlib.pyplot as plt

def generate_sample(generator, device, sample_size):
  latent_space_samples = torch.randn(sample_size, 100).to(device=device)
  generated_samples = generator(latent_space_samples)
  return generated_samples.cpu().detach()