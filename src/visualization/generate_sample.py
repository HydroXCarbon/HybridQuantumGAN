import torch
import matplotlib.pyplot as plt

def generate_sample(generator, device, batch_size):
  latent_space_samples = torch.randn(batch_size, 100).to(device=device)
  generated_samples = generator(latent_space_samples)
  generated_samples = generated_samples.cpu().detach()

  fig = plt.figure(figsize=(8, 8))
  fig.canvas.manager.set_window_title('Generated Samples') 
  for i in range(batch_size):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(generated_samples[i].reshape(28, 28), cmap="gray_r")
    plt.xticks([])
    plt.yticks([])

  plt.show(block=False)
  plt.pause(0.1)