from IPython.display import clear_output
import matplotlib.pyplot as plt

def plot_training_progress(generator_loss_values, discriminator_loss_values, entropy_values):
  # we don't plot if we don't have enough data
  if len(generator_loss_values) < 2:
    return

  clear_output(wait=True)
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))

  # Generator Loss
  ax1.set_title("Loss")
  ax1.plot(generator_loss_values, label="generator loss", color="royalblue")
  ax1.plot(discriminator_loss_values, label="discriminator loss", color="magenta")
  ax1.legend(loc="best")
  ax1.set_xlabel("Iteration")
  ax1.set_ylabel("Loss")
  ax1.grid()

  # Relative Entropy
  ax2.set_title("Relative entropy")
  ax2.plot(entropy_values)
  ax2.set_xlabel("Iteration")
  ax2.set_ylabel("Relative entropy")
  ax2.grid()

  plt.show()