import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


class PlotTrainingProgress:

  def __init__(self):
    plt.ion()  # Enable interactive mode
    self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(18, 9))
    self.fig.canvas.manager.set_window_title('Training Progress') 

    self.ax1.set_xlabel("Iteration")
    self.ax1.set_ylabel("Loss")
    self.ax1.grid()

    self.ax2.set_xlabel("Iteration")
    self.ax2.set_ylabel("Relative entropy")
    self.ax2.grid()

  def plot(self, epoch, loss_values, fid_score):
    self.ax1.cla()
    self.ax2.cla()

    self.ax1.set_title(f"Overall Loss (Epoch {epoch})")
    # Plot generator loss
    for generator in loss_values.generator_loss_values:
      values = loss_values.generator_loss_values[generator]
      self.ax1.plot(values, label=generator)

    # Plot discriminator loss
    for discriminator in loss_values.discriminator_loss_values:
      values = loss_values.discriminator_loss_values[discriminator]
      self.ax1.plot(values, label=discriminator, alpha=0.5)

    self.ax1.legend(loc="best")
    self.ax1.set_xlabel("Iteration")
    self.ax1.set_ylabel("Loss value")
    self.ax1.xaxis.set_major_locator(MaxNLocator(integer=True))


    self.ax2.set_title(f"FID Score (Epoch {epoch})")
    data, epoch = zip(*fid_score)
    self.ax2.plot(epoch, data, marker='o')
    self.ax2.set_xlabel("Epochs")
    self.ax2.set_ylabel("FID Score")
    self.ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    self.fig.canvas.draw()
    plt.pause(0.1)