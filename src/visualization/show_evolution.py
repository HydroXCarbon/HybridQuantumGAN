import matplotlib.pyplot as plt
from math import sqrt, ceil

class PlotEvolution:

  def __init__(self, epochs):
    plt.ion()  # Enable interactive mode
    if epochs == 0 or epochs == 1:
      epochs = 2
    cols = ceil(sqrt(epochs))
    rows = ceil(epochs / cols)
    
    self.fig, self.axes = plt.subplots(rows, cols, figsize=(18, 14))
    self.fig.canvas.manager.set_window_title('Training Evolution') 
    for ax in self.axes.flatten():
      ax.set_xticks([])
      ax.set_yticks([])

  def plot(self, data, epoch, epoch_i):

    axes = self.axes.flatten()

    axes[epoch_i].imshow(data.reshape(28, 28), cmap="gray_r")
    axes[epoch_i].set_title(f'Epoch {epoch}')
      
    self.fig.canvas.draw()
    plt.pause(0.1)