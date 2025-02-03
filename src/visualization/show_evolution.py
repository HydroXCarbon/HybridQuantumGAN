import matplotlib.pyplot as plt
from math import sqrt, ceil

class PlotEvolution:
  counter = 0

  def __init__(self, epochs):
    plt.ion()  # Enable interactive mode
    if epochs == 0 or epochs == 1:
      epochs = 2
    self.cols = ceil(sqrt(epochs))
    self.rows = ceil(epochs / self.cols)
    self.cols = 5
    self.rows = 4
    
    self.fig, self.axes = plt.subplots(self.rows, self.cols, figsize=(18, 14))
    self.fig.canvas.manager.set_window_title('Training Evolution') 
    for ax in self.axes.flatten():
      ax.set_xticks([])
      ax.set_yticks([])

  def plot(self, data, epoch):
    if self.counter > (self.rows * self.cols) - 1:
      return
    
    axes = self.axes.flatten()

    axes[self.counter].imshow(data.reshape(28, 28), cmap="gray_r")
    axes[self.counter].set_title(f'Epoch {epoch}')
    self.counter += 1
    self.fig.canvas.draw()
    plt.pause(0.1)

  def save(self, filename):
        self.fig.savefig(filename, bbox_inches='tight')
        print(f"Figure saved as {filename}")