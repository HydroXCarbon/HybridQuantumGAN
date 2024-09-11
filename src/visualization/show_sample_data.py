import matplotlib.pyplot as plt
from math import sqrt, ceil

def show_sample_data(sample, sample_size=0, title='Sample Data', epoch=0):
  if sample_size == 0:
    sample_size = len(sample)
  # Calculate the number of rows and columns
  cols = ceil(sqrt(sample_size))
  if cols == 0:
    cols = 1
  rows = ceil(sample_size / cols)

  fig = plt.figure(figsize=(cols * 2, rows * 2))
  fig.canvas.manager.set_window_title(title) 
  for i in range(sample_size):
      plt.subplot(rows, cols, i + 1)
      plt.imshow(sample[i].reshape(28, 28), cmap="gray_r") 
      plt.xticks([]) 
      plt.yticks([]) 
      if epoch > 0:
        plt.title(f'Epoch {i}')
      
  plt.show(block=False)
  plt.pause(0.1)

