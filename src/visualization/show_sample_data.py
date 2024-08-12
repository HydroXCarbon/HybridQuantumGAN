import matplotlib.pyplot as plt
import time
def show_sample_data(data, sample_size=16):
  fig = plt.figure(figsize=(8, 8))
  fig.canvas.manager.set_window_title('Sample Data') 
  for i in range(sample_size):
      ax = plt.subplot(4, 4, i + 1)
      plt.imshow(data[i].reshape(28, 28), cmap="gray_r") 
      plt.xticks([]) 
      plt.yticks([]) 
      
  plt.show(block=False)
  plt.pause(0.1)

