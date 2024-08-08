import matplotlib.pyplot as plt

def show_sample_data(data, sample_size=16):
    for i in range(sample_size):
      ax = plt.subplot(4, 4, i + 1)
      plt.imshow(data[i].reshape(28, 28), cmap="gray_r")
      plt.xticks([])
      plt.yticks([])
    plt.savefig('output.png')