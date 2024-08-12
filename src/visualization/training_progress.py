import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

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

        self.epoch = 0
        self.generator_loss_values = []
        self.discriminator_loss_values = []
        self.entropy_values = []

        self.anim = FuncAnimation(self.fig, self.update_plot, interval=10, cache_frame_data=False)

    def update_plot(self, _):
        self.ax1.cla()
        self.ax2.cla()

        self.ax1.set_xlabel("Iteration")
        self.ax1.set_ylabel("Loss")
        self.ax1.grid()

        self.ax2.set_xlabel("Iteration")
        self.ax2.set_ylabel("Relative entropy")
        self.ax2.grid()

        self.ax1.set_title(f"Loss (Epoch {self.epoch})")
        self.ax1.plot(self.generator_loss_values, label="generator loss", color="royalblue")
        self.ax1.plot(self.discriminator_loss_values, label="discriminator loss", color="magenta")
        self.ax1.legend(loc="best")

        self.ax2.set_title(f"Relative entropy (Epoch {self.epoch})")
        self.ax2.plot(self.entropy_values)

        self.fig.canvas.draw()

    def plot(self, epoch, generator_loss_values, discriminator_loss_values, entropy_values):
        self.epoch = epoch
        self.generator_loss_values = generator_loss_values
        self.discriminator_loss_values = discriminator_loss_values
        self.entropy_values = entropy_values
        self.fig.canvas.flush_events()  