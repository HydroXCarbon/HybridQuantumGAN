from torch import nn

class GAN_Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(100, 256),
      nn.ReLU(),
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, 1024),
      nn.ReLU(),
      nn.Linear(1024, 784),
      nn.Tanh(),
    )

  def forward(self, x):
    output = self.model(x)
    output = output.view(x.size(0), 1, 28, 28)
    return output