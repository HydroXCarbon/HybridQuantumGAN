from torch import nn

class FCL_Classical_Discriminator_2(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Flatten(),
      nn.Linear(784, 512),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(512, 256),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(256, 1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    x = self.model(x)
    return x
    