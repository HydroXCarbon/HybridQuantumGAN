from torch import nn

class ClassicalDiscriminator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.Conv2d(1, 2, kernel_size=5),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Conv2d(2, 16, kernel_size=5),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2),
      nn.Dropout2d(),
      nn.Flatten(),
      nn.Linear(256, 64),
      nn.ReLU(),
      nn.Linear(64, 1),
      nn.Sigmoid(), 
    )

  def forward(self, x):
    x = self.model(x)
    return x
    