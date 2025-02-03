from torch import nn

class DC_Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      nn.ConvTranspose2d(100, 128, kernel_size=4, stride=1, padding=0),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      # Output size: (128) x 4 x 4

      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      # Output size: (64) x 8 x 8

      nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(32),
      nn.ReLU(True),
      # Output size: (32) x 16 x 16

      nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(16),
      nn.ReLU(True),
      # Output size: (16) x 32 x 32

      nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=3),
      nn.Tanh()  # Output size will be (1) x 28 x 28
      # Output size: (1) x 28 x 28
    )

  def forward(self, x):
    x = x.view(x.size(0), 100, 1, 1)
    output = self.model(x)
    return output