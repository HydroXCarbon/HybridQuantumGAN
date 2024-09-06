from torch import nn

class DCGAN_Generator(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = nn.Sequential(
      # First layer: latent vector input (100), output 128 feature maps
      nn.ConvTranspose2d(100, 128, kernel_size=4, stride=1, padding=0),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      
      # Second layer: input 128, output 64 feature maps
      nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(64),
      nn.ReLU(True),
      
      # Final layer: input 64, output 1 (for grayscale)
      nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
      nn.Tanh()  # Tanh activation for output
    )

  def forward(self, x):
    output = self.model(x)
    output = output.view(x.size(0), 1, 28, 28)
    return output