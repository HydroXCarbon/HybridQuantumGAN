from torch import nn

class DC_Classical_Discriminator(nn.Module):
  def __init__(self):
    super().__init__()
    # Define the layers
    self.conv1 = nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False)  # Output: (batch_size, 64, 14, 14)
    self.act1 = nn.LeakyReLU(0.2, inplace=True)
    
    self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)  # Output: (batch_size, 128, 7, 7)
    self.bn2 = nn.BatchNorm2d(128)
    self.act2 = nn.LeakyReLU(0.2, inplace=True)
    
    self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)  # Output: (batch_size, 256, 3, 3)
    self.bn3 = nn.BatchNorm2d(256)
    self.act3 = nn.LeakyReLU(0.2, inplace=True)
    
    self.conv4 = nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=0, bias=False)  # Output: (batch_size, 1, 1, 1)
    self.output_act = nn.Sigmoid()

  def forward(self, x):
    # Pass input through the layers
    x = self.act1(self.conv1(x))
    x = self.act2(self.bn2(self.conv2(x)))
    x = self.act3(self.bn3(self.conv3(x)))
    x = self.output_act(self.conv4(x))
    # Flatten output for binary classification
    return x.view(-1, 1)