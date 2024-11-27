import torch.nn as nn
class DC_Classical_Discriminator(nn.Module):
	def __init__(self):
		super().__init__()
		self.model = nn.Sequential(
			# Input: 1 x 28 x 28 (grayscale image)
			nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
			nn.LeakyReLU(0.2, inplace=True),
			# State: 64 x 14 x 14
			
			nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace=True),
			# State: 128 x 7 x 7
			
			nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace=True),
			# State: 256 x 4 x 4
			
			nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0),
			nn.Sigmoid()
			# Output: 1 (binary classification)
		)

	def forward(self, x):
		x = self.model(x)
		return x.view(-1,1)
