import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os

def get_data_loader(batch_size=64, num_workers=2):
  data_dir = '../../data'
  os.makedirs(data_dir, exist_ok=True)
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
  trainset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
  trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
  return trainloader