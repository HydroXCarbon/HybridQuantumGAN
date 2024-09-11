import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from os import makedirs

def get_data_loader(data_folder, batch_size=64):
  makedirs(data_folder, exist_ok=True)
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
  trainset = datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
  trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4, prefetch_factor=2)
  return trainloader