import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DistributedSampler,DataLoader
from os import makedirs

def get_data_loader(rank, world_size, data_folder, batch_size):
  makedirs(data_folder, exist_ok=True)
  transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
  ])
  trainset = datasets.MNIST(root=data_folder, train=True, download=True, transform=transform)
  sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True)
  trainloader = DataLoader(
      trainset,
      batch_size=batch_size,
      sampler=sampler,  # Use the sampler here
      drop_last=True,
      num_workers=4,
      prefetch_factor=2
    )
  return trainloader