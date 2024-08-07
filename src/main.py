from model.Generator import Generator
from model.ClassicalDiscriminator import Discriminator as Cdiscriminator
from model.QuantumDiscriminator import Discriminator as Qdiscriminator

import torch

# Set seed
torch.manual_seed(111)

# Use cuda if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using device:', device)
    print(torch.cuda.get_device_name(device=device))
    print('Total memory',torch.cuda.get_device_properties(device).total_memory/ 1e9, 'GB')
else:
    device = torch.device("cpu")
    print('Using device:', device)
    exit()

# Load models
generator = Generator().to(device)
classicalDiscriminator = Cdiscriminator.to(device)
quantumDiscriminator = Qdiscriminator.to(device)