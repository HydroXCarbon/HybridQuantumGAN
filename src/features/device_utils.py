import torch.cuda
from colorama import Fore, Style, init

def get_device(device):
  if torch.cuda.is_available() and device == 'cuda':
    device = torch.device("cuda")
    print(Fore.GREEN + 'Using cuda:' + Style.RESET_ALL, torch.cuda.get_device_name(device=device))
    return device
  elif device == 'cpu':
    device = torch.device("cpu")
    print(Fore.GREEN+ 'Using cpu (force):' + Style.RESET_ALL + ' CPU')
    return device
  else:
    device = torch.device("cpu")
    print(Fore.RED + 'Using cpu:' + Style.RESET_ALL + ' aborded')
    exit()
