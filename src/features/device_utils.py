import torch.cuda
from colorama import Fore, Style, init

def get_device(device):
  if torch.cuda.is_available() and device.startswith('cuda'):
    try:
      torch_device = torch.device(device)
      print(Fore.GREEN + 'Using cuda:' + Style.RESET_ALL, torch.cuda.get_device_name(device=torch_device))
      return torch_device
    except Exception as e:
      print(Fore.RED + f'Error:' + Style.RESET_ALL + f' {e} aborded mission')
      exit()
  elif device == 'cpu':
    torch_device = torch.device("cpu")
    print(Fore.GREEN+ 'Using cpu (force):' + Style.RESET_ALL + ' CPU')
    return torch_device
  else:
    device = torch.device("cpu")
    print(Fore.RED + 'Using cpu:' + Style.RESET_ALL + ' aborded mission')
    exit()
