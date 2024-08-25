import torch.cuda
from colorama import Fore, Style, init

def get_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print(Fore.GREEN + 'Using cuda:' + Style.RESET_ALL , torch.cuda.get_device_name(device=device))
    return device
  else:
    device = torch.device("cpu")
    print(Fore.RED + 'Using cpu:' + Style.RESET_ALL + ' aborded')
    exit()
