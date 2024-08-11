import torch.cuda

def get_device():
  if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Using cuda:', torch.cuda.get_device_name(device=device))
    return device
  else:
    device = torch.device("cpu")
    print('Using cpu: aborded')
    exit()
