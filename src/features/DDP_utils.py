import os
import torch.distributed as dist

def setup(rank, world_size, device):
    # Set up distributed training
    if device.type == 'cpu':
        backend = 'gloo'
        rank = 'cpu'
    else:
        backend = 'nccl'

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # initialize the process group
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    return rank

def cleanup():
    dist.destroy_process_group()