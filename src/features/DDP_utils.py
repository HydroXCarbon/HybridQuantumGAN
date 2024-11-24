import os
import torch.distributed as dist

def setup(rank, world_size, device):
    # Set up distributed training
    backend = 'gloo' if device.type == 'cpu' else 'nccl'

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'

    # Initialize the process group
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()