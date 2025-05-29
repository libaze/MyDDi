import torch

optimizers = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

def get_optimizer(optimizer_type):
    if optimizer_type not in optimizers:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
    return optimizers[optimizer_type]

