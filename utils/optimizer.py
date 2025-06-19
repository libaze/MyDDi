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


def get_scheduler(optimizer, scheduler_config):
    if scheduler_config["type"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config["CosineAnnealingLR"]["T_max"],
            eta_min=scheduler_config["CosineAnnealingLR"]["eta_min"],
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config["StepLR"]["step_size"],
            gamma=scheduler_config["StepLR"]["gamma"],
        )
    return scheduler
