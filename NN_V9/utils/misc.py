# utils/misc.py

import torch


def safe_tensor(x):
    """
    Replace NaN / inf values
    """
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def safe_blocks(blocks):
    return [safe_tensor(b) for b in blocks]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def print_model_info(model):
    n_params = count_parameters(model)
    print(f"Model parameters: {n_params:,}")


def to_device(data, device):
    """
    Move tensors (or lists of tensors) to device
    """
    if isinstance(data, list):
        return [to_device(x, device) for x in data]
    if torch.is_tensor(data):
        return data.to(device)
    return data