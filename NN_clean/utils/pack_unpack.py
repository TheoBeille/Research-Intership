# utils/pack_unpack.py

import torch
@staticmethod
def pack(blocks):

    return torch.cat(blocks, dim=1)

def unpack(tensor, shapes):
    """
    Convert tensor → list of blocks

    Args:
        tensor: [B, C_total, H, W]
        shapes: list of shapes defining each block

    Returns:
        list of tensors
    """
    out = []
    c = 0

    for s in shapes:
        ch = s[1]
        out.append(tensor[:, c:c + ch])
        c += ch

    return out


def ensure_4d(x):
    """
    Ensure tensor is [B,C,H,W]
    """
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    return x


def prepare_blocks(blocks):
    """
    Ensure all blocks are well formatted tensors
    """
    fixed = []
    for b in blocks:
        b = ensure_4d(b)
        fixed.append(b.float())
    return fixed