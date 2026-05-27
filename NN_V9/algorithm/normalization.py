
import torch


def norm_sq(x):
    """
    Squared norm over spatial tensor [B,C,H,W]
    """
    return torch.sum(x**2, dim=(1,2,3), keepdim=True)


def safe_sqrt(x):
    """
    Safe sqrt to avoid NaNs (same idea as repo)
    """
    return torch.sqrt(torch.where(x == 0, x + 1e-12, x))


def norm(x):
    return safe_sqrt(norm_sq(x))


def safely_normalize(x):
    """
    Normalize tensor safely (no division by zero)
    """
    return x / (safe_sqrt(norm_sq(x)) + 1e-12)




def block_norm_sq(blocks):
    return sum((b**2).sum(dim=(1,2,3), keepdim=True) for b in blocks)


def block_norm(blocks):
    return safe_sqrt(block_norm_sq(blocks))


def safely_normalize_blocks(blocks):
    n = block_norm(blocks)
    return [b / (n + 1e-12) for b in blocks]




def scale_blocks(blocks, scale):
    return [scale * b for b in blocks]


def normalize_and_scale(u_raw, v_raw, delta, alpha=0.9):



    u_dir = safely_normalize_blocks(u_raw)
    v_dir = safely_normalize_blocks(v_raw)


    scale = safe_sqrt(delta + 1e-12)


    while scale.dim() < 4:
        scale = scale.unsqueeze(-1)


    u_scaled = scale_blocks(u_dir, alpha * scale)
    v_scaled = scale_blocks(v_dir, alpha * scale)

    return u_scaled, v_scaled