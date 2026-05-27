import numpy as np

def psnr_history(x_list, clean, max_val=1.0):
    """
    Compute PSNR at each iteration.

    Args:
        x_list : list of length T, each element is either
                 - a list of blocks [x0, x1, ...] → primal is element [0]
                 - a single tensor               → used directly
        clean  : clean image tensor, shape (1, 1, H, W)
        max_val: dynamic range (default 1.0 for Shepp-Logan)

    Returns:
        list of float, length T
    """
    clean_img = clean[:, 0:1, :, :].float()
    hist = []

    for x in x_list:
        pred = x[0] if isinstance(x, list) else x
        pred = pred[:, 0:1, :, :].float()
        mse  = ((pred - clean_img) ** 2).mean().item()
        psnr = 20.0 * np.log10(max_val / np.sqrt(mse)) if mse > 1e-12 else 100.0
        hist.append(psnr)

    return hist