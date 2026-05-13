

import torch

def convergence_loss(residuals, k=15, eps=1e-8):
    R = torch.stack(residuals)
    k = min(k, len(R))
    R_early = R[:k]

    # 1. vitesse globale
    log_R = torch.log(R_early + eps)
    t = torch.arange(k, device=R.device, dtype=R.dtype)
    weights = torch.exp(-0.3 * t)
    weights = weights / weights.sum()
    L_fast = (weights * log_R).sum()

    # 2. vitesse locale (ratios)
    ratios = R_early[1:] / (R_early[:-1] + eps)
    L_ratio = ratios.mean()

    return L_fast + 0.5 * L_ratio

def kkt_loss(AxCx_list):
    T = len(AxCx_list)
    weights = torch.linspace(0.1, 1.0, T, device=AxCx_list[0].device)
    return sum(w * a for w, a in zip(weights, AxCx_list))


