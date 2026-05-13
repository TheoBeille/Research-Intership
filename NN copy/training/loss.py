

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

def tgv_loss(F_vals, k=15, eps=1e-8):
    """
    Speed-focused loss for TGV energy.

    Objective:
    - minimize energy FAST in first k iterations
    - ignore long-term convergence
    """

    T = len(F_vals)
    k = min(k, T)

    F = torch.stack(F_vals)

    # normalize (important for scale invariance)
    F0 = F[0].detach() + eps
    F_norm = F / F0

    F_early = F_norm[:k]

    # log scale = multiplicative decrease
    log_F = torch.log(F_early + eps)

    # exponential weights → strong focus on early iterations
    t = torch.arange(k, device=F.device, dtype=F.dtype)
    weights = torch.exp(-0.4 * t)
    weights = weights / weights.sum()

    loss = (weights * log_F).sum()

    return loss


