

import torch


def convergence_loss(residuals, eps=1e-8, w_early=2.0, w_final=1.0, w_monot=5.0):
    """
    Main loss used to encourage fast and stable convergence.

    Terms:
    - early residual average: encourages fast decrease at the beginning
    - final residual: encourages good final performance
    - monotonicity penalty: discourages oscillations / increases
    """
    R = torch.stack(residuals) 

    early_len = max(3, len(residuals) // 4)
    early = R[:early_len].mean()
    final = R[-1]
    monot = torch.relu(R[1:] - R[:-1]).mean() if len(residuals) > 1 else torch.zeros_like(final)

    loss = (
        w_early * torch.log10(early + eps)
        + w_final * torch.log10(final + eps)
        + w_monot * monot
    )
    return loss


def final_residual_loss(residuals, eps=1e-8):
    """
    Simpler loss: only final residual.
    Useful as a baseline.
    """
    R = torch.stack(residuals)
    return torch.log10(R[-1] + eps)


def weighted_residual_loss(residuals, eps=1e-8, power=1.0):
    """
    Weighted average over all residuals.
    Later iterations can be emphasized with power > 1.
    """
    R = torch.stack(residuals)
    T = len(residuals)

    weights = torch.arange(1, T + 1, device=R.device, dtype=R.dtype) ** power
    weights = weights / weights.sum()

    return torch.sum(weights * torch.log10(R + eps))


def residual_mse_loss(residuals, target=0.0):
    """
    MSE-to-target loss on the whole residual trajectory.
    Mostly useful for experiments.
    """
    R = torch.stack(residuals)
    target_tensor = torch.full_like(R, fill_value=target)
    return torch.mean((R - target_tensor) ** 2)