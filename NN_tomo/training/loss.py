

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

def tgv_loss(F_vals, eps=1e-8):
    """
    F_vals: list of energy values (TGV)

    """

    T = len(F_vals)


    F0 = F_vals[0].detach() + eps
    F_norm = [F / F0 for F in F_vals]

    L_final = F_norm[-1]

    weights = torch.linspace(1.0, 0.1, T, device=F_vals[0].device)
    weights = weights / weights.sum()

    L_fast = sum(w * F for w, F in zip(weights, F_norm))


    L_monot = sum(
        torch.relu(F_norm[i+1] - F_norm[i])
        for i in range(T-1)
    )



    # ======================
    # FINAL LOSS
    # ======================
    loss = (
        1.0 * L_final       # accuracy
        + 0.7 * L_fast      # speed
        + 5.0 * L_monot     # stability
    )


    return loss