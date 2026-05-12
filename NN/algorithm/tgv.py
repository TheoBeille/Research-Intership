import torch
import torch.nn.functional as F


# ---------------------------
# Gradient (forward differences)
# ---------------------------
def gradient(u):
    dx = F.pad(u[..., 1:, :] - u[..., :-1, :], (0,0,0,1), mode='replicate')
    dy = F.pad(u[..., :, 1:] - u[..., :, :-1], (0,1,0,0), mode='replicate')

    return torch.stack((dx, dy), dim=-3)  # shape: (2, H, W)


# ---------------------------
# Symmetrized gradient E(w)
# ---------------------------
def sym_grad(w):
    w1, w2 = w[:, 0], w[:, 1]

    dx_w1 = w1[..., 1:, :] - w1[..., :-1, :]
    dy_w2 = w2[..., :, 1:] - w2[..., :, :-1]

    dx_w2 = w2[..., 1:, :] - w2[..., :-1, :]
    dy_w1 = w1[..., :, 1:] - w1[..., :, :-1]

    dx_w1 = F.pad(dx_w1, (0, 0, 0, 1))
    dy_w2 = F.pad(dy_w2, (0, 1, 0, 0))
    dx_w2 = F.pad(dx_w2, (0, 0, 0, 1))
    dy_w1 = F.pad(dy_w1, (0, 1, 0, 0))

    e1 = dx_w1
    e2 = dy_w2
    e3 = 0.5 * (dy_w1 + dx_w2)

    return torch.stack((e1, e2, e3), dim=-3)  # shape: (3, H, W)


# ---------------------------
# TGV loss
# ---------------------------
def tgv(u, w, data, alpha=1e-1, beta=1.0):
    """
    u: (B, 1, H, W)
    w: (B, 2, H, W)
    A: function or nn.Module
    data: (B, 1, H, W)
    """

def tgv(u, w, data, alpha=1e-1, beta=1.0):

    term_data = 0.5 * torch.sum((u - data) ** 2)

    Gu = gradient(u)
    term_tgv1 = alpha * torch.sum(torch.abs(Gu - w))

    Ew = sym_grad(w)
    term_tgv2 = alpha * beta * torch.sum(torch.abs(Ew))

    return term_data + term_tgv1 + term_tgv2



