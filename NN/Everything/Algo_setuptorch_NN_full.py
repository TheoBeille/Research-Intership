# Algo_setuptorch_NN_full.py
import odl
import numpy as np
import odl.contrib.torch as odl_torch
import torch

def get_setup(seed=0, noise_level=0.1, device=None):
    size = 16
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    space = odl.uniform_discr([0, 0], [100, 100], [size, size])

    phantom = odl.phantom.shepp_logan(space, modified=True)
    noisy   = phantom + noise_level * odl.phantom.white_noise(space, seed=seed)

    D  = odl.Gradient(space, method='forward', pad_mode='symmetric')
    Dx = odl.PartialDerivative(space, 0, method='forward', pad_mode='symmetric')
    Dy = odl.PartialDerivative(space, 1, method='forward', pad_mode='symmetric')

    V  = D.range
    P0 = odl.ComponentProjection(V, 0)
    P1 = odl.ComponentProjection(V, 1)

    E = odl.BroadcastOperator(
        Dx * P0,
        0.5 * Dy * P0 + 0.5 * Dx * P1,
        0.5 * Dx * P1 + 0.5 * Dy * P0,
        Dy * P1
    )

    # Torch wrappers
    D_layer     = odl_torch.OperatorModule(D)
    D_adj_layer = odl_torch.OperatorModule(D.adjoint)
    E_layer     = odl_torch.OperatorModule(E)
    E_adj_layer = odl_torch.OperatorModule(E.adjoint)

    def odl_op(layer, x):
        # ensure CPU for ODL operator then back to device
        dev = x.device
        out = layer(x.cpu())
        return out.to(dev)

    def K_torch(u):
        # u is list of 4 tensors
        out0 =  odl_op(D_adj_layer, u[2])
        out1 = -u[2] + odl_op(E_adj_layer, u[3])
        out2 = -odl_op(D_layer, u[0]) + u[1]
        out3 = -odl_op(E_layer, u[1])
        return [out0, out1, out2, out3]

    noisy_torch = (torch.tensor(np.asarray(noisy), dtype=torch.float32)
                   .unsqueeze(0)  # batch dim
                   .to(device))

    return dict(
        space=space,
        noisy=noisy_torch,
        K=K_torch,
        device=device,
        size=size,
    )


class Params:
    def __init__(self,
                 lam0=1.0,
                 beta_bar=1.0,
                 gamma0=0.1,
                 alpha1=0.1,
                 alpha2=0.1,
                 zeta=0.9):
        self.lam0     = lam0
        self.beta_bar = beta_bar
        self.gamma0   = gamma0
        self.alpha1   = alpha1
        self.alpha2   = alpha2
        self.zeta     = zeta


def build_algo_functions(setup, params):
    noisy_torch = setup['noisy']
    K_torch     = setup['K']
    device      = setup['device']

    eps = 1e-12

    def proj_ball_torch(z, radius):
        # project per batch element (norm over channels+spatial)
        if z is None:
            return z
        if z.dim() == 4:
            # (B,C,H,W) -> norm per batch
            flat = z.view(z.shape[0], -1)
            nrm = torch.norm(flat, dim=1, keepdim=True)  # (B,1)
            nrm = nrm.view(-1, 1, 1, 1)
            scale = torch.where(nrm <= radius, torch.ones_like(nrm), (radius / (nrm + eps)))
            return z * scale
        else:
            # fallback single tensor
            nrm = z.view(-1).norm()
            if nrm <= radius:
                return z
            return (radius / (nrm + eps)) * z

    def RA_torch(z, gam, max_iter=50, tol=1e-4):
        # z: list of 4 tensors
        u = [t.clone() for t in z]
        for it in range(max_iter):
            Ku  = K_torch(u)
            res = [z[i] - gam * Ku[i] for i in range(len(z))]
            # apply projections on indices 2 and 3 (as in original)
            new = [
                res[0],
                res[1],
                proj_ball_torch(res[2], params.alpha1),
                proj_ball_torch(res[3], params.alpha2),
            ]
            diff = sum((new[i] - u[i]).norm()**2 for i in range(len(new))).sqrt()
            u = new
            if diff.item() < tol:
                break
        return u

    def C(u):
        noisy_4d = noisy_torch.unsqueeze(1)   # (1,1,H,W)
        return [
            u[0] - noisy_4d.to(u[0].device),
            torch.zeros_like(u[1]),
            torch.zeros_like(u[2]),
            torch.zeros_like(u[3]),
        ]

    def const(lambda_n, mu_n, gamma_n, gamma_prev, beta_bar):
        # same formula as main but safe with eps
        denom_a = lambda_n + mu_n
        a_n = mu_n / (denom_a + eps)
        denom_ab = (gamma_prev + eps) * (lambda_n + mu_n)
        ab_n = (gamma_n * mu_n) / (denom_ab + eps)
        th       = (4.0 - gamma_n * beta_bar) * (lambda_n + mu_n) - 2.0 * (lambda_n ** 2)
        th_h     = 2.0 * lambda_n + 2.0 * mu_n - gamma_n * beta_bar * (lambda_n ** 2)
        th_b     = (lambda_n + mu_n) - lambda_n ** 2
        th_tilde = (lambda_n + mu_n) * gamma_n * beta_bar
        return a_n, ab_n, th, th_h, th_b, th_tilde

    def compute_delta_torch(p, x, p_prev, z, z_prev, y, y_prev, u, v,
                            a_n, th, th_h, th_b, gamma, lambda_n, gam_prev, mu_n):
        core  = [
            p[i] - x[i]
            + a_n * (x[i] - p_prev[i])
            + (gamma * params.beta_bar * (lambda_n**2) / (th_h + eps)) * u[i]
            - (2 * th_b / (th + eps)) * v[i]
            for i in range(len(p))
        ]
        term1 = th / 2 * sum(c.norm()**2 for c in core)

        diff_z   = [(z[i] - p[i]) / (gamma + eps) - (z_prev[i] - p_prev[i]) / (gam_prev + eps)
                    for i in range(len(p))]
        diff_pp  = [p[i] - p_prev[i] for i in range(len(p))]
        term2    = 2 * mu_n * gamma * sum((diff_z[i] * diff_pp[i]).sum() for i in range(len(p)))

        diff_py = [(p[i] - y[i]) - (p_prev[i] - y_prev[i]) for i in range(len(p))]
        term3   = (mu_n * gamma * params.beta_bar / 2.0) * sum(d.norm()**2 for d in diff_py)

        result = term1 + term2 + term3
        result = torch.as_tensor(result, dtype=torch.float32, device=device)
        return torch.clamp(result, min=0.0)

    return dict(
        RA = RA_torch,
        C  = C,
        compute_delta = compute_delta_torch,
        const = const,
    )
