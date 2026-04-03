import odl
import numpy as np
import odl.contrib.torch as odl_torch
import torch

def get_setup(seed=0, noise_level=0.1, device=None):
    size = 16
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    space   = odl.uniform_discr([0, 0], [100, 100], [size, size])
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

    D_layer     = odl_torch.OperatorModule(D)
    D_adj_layer = odl_torch.OperatorModule(D.adjoint)
    E_layer     = odl_torch.OperatorModule(E)
    E_adj_layer = odl_torch.OperatorModule(E.adjoint)

   
    def odl_op(layer, x):
        return layer(x.cpu()).to(x.device)

    def K_torch(u):
        out0 =  odl_op(D_adj_layer, u[2])
        out1 = -u[2] + odl_op(E_adj_layer, u[3])
        out2 = -odl_op(D_layer, u[0]) + u[1]
        out3 = -odl_op(E_layer, u[1])
        return [out0, out1, out2, out3]

    noisy_torch = (torch.tensor(np.asarray(noisy), dtype=torch.float32)
                   .unsqueeze(0)
                   .to(device))

    return dict(
        space=space,
        noisy=noisy_torch,
        K=K_torch,
        device=device,
        size=size,
    )


class Params:
    def __init__(self, lam0=0.9, beta_bar=1.0, gamma0=0.1,
                 alpha1=0.1, alpha2=0.1, zeta=0.9, t=0.5):
        self.lam0     = lam0
        self.beta_bar = beta_bar
        self.gamma0   = gamma0
        self.alpha1   = alpha1
        self.alpha2   = alpha2
        self.zeta     = zeta
        self.t        = t

    def lam(self, n):       return self.lam0 * (1 + n) ** 0.3
    def gamma(self, n):     return self.gamma0
    def mu(self, n):
        l = self.lam(n);    return (1 / self.lam0) * l**2 - l
    def alpha(self, n):
        l = self.lam(n);    return 0 if l == 0 else (l - self.lam0) / l
    def alpha_bar(self, n):
        if n == 0: return 0
        l = self.lam(n);    return (self.gamma(n) / self.gamma(n - 1)) * (l - self.lam0) / l
    def theta(self, n):
        return (4 - self.gamma(n)*self.beta_bar - 2*self.lam0) / self.lam0 * self.lam(n)**2
    def theta_hat(self, n):
        return (2 - self.lam0*self.gamma(n)*self.beta_bar) / self.lam0 * self.lam(n)**2
    def theta_bar(self, n):
        return (1 - self.lam0) / self.lam0 * self.lam(n)**2
    def theta_tilde(self, n):
        return self.gamma(n)*self.beta_bar / self.lam0 * self.lam(n)**2


def build_algo_functions(setup, params):
    noisy_torch = setup['noisy']
    K_torch     = setup['K']
    # CORRECTION : device extrait du tenseur noisy pour tous les nouveaux tenseurs
    dev         = noisy_torch.device

    def proj_ball_torch(z, alpha):
        nrm = z.norm()
        return torch.where(nrm <= alpha, z, (alpha / nrm) * z)

    def RA_torch(z, gam, max_iter=50, tol=1e-4):
        u = z
        for _ in range(max_iter):
            Ku  = K_torch(u)
            res = [z[i] - gam * Ku[i] for i in range(4)]
            new = [
                res[0],
                res[1],
                proj_ball_torch(res[2], params.alpha1),
                proj_ball_torch(res[3], params.alpha2),
            ]
            diff = ((new[0]-u[0]).norm()**2 + (new[1]-u[1]).norm()**2 +
                    (new[2]-u[2]).norm()**2 + (new[3]-u[3]).norm()**2).sqrt()
            u = new
            if diff.item() < tol:
                break
        return u

    def C(u):
        return [
            u[0] - noisy_torch,
            torch.zeros_like(u[1]),
            torch.zeros_like(u[2]),
            torch.zeros_like(u[3]),
        ]

    def compute_delta_torch(p, x, p_prev, z, z_prev, y, y_prev, u, v, n):
        th   = params.theta(n)
        th_h = params.theta_hat(n)
        th_b = params.theta_bar(n)
        gam  = params.gamma(n)
        mu_n = params.mu(n)
        a_n  = params.alpha(n)

        core = [
            p[i] - x[i]
            + a_n * (x[i] - p_prev[i])
            + (gam * params.beta_bar * params.lam(n)**2 / th_h) * u[i]
            - (2 * th_b / th) * v[i]
            for i in range(4)
        ]
        term1 = th / 2 * sum(c.norm()**2 for c in core)

        gam_prev = params.gamma(n - 1) if n > 0 else gam
        diff_z   = [(z[i] - p[i]) / gam - (z_prev[i] - p_prev[i]) / gam_prev
                    for i in range(4)]
        diff_pp  = [p[i] - p_prev[i] for i in range(4)]
        term2    = 2 * mu_n * gam * sum(
                    (diff_z[i] * diff_pp[i]).sum() for i in range(4))

        diff_py  = [(p[i] - y[i]) - (p_prev[i] - y_prev[i]) for i in range(4)]
        term3    = (mu_n * gam * params.beta_bar / 2.0) * sum(
                    d.norm()**2 for d in diff_py)

        return torch.clamp(term1 + term2 + term3, min=0.0)

    def compute_deviations_torch(p, x, p_prev, z, z_prev, y, y_prev, u, v, n, dir_u, dir_v):
        delta  = compute_delta_torch(p, x, p_prev, z, z_prev, y, y_prev, u, v, n)
        budget = params.zeta * delta

        lam_mu  = params.lam(n + 1) + params.mu(n + 1)
        coeff_u = lam_mu * (params.theta_tilde(n + 1) / params.theta_hat(n + 1))
        coeff_v = lam_mu * (params.theta_hat(n + 1)   / params.theta(n + 1))

        norm_u = torch.sqrt(sum(d.norm()**2 for d in dir_u) + 1e-12)
        norm_v = torch.sqrt(sum(d.norm()**2 for d in dir_v) + 1e-12)

        max_u = torch.sqrt(torch.clamp(params.t * budget / coeff_u,         min=0.0))
        max_v = torch.sqrt(torch.clamp((1 - params.t) * budget / coeff_v,   min=0.0))

        u_new = [max_u / norm_u * d for d in dir_u]
        v_new = [max_v / norm_v * d for d in dir_v]
        return u_new, v_new

    return dict(
        lam=params.lam, gamma=params.gamma, mu=params.mu,
        alpha=params.alpha, alpha_bar=params.alpha_bar,
        theta=params.theta, theta_hat=params.theta_hat,
        theta_bar=params.theta_bar, theta_tilde=params.theta_tilde,
        RA=RA_torch, C=C,
        compute_delta_torch=compute_delta_torch,   
        compute_deviations=compute_deviations_torch,
    )