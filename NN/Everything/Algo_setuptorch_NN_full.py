import odl
import numpy as np
import odl.contrib.torch as odl_torch
import torch

# ============================================================
# SETUP PROBLEM in ODL + Converting in torch
# ============================================================

def get_setup(seed=0, noise_level=0.1, device=None):
    size=16
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


    D_layer     = odl_torch.OperatorModule(D)
    D_adj_layer = odl_torch.OperatorModule(D.adjoint)
    E_layer     = odl_torch.OperatorModule(E)
    E_adj_layer = odl_torch.OperatorModule(E.adjoint)

    def odl_op(layer, x):
        
        dev = x.device
        return layer(x.cpu()).to(dev)

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


# ============================================================
# PARAMETERS
# ============================================================

class Params:

    def __init__(self,
                 lam0=1.0,
                 beta_bar=1.0,
                 gamma0=0.1,
                 alpha1=0.1,
                 alpha2=0.1,
                 zeta=0.9,
                 ):

        self.lam0     = lam0
        self.beta_bar = beta_bar
        self.gamma0   = gamma0
        self.alpha1   = alpha1
        self.alpha2   = alpha2
        self.zeta     = zeta






# ============================================================
# BUILD ALGORITHM FUNCTIONS
# ============================================================

def build_algo_functions(setup, params):

    noisy_torch = setup['noisy']   # (1, 32, 32) on device
    K_torch     = setup['K']
    device      = setup['device']
    beta_bar=params.beta_bar
    
    
    def proj_ball_torch(z, radius):
        nrm = z.norm()
        return torch.where(nrm <= radius, z, (radius / nrm) * z)

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
            diff = sum((new[i] - u[i]).norm()**2 for i in range(4)).sqrt()
            u = new
            if diff.item() < tol:
                break
        return u

    def C(u):
        """Compute constraint operator."""
        noisy_4d = noisy_torch.unsqueeze(1)   
        return [
            u[0] - noisy_4d,
            torch.zeros_like(u[1]),
            torch.zeros_like(u[2]),
            torch.zeros_like(u[3]),
        ]



    def alpha(lambda_n, mu_n):
        denom = lambda_n + mu_n
        return 0.0 if denom == 0.0 else mu_n / denom


    def alpha_bar(lambda_n, mu_n, gamma_n, gamma_prev):
        denom = (gamma_prev or 0.0) * (lambda_n + mu_n)
        return 0.0 if denom == 0.0 else (gamma_n * mu_n) / denom


    def theta(lambda_n, mu_n, gamma_n, beta_bar):
        return (4.0 - gamma_n * beta_bar) * (lambda_n + mu_n) - 2.0 * (lambda_n ** 2)

    def theta_hat(lambda_n, mu_n, gamma_n, beta_bar):
        return 2.0 * lambda_n + 2.0 * mu_n - gamma_n * beta_bar * (lambda_n ** 2)


    def theta_bar(lambda_n, mu_n) :
        return (lambda_n + mu_n) - (lambda_n ** 2)

    def theta_tilde(lambda_n, mu_n, gamma_n, beta_bar) :
        return (lambda_n + mu_n) * gamma_n * beta_bar

    def const(lambda_n, mu_n, gamma_n, gamma_prev, beta_bar):
        eps = 1e-12

        # alpha
        denom_a = lambda_n + mu_n
        a_n = mu_n / (denom_a + eps)

        # alpha_bar
        denom_ab = gamma_prev * (lambda_n + mu_n)
        ab_n = (gamma_n * mu_n) / (denom_ab + eps)

        # thetas
        th       = (4.0 - gamma_n * beta_bar) * (lambda_n + mu_n) - 2.0 * lambda_n ** 2
        th_h     = 2.0 * lambda_n + 2.0 * mu_n - gamma_n * beta_bar * lambda_n ** 2
        th_b     = (lambda_n + mu_n) - lambda_n ** 2
        th_tilde = (lambda_n + mu_n) * gamma_n * beta_bar

        return a_n, ab_n, th, th_h, th_b, th_tilde
    
    
    def compute_delta_torch(p, x, p_prev, z, z_prev, y, y_prev, u, v,a_n,th ,th_h,th_b,gamma,lambda_n,gam_prev,mu_n):


        # term1
        core  = [p[i] - x[i]
                 + a_n * (x[i] - p_prev[i])
                 + (gamma * params.beta_bar * lambda_n**2 / th_h) * u[i]
                 - (2 * th_b / th) * v[i]
                 for i in range(4)]
        term1 = th / 2 * sum(c.norm()**2 for c in core)

        # term2
     
        diff_z   = [(z[i] - p[i]) / gamma - (z_prev[i] - p_prev[i]) / gam_prev
                    for i in range(4)]
        diff_pp  = [p[i] - p_prev[i] for i in range(4)]
        term2    = 2 * mu_n * gamma * sum(
                       (diff_z[i] * diff_pp[i]).sum() for i in range(4))

        # term3
        diff_py = [(p[i] - y[i]) - (p_prev[i] - y_prev[i]) for i in range(4)]
        term3   = (mu_n * gamma * params.beta_bar / 2.0) * sum(
                      d.norm()**2 for d in diff_py)
        result=term1+term2+term3
        result = torch.as_tensor(result, dtype=torch.float32, device=device)
        return torch.clamp(result, min=0.0)
        
    return dict(

        RA          = RA_torch,
        C           = C,
        compute_delta_torch = compute_delta_torch,
        const=const,
    )


    