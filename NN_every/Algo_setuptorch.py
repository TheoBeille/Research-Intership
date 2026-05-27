import odl
import numpy as np
import odl.contrib.torch as odl_torch
import torch
from odl.phantom.noise import white_noise
from odl.operator.pspace_ops import (
    ProductSpaceOperator,
    ComponentProjection,
    ReductionOperator,
    BroadcastOperator,
)

# ============================================================
# SETUP PROBLEM in ODL + Converting in torch
# ============================================================

def ensure_4d(x):
    """
    Ensure tensor is [B, C, H, W]
    """
    if x.dim() == 5:
        x = x.squeeze(1)

    if x.dim() == 3:
        x = x.unsqueeze(0)

    return x.float()



def get_setup(size,seed=0, noise_level=0.1, device=None):
   
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    U = odl.uniform_discr([0, 0], [100, 100], [size, size])

    A = odl.IdentityOperator(U)

    phantom = odl.phantom.shepp_logan(U, modified=True)
    noise= noise_level * odl.phantom.white_noise(U, seed=seed)
    data   = phantom + noise

    D  = odl.Gradient(U, method='forward', pad_mode='symmetric')
    Dx = odl.PartialDerivative(U, 0, method='forward', pad_mode='symmetric')
    Dy = odl.PartialDerivative(U, 1, method='forward', pad_mode='symmetric')
    V  = D.range
    

    E = ProductSpaceOperator(
        [
            [Dx, 0],
            [0, Dy],
            [0.5 * Dy, 0.5 * Dx],
        ]
    )


    domain = odl.ProductSpace(U, V)

    D_layer     = odl_torch.OperatorModule(D)
    D_adj_layer = odl_torch.OperatorModule(D.adjoint)
    E_layer     = odl_torch.OperatorModule(E)
    E_adj_layer = odl_torch.OperatorModule(E.adjoint)

    def odl_op(layer, x):
        
        dev = x.device
        return layer(x.cpu()).to(dev)

    def K_torch(u):
        out0 = ensure_4d(odl_op(D_adj_layer, u[2]))
        out0 = out0.float()
        
        out1 = ensure_4d(-u[2] + odl_op(E_adj_layer, u[3]))
        out1 = out1.float()
        
        Dop = ensure_4d(odl_op(D_layer, u[0]))
        out2 = -Dop + u[1]
        
        out3 = ensure_4d(-odl_op(E_layer, u[1]))

        out3 = out3.float()
        
        return [out0, out1, out2, out3]


    data = (torch.tensor(np.asarray(data), dtype=torch.float32)
                   .unsqueeze(0)
                   .to(device))

    return dict(
        space=U,
        initial_state=data,
        K=K_torch,
        device=device,
        A=A,
        D=D,
        E=E,
        domain=domain,
        D_layer=D_layer,
        D_adj_layer=D_adj_layer,
        E_layer=E_layer,
        E_adj_layer=E_adj_layer,
    )


# ============================================================
# PARAMETERS
# ============================================================

class Params:

    def __init__(self,
                 lam0=0.9,
                 beta_bar=1.0,
                 gamma0=0.1,
                 alpha1=0.1,
                 alpha2=0.1,
                 zeta=0.9,
                 size=128,
                 ):

        self.lam0     = lam0
        self.beta_bar = beta_bar
        self.gamma0   = gamma0
        self.alpha1   = alpha1
        self.alpha2   = alpha2
        self.zeta     = zeta
        self.size=size


    def lam(self, n):
        return self.lam0 * (1 + n) ** 0.3
        #return self.lam0 

    def gamma(self, n):
        return self.gamma0

    def mu(self, n):
        l = self.lam(n)
        return (1 / self.lam0) * l**2 - l

    def alpha(self, n):
        l = self.lam(n)
        return 0 if l == 0 else (l - self.lam0) / l

    def alpha_bar(self, n):
       
        gam  = self.gamma(n)
        gam_prev = self.gamma(n-1) if n > 0 else gam

        if n == 0:
            return 0
        l = self.lam(n)
        return (gam /gam_prev) * (l - self.lam0) / l

    def theta(self, n):
        
 
        gam  = self.gamma(n)
        
        return (4 - gam * self.beta_bar - 2 * self.lam0) \
               / self.lam0 * self.lam(n)**2

    def theta_hat(self, n):
            
    
        gam  = self.gamma(n)
            
        
            
        return (2 - self.lam0 * gam * self.beta_bar) \
               / self.lam0 * self.lam(n)**2

    def theta_bar(self, n):
        return (1 - self.lam0) / self.lam0 * self.lam(n)**2

    def theta_tilde(self, n):
  
        gam  = self.gamma(n)

        return gam * self.beta_bar \
               / self.lam0 * self.lam(n)**2


# ============================================================
# BUILD ALGORITHM FUNCTIONS
# ============================================================

def build_algo_functions(setup, params):
    x0 = setup['initial_state']
    device = setup['device']
    K_torch = setup['K']

    D_layer     = setup['D_layer']
    D_adj_layer = setup['D_adj_layer']
    E_layer     = setup['E_layer']
    E_adj_layer = setup['E_adj_layer']

    def odl_op(layer, x):
        return layer(x.cpu()).to(x.device)

    def grad_torch(u):
        return ensure_4d(odl_op(D_layer, u)).float()

    def gradT_torch(p):
        return ensure_4d(odl_op(D_adj_layer, p)).float()

    def E_torch(w):
        return ensure_4d(odl_op(E_layer, w)).float()

    def ET_torch(q):
        return ensure_4d(odl_op(E_adj_layer, q)).float()

    def proj_linf_ball(z, alpha):
        return torch.clamp(z, -alpha, alpha)

    def resolvent_A(z, gamma, max_iter=500, tol=1e-6):
        u = z[0].clone()
        w = z[1].clone()
        p = z[2].clone()
        q = z[3].clone()

        for _ in range(max_iter):
            # dual projections from current primal variables
            p_arg = z[2] + gamma * (grad_torch(u) - w)
            q_arg = z[3] + gamma * E_torch(w)

            p_new = proj_linf_ball(p_arg, params.alpha1)
            q_new = proj_linf_ball(q_arg, params.alpha2)

            # primal updates from current dual variables
            u_new = z[0] - gamma * gradT_torch(p_new)
            w_new = z[1] + gamma * p_new - gamma * ET_torch(q_new)

            res = torch.sqrt(
                (u_new - u).pow(2).sum()
                + (w_new - w).pow(2).sum()
                + (p_new - p).pow(2).sum()
                + (q_new - q).pow(2).sum()
            )

            u, w, p, q = u_new, w_new, p_new, q_new

            if res.item() < tol:
                break

        return [u, w, p, q]

    def C(u):
        noisy_4d = x0.unsqueeze(1)
        return [
            u[0] - noisy_4d,
            torch.zeros_like(u[1]),
            torch.zeros_like(u[2]),
            torch.zeros_like(u[3]),
        ]


    def kkt_residual(x):
        """
        Compute the KKT / monotone inclusion residual:

            0 ∈ A(x) + C(x)

        for the TGV² problem.

        Returns:
            scalar residual norm
        """

        u, w, p, q = x

        noisy_4d = x0.unsqueeze(1)

        # K^T(Ku - y) + ∇^T p
        r1 = (u - noisy_4d) + gradT_torch(p)

        # -p + E^T q
        r2 = -p + ET_torch(q)

        # 0 ∈ ∇u - w + N_{||·||∞≤α1}(p)

        #
        r3 = p - proj_linf_ball(
            p + grad_torch(u) - w,
            params.alpha1
        )

        # 0 ∈ Ew + N_{||·||∞≤α0}(q)

        #
        r4 = q - proj_linf_ball(
            q + E_torch(w),
            params.alpha2
        )

        return r1,r2,r3,r4

    def kkt_residual_norm(x):

        r1, r2, r3, r4 = kkt_residual(x)

        return torch.sqrt(
            r1.pow(2).sum()
            + r2.pow(2).sum()
            + r3.pow(2).sum()
            + r4.pow(2).sum()
        )


    def compute_delta_torch(p, x, p_prev, z, z_prev, y, y_prev, u, v, n):
        
  
        gam  = params.gamma(n)
        gam_prev = params.gamma(n-1) if n > 0 else gam

            
        th   = params.theta(n)
        th_h = params.theta_hat(n)
        th_b = params.theta_bar(n)
        mu_n = params.mu(n)
        a_n  = params.alpha(n)

        # term1
        core  = [p[i] - x[i]
                 + a_n * (x[i] - p_prev[i])
                 + (gam * params.beta_bar * params.lam(n)**2 / th_h) * u[i]
                 - (2 * th_b / th) * v[i]
                 for i in range(4)]
        term1 = th / 2 * sum(c.norm()**2 for c in core)


        diff_z   = [(z[i] - p[i]) / gam - (z_prev[i] - p_prev[i]) / gam_prev
                    for i in range(4)]
        diff_pp  = [p[i] - p_prev[i] for i in range(4)]
        term2    = 2 * mu_n * gam * sum(
                       (diff_z[i] * diff_pp[i]).sum() for i in range(4))

        # term3
        diff_py = [(p[i] - y[i]) - (p_prev[i] - y_prev[i]) for i in range(4)]
        term3   = (mu_n * gam * params.beta_bar / 2.0) * sum(
                      d.norm()**2 for d in diff_py)
        result=term1+term2+term3
        result = torch.as_tensor(result, dtype=torch.float32, device=device)
        return torch.clamp(result, min=0.0)
        
    return dict(

        RA=resolvent_A,
        C=C,
        K=K_torch,
        grad=grad_torch,
        gradT=gradT_torch,
        E=E_torch,
        ET=ET_torch,
        compute_delta_torch = compute_delta_torch,
        kkt_residual=kkt_residual,
        kkt_residual_norm=kkt_residual_norm,

    )


    