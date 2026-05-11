import odl
import numpy as np
import odl.contrib.torch as odl_torch
import torch


# ============================================================
# UTILS
# ============================================================

def ensure_4d(x):
    if x.dim() == 5:
        x = x.squeeze(1)
    if x.dim() == 3:
        x = x.unsqueeze(0)
    return x.float()


# ============================================================
# SETUP COMPLET (COMME TON ORIGINAL)
# ============================================================


def ray(device,size):
    space = odl.uniform_discr(
        [-1, -1], [1, 1],
        [size, size],
        dtype='float32'
    )

    # ============================================================
    # TOMOGRAPHY
    # ============================================================

    angle_partition = odl.uniform_partition(0, np.pi, 60)
    detector_partition = odl.uniform_partition(-1.0, 1.0, size)

    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)

    T = odl.tomo.RayTransform(space, geometry)

    # NORMALIZATION
    T_norm = T.norm(estimate=True)
    T = (1 / T_norm) * T

    A  = odl_torch.OperatorModule(T).to(device)
    AT = odl_torch.OperatorModule(T.adjoint).to(device)
    
    return {
        "A": A,
        "AT": AT,
        "space": space
    }
    
def get_setup(A,AT,space, seed=0, noise_level=0.1, device=None):

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ============================================================
    # SPACE
    # ============================================================

    

    # ============================================================
    # TGV OPERATORS
    # ============================================================

    D  = odl.Gradient(space)
    Dx = odl.PartialDerivative(space, 0)
    Dy = odl.PartialDerivative(space, 1)

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
        out0 = ensure_4d(odl_op(D_adj_layer, u[2]))
        out1 = ensure_4d(-u[2] + odl_op(E_adj_layer, u[3]))
        out2 = -ensure_4d(odl_op(D_layer, u[0])) + u[1]
        out3 = ensure_4d(-odl_op(E_layer, u[1]))
        return [out0, out1, out2, out3]

    # ============================================================
    # DATA GENERATION (SEED DEPENDENT)
    # ============================================================

    np.random.seed(seed)

    phantom = odl.phantom.shepp_logan(space, modified=True)
    x_np = np.asarray(phantom, dtype=np.float32)

    x_true = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).to(device)

    
    noisy = A(x_true)


    sigma = noise_level * torch.max(torch.abs(noisy))
    noisy = noisy + sigma * torch.randn_like(noisy)

    init_state=AT(noisy)



    return dict(
        A=A,
        AT=AT,
        K=K_torch,
        x_true=x_true,
        noisy=noisy,
        initial_state=init_state,
        device=device
    )


# ============================================================
# PARAMETERS (INCHANGÉ)
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
# BUILD FUNCTIONS
# ============================================================

def build_algo_functions(setup, params):

    A = setup['A']
    AT = setup['AT']
    K_torch = setup['K']
    noisy = setup['noisy']
    device = setup['device']

    # ============================================================
    # PROJECTION
    # ============================================================

    def proj_ball_torch(z, alpha):
        return torch.clamp(z, -alpha, alpha)

    # ============================================================
    # RESOLVENT
    # ============================================================

    def RA_torch(z, gam, max_iter=50, tol=1e-4):
        u = [zi.clone() for zi in z]

        for _ in range(max_iter):
            Ku = K_torch(u)

            res = [z[i] - gam * Ku[i] for i in range(4)]

            new = [
                res[0],
                res[1],
                proj_ball_torch(res[2], params.alpha1),
                proj_ball_torch(res[3], params.alpha2),
            ]

            diff = torch.sqrt(sum((new[i] - u[i]).pow(2).sum() for i in range(4)) + 1e-12)

            u = new

            if diff.item() < tol:
                break

        return u

    # ============================================================
    # C(u) 
    # ============================================================

    def C(u):
        x = u[0]
        data_grad = AT(A(x) - noisy)

        return [
            data_grad,
            torch.zeros_like(u[1]),
            torch.zeros_like(u[2]),
            torch.zeros_like(u[3]),
        ]


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

        RA          = RA_torch,
        C           = C,
        compute_delta_torch = compute_delta_torch,
        K=K_torch,
        A=A,
        AT=AT,
        sinogram=noisy,
    )


    