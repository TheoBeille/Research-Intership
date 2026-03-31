import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib with LaTeX
plt.rcParams['text.usetex'] = True

from Algo_setuptorch import get_setup, Params, build_algo_functions

# ============================================================
#  SETUP
# ============================================================

params = Params()

TRAIN_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
TEST_SEEDS  = [8, 9]

size     = 8
SHAPES   = [
    (1, 1, size, size),
    (1, 2, size, size),
    (1, 2, size, size),
    (1, 4, size, size),
]
N_CH     = sum(s[1] for s in SHAPES)   # 9
N_BLOCKS = len(SHAPES)                  # 4
DIM      = N_CH * size * size           # dimension aplatie d'un état = 576

T=50



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")


def load_sample(seed, noise_level=0.1):
    setup       = get_setup(seed=seed, noise_level=noise_level, device=device)
    setup_clean = get_setup(seed=seed, noise_level=0.0,         device=device)
    functions   = build_algo_functions(setup, params)
    noisy = setup['noisy'].unsqueeze(1)
    clean = setup_clean['noisy'].unsqueeze(1)
    return noisy, clean, functions

print("Loading data...")
train_data = [load_sample(s) for s in TRAIN_SEEDS]
test_data  = [load_sample(s) for s in TEST_SEEDS]
print(f"  Train: {len(train_data)} images  |  Test: {len(test_data)} images\n")

functions     = train_data[0][2]
RA            = functions['RA']
C             = functions['C']
compute_delta = functions['compute_delta_torch']
gamma         = functions['gamma']
alpha_fn      = functions['alpha']
alpha_bar_fn  = functions['alpha_bar']








def pack(blocks):

    fixed = []
    for b in blocks:
        b = b.float()
        # needs to be in 4D (B, C, H, W)
        if b.dim() == 5:
            b = b.squeeze(0) 
        if b.dim() == 3:
            b = b.unsqueeze(0)   
        elif b.dim() == 2:
            b = b.unsqueeze(0).unsqueeze(0)  
        fixed.append(b)
   
    for b in fixed:
        assert b.dim() == 4, f"Bloc dimension error : {b.shape}"
    return torch.cat(fixed, dim=1)

def unpack(tensor):
    out, c = [], 0
    for s in SHAPES:
        out.append(tensor[:, c:c + s[1]])
        c += s[1]
    return out


# ============================================================
#  DEVIATIONNET — MLP hidden=256
# ============================================================

class DeviationNet(nn.Module):
    
    def __init__(self, hidden=64):
        super().__init__()

        in_features  = 4 * N_CH * size * size   
        out_features = 2 * N_CH * size * size   

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_bl, p_prev_bl, y_prev_bl, z_prev_bl):
        inp = torch.cat([
            pack(x_bl),
            pack(p_prev_bl),
            pack(y_prev_bl),
            pack(z_prev_bl),
        ], dim=1)                                # (B, 36, H, W)

        B   = inp.shape[0]
        inp = inp.view(B, -1)                    # (B, 36864)
        out = self.net(inp)                      # (B, 18432)
        out = out.view(B, 2 * N_CH, size, size)  # (B, 18, 32, 32)

        u_raw = unpack(out[:, :N_CH])
        v_raw = unpack(out[:, N_CH:])
        return u_raw, v_raw


# ============================================================
#  SAFEGUARDING LAYER
# ============================================================

class SafeguardingLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.p = params

    def forward(self, u_raw_bl, v_raw_bl, delta, n):
        p   = self.p
        lam = p.lam(n + 1)
        mu  = p.mu(n + 1)
        lpm = lam + mu

        c_u = lpm * p.theta_tilde(n + 1) / p.theta_hat(n + 1)
        c_v = lpm * p.theta_hat(n + 1)   / p.theta(n + 1)

        norm_u_sq = sum(u.pow(2).sum() for u in u_raw_bl)
        norm_v_sq = sum(v.pow(2).sum() for v in v_raw_bl)

        Q      = c_u * norm_u_sq + c_v * norm_v_sq + 1e-12
        budget = p.zeta * delta.clamp(min=0.0)
        alpha  = torch.sqrt((budget / Q).clamp(max=1.0))

        u_bl = [alpha * u for u in u_raw_bl]
        v_bl = [alpha * v for v in v_raw_bl]
        return u_bl, v_bl


# ============================================================
#  UNE ITERATION DE L'ALGORITHME
# ============================================================

def one_step(x, y_prev, p_prev, z_prev, u, v, n):
    p    = params
    gam  = gamma(n)
    a_n  = alpha_fn(n)
    ab_n = alpha_bar_fn(n)
    th_b = p.theta_bar(n)
    th_h = p.theta_hat(n)
    lam  = p.lam(n)

    # Step 5
    y = [x[i] + a_n * (y_prev[i] - x[i]) + u[i]
         for i in range(N_BLOCKS)]

    # Step 6
    z = [
        x[i]
        + a_n  * (p_prev[i] - x[i])
        + ab_n * (z_prev[i] - p_prev[i])
        + (th_b * gam * p.beta_bar / th_h) * u[i]
        + v[i]
        for i in range(N_BLOCKS)
    ]

    # Step 7 : forward-backward
    Cy          = C(y)
    z_minus_gCy = [z[i] - gam * Cy[i] for i in range(N_BLOCKS)]
    pr          = RA(z_minus_gCy, gam)

    # Step 8
    x_new = [
        x[i] + lam * (pr[i] - z[i]) + ab_n * lam * (z_prev[i] - p_prev[i])
        for i in range(N_BLOCKS)
    ]

    residual = torch.sqrt(
        sum((pr[i] - y[i]).pow(2).sum() for i in range(N_BLOCKS)) + 1e-12
    )

    return x_new, y, pr, z, residual


# ============================================================
#  MODELE UNROLLED
# ============================================================

class UnrolledFBS(nn.Module):
    def __init__(self, T=20):
        super().__init__()
        self.T        = T
        self.dev_net  = DeviationNet(hidden=256)
        self.sg_layer = SafeguardingLayer(params)

    def _init_state(self, noisy):
        dev = noisy.device
        x = [
            noisy.clone(),
            torch.zeros(SHAPES[1], device=dev),
            torch.zeros(SHAPES[2], device=dev),
            torch.zeros(SHAPES[3], device=dev),
        ]
        p_prev = [t.clone() for t in x]
        y_prev = [t.clone() for t in x]
        z_prev = [t.clone() for t in x]
        u      = [torch.zeros_like(t) for t in x]
        v      = [torch.zeros_like(t) for t in x]
        return x, y_prev, p_prev, z_prev, u, v

    def forward(self, noisy):
        x, y_prev, p_prev, z_prev, u, v = self._init_state(noisy)
        residuals = []

        for n in range(self.T):
           
            x_new, y, p, z, res = one_step(
                x, y_prev, p_prev, z_prev, u, v, n
            )

           
            delta = compute_delta(
                p, x_new, p_prev, z, z_prev, y, y_prev, u, v, n
            )

       
            u_raw, v_raw = self.dev_net(x_new, p, y, z)

    
            u_new, v_new = self.sg_layer(u_raw, v_raw, delta, n)

  
            x, y_prev, p_prev, z_prev = x_new, y, p, z
            u, v = u_new, v_new

            residuals.append(res)

        return p_prev, residuals



def convergence_loss(residuals):
    T       = len(residuals)
    weights = torch.linspace(float(T), 1.0, T, device=residuals[0].device)
    safe    = [r.clamp(max=1e6) for r in residuals]
    r0      = safe[0].detach().clamp(min=1e-8)
    norm    = [r / r0 for r in safe]
    return sum(w * r for w, r in zip(weights, norm)) / weights.sum()


# ============================================================
#  TRAINING END-TO-END
# ============================================================

def train(n_epochs=200, lr=1e-4, T=50):
    train_loss_hist, test_loss_hist = [], []
    epoch_global = 0
    model     = UnrolledFBS(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Training on {n_epochs} epochs...\n")

    loss_history = []

    for epoch in range(n_epochs):
            model.train()
            epoch_loss, valid_count = 0.0, 0

            for noisy, _, _ in train_data:
                optimizer.zero_grad()
                _, residuals = model(noisy)
                loss = convergence_loss(residuals)
                if not torch.isfinite(loss):
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss  += loss.item()
                valid_count += 1

            epoch_loss = epoch_loss / valid_count if valid_count > 0 else float('nan')
            train_loss_hist.append(epoch_loss)

            model.eval()
            with torch.no_grad():
                test_loss = sum(
                    convergence_loss(model(noisy)[1]).item()
                    for noisy, _, _ in test_data
                ) / len(test_data)
            test_loss_hist.append(test_loss)

            if epoch % 20 == 0:
                print(f"  Epoch {epoch_global:4d} | Train={epoch_loss:.6f} | Test={test_loss:.6f}")
            epoch_global += 1

    print("\nEntraînement terminé.")
    return model, train_loss_hist, test_loss_hist


# ============================================================
#  COMPARAISON : ZERO  / APPRIS
# ============================================================

def run_zero(T=20):

    x = [
        noisy.clone(),
        torch.zeros(SHAPES[1], device=device),
        torch.zeros(SHAPES[2], device=device),
        torch.zeros(SHAPES[3], device=device),
    ]
    p_prev = [t.clone() for t in x]
    y_prev = [t.clone() for t in x]
    z_prev = [t.clone() for t in x]
    u = [torch.zeros_like(t) for t in x]
    v = [torch.zeros_like(t) for t in x]

    residuals = []
    with torch.no_grad():
        for n in range(1000):
            x, y_prev, p_prev, z_prev, res = one_step(
                x, y_prev, p_prev, z_prev, u, v, n
            )
            residuals.append(res.item())
            print(res.item())
            if res.item() < 1e-3:
                print("converged")
    
                break
    return residuals



def run_learned(model, T=20):
   
    model.eval()
    with torch.no_grad():
        _, residuals = model(noisy)
    return [r.item() for r in residuals]


# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    T = 20

    # Entrainement
    model, loss_history = train(n_epochs=200, lr=1e-4, T=T)

    # Comparaison
    print("\nComparaison des 3 methodes...")
    res_zero    = run_zero(T)

    res_learned = run_learned(model, T)

    print(f"\nResidu final :")
    print(f"  Zero    : {res_zero[-1]:.6f}")

    print(f"  Appris  : {res_learned[-1]:.6f}")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Courbe d'entrainement
    axes[0].semilogy(loss_history)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Residu final (log)")
    axes[0].set_title("Courbe d'entrainement")
    axes[0].grid(True, alpha=0.3)

    # Comparaison des residus
    axes[1].semilogy(res_zero,    label="Zero deviations",    linewidth=2)
    axes[1].semilogy(res_random,  label="Aleatoires",         linewidth=2, linestyle='--')
    axes[1].semilogy(res_learned, label="Apprises (MLP)",     linewidth=2, linestyle='-.')
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Residu ||p - y|| (log)")
    axes[1].set_title("Comparaison des 3 methodes")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150)
    plt.show()
    print("Figure sauvegardee : comparison.png")