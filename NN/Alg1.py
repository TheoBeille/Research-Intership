import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.fontset'] = 'cm'

from Algo_setuptorch import get_setup, Params, build_algo_functions

# ============================================================
# 1. Configuration
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

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

T_BASE    = 360
T_COMPARE = 20
T_TRAIN   = 100

# ============================================================
# 2. Chargement des données
# ============================================================

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

# ============================================================
# 3. Utilitaires shape
# ============================================================

def ensure_4d(t):
    while t.dim() > 4: t = t.squeeze(0)
    while t.dim() < 4: t = t.unsqueeze(0)
    return t

def fix_blocks(blocks):
    return [ensure_4d(b.float()) for b in blocks]

def pack(blocks):
    return torch.cat(fix_blocks(blocks), dim=1)

def unpack(tensor):
    tensor = ensure_4d(tensor)
    out, c = [], 0
    for s in SHAPES:
        out.append(tensor[:, c:c + s[1]])
        c += s[1]
    return out

# ============================================================
# 4. DeviationNet — MLP
# ============================================================

class DeviationNet(nn.Module):

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * DIM, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2 * DIM),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_bl, p_bl, y_bl, z_bl):
        inp = torch.cat([
            pack(x_bl).flatten(),
            pack(p_bl).flatten(),
            pack(y_bl).flatten(),
            pack(z_bl).flatten(),
        ])
        inp = torch.nan_to_num(inp, nan=0.0, posinf=1e4, neginf=-1e4)
        out = self.net(inp)                          # (2 * DIM,)
        dir_u = out[:DIM].view(1, N_CH, size, size)
        dir_v = out[DIM:].view(1, N_CH, size, size)
        return unpack(dir_u), unpack(dir_v)

# ============================================================
# 5. SafeguardingLayer
# ============================================================

class SafeguardingLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.p = params

    def forward(self, dir_u_bl, dir_v_bl, delta, n):
        p   = self.p
        lpm = p.lam(n + 1) + p.mu(n + 1)
        c_u = lpm * p.theta_tilde(n + 1) / p.theta_hat(n + 1)
        c_v = lpm * p.theta_hat(n + 1)   / p.theta(n + 1)

        Q      = (c_u * sum(u.pow(2).sum() for u in dir_u_bl)
                + c_v * sum(v.pow(2).sum() for v in dir_v_bl)) + 1e-8
        budget = p.zeta * delta.clamp(min=0.0) + 1e-8
        scale  = torch.sqrt((budget / Q).clamp(max=1.0))

        return [scale * u for u in dir_u_bl], [scale * v for v in dir_v_bl]

# ============================================================
# 6. One step
# ============================================================

def one_step(x, y_prev, p_prev, z_prev, u, v, n):
    p    = params
    gam  = gamma(n)
    a_n  = alpha_fn(n)
    ab_n = alpha_bar_fn(n)
    th_b = p.theta_bar(n)
    th_h = p.theta_hat(n)
    lam  = p.lam(n)

    y = [x[i] + a_n * (y_prev[i] - x[i]) + u[i] for i in range(N_BLOCKS)]
    z = [x[i] + a_n  * (p_prev[i] - x[i])
              + ab_n * (z_prev[i] - p_prev[i])
              + (th_b * gam * p.beta_bar / th_h) * u[i]
              + v[i] for i in range(N_BLOCKS)]

    Cy          = fix_blocks(C(y))
    z_minus_gCy = [z[i] - gam * Cy[i] for i in range(N_BLOCKS)]
    pr          = fix_blocks(RA(z_minus_gCy, gam))

    x_new = fix_blocks([
        x[i] + lam * (pr[i] - z[i]) + ab_n * lam * (z_prev[i] - p_prev[i])
        for i in range(N_BLOCKS)
    ])

    sq_sum   = sum((x_new[i] - x[i]).pow(2).sum() for i in range(N_BLOCKS))
    sq_sum   = torch.nan_to_num(sq_sum, nan=0.0, posinf=1e8)
    residual = torch.sqrt(sq_sum + 1e-8)

    return x_new, y, pr, z, residual

# ============================================================
# 7. Initial Sate
# ============================================================

def init_state(noisy):
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

# ============================================================
# 8. UnrolledFBS
# ============================================================

class UnrolledFBS(nn.Module):
    def __init__(self, T=T_TRAIN):
        super().__init__()
        self.T        = T
        self.dev_net  = DeviationNet(hidden=128)
        self.sg_layer = SafeguardingLayer(params)

    def forward(self, noisy):
        x, y_prev, p_prev, z_prev, u, v = init_state(noisy)
        residuals = []
        for n in range(self.T):
            x_new, y, p, z, res = one_step(x, y_prev, p_prev, z_prev, u, v, n)
            delta        = compute_delta(p, x_new, p_prev, z, z_prev, y, y_prev, u, v, n)
            dir_u, dir_v = self.dev_net(x_new, p, y, z)
            u_new, v_new = self.sg_layer(dir_u, dir_v, delta, n)
            x, y_prev, p_prev, z_prev = x_new, y, p, z
            u, v = u_new, v_new
            residuals.append(res)
        return x[0], residuals

# ============================================================
# 9. Loss 
# ============================================================

def convergence_loss(residuals):
    T       = len(residuals)
    weights = torch.linspace(float(T), 1.0, T, device=residuals[0].device)
    safe    = [r.clamp(max=1e6) for r in residuals]
    r0      = safe[0].detach().clamp(min=1e-8)
    norm    = [r / r0 for r in safe]
    return sum(w * r for w, r in zip(weights, norm)) / weights.sum()

# ============================================================
# 10. Entraînement avec curriculum
# ============================================================

def train(lr=1e-4):
    model     = UnrolledFBS(T=T_TRAIN).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Paramètres : {n_params:,}\n")

    curriculum = [
        (60,  20),
        (80,  50),
        (100, 100),
    ]

    train_loss_hist, test_loss_hist = [], []
    epoch_global = 0

    for phase, (n_ep, t_phase) in enumerate(curriculum):
        model.T = t_phase
        print(f"--- Phase {phase+1} : T={t_phase}, {n_ep} époques ---")

        for epoch in range(n_ep):
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
# 11. Comparaison
# ============================================================

def run_zero(noisy, T=T_COMPARE):
    x, y_prev, p_prev, z_prev, u, v = init_state(noisy)
    residuals = []
    with torch.no_grad():
        for n in range(T):
            x, y_prev, p_prev, z_prev, res = one_step(x, y_prev, p_prev, z_prev, u, v, n)
            residuals.append(res.item())
    return x[0], residuals

def run_learned(model, noisy, T=T_COMPARE):
    model.eval()
    x, y_prev, p_prev, z_prev, u, v = init_state(noisy)
    residuals = []
    with torch.no_grad():
        for n in range(T):
            x_new, y, p, z, res = one_step(x, y_prev, p_prev, z_prev, u, v, n)
            delta        = compute_delta(p, x_new, p_prev, z, z_prev, y, y_prev, u, v, n)
            dir_u, dir_v = model.dev_net(x_new, p, y, z)
            u, v         = model.sg_layer(dir_u, dir_v, delta, n)
            x, y_prev, p_prev, z_prev = x_new, y, p, z
            residuals.append(res.item())
    return x[0], residuals

# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":

    model, train_loss, test_loss = train(lr=1e-4)

    print(f"\nComparaison sur TEST ({T_COMPARE} itérations)...")
    all_zero, all_learned = [], []
    for noisy, _, _ in test_data:
        _, r_z = run_zero(noisy,           T=T_COMPARE)
        _, r_l = run_learned(model, noisy, T=T_COMPARE)
        all_zero.append(r_z)
        all_learned.append(r_l)

    def pad(lst, length):
        return [r + [r[-1]] * (length - len(r)) for r in lst]

    res_zero    = np.mean(pad(all_zero,    T_COMPARE), axis=0)
    res_learned = np.mean(pad(all_learned, T_COMPARE), axis=0)

    print(f"  Zéro   résidu final : {res_zero[-1]:.6f}")
    print(f"  Appris résidu final : {res_learned[-1]:.6f}")

    # Figures
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(train_loss, label="Train", linewidth=2)
    axes[0].semilogy(test_loss,  label="Test",  linewidth=2, linestyle='--')
    axes[0].set_xlabel("Époque")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Courbe d'entraînement (curriculum)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    iters = np.arange(1, T_COMPARE + 1)
    axes[1].semilogy(iters, res_zero,    label="Zéro (base)", linewidth=2.5)
    axes[1].semilogy(iters, res_learned, label="Appris (MLP)", linewidth=2, linestyle='-.')
    axes[1].set_xlabel("Itération")
    axes[1].set_ylabel(r"Résidu $\|x_{n+1} - x_n\|$")
    axes[1].set_title(f"Vitesse de convergence — TEST ({T_COMPARE} iters)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150)
    plt.show()
    print("Figure sauvegardée : convergence_comparison.png")