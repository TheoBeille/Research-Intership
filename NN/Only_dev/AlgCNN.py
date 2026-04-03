import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
torch.backends.cudnn.enabled = False
# Désactiver LaTeX dans matplotlib
plt.rcParams['text.usetex'] = False

from Algo_setuptorch import get_setup, Params, build_algo_functions

# ============================================================
#  SETUP
# ============================================================

params = Params()

TRAIN_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
TEST_SEEDS  = [8, 9]

size     = 16
SHAPES   = [
    (1, 1, size, size),
    (1, 2, size, size),
    (1, 2, size, size),
    (1, 4, size, size),
]
N_CH     = sum(s[1] for s in SHAPES)   # 9
N_BLOCKS = len(SHAPES)                  # 4
DIM      = N_CH * size * size



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

torch.manual_seed(0)
np.random.seed(0)

#------------------------
#Building the data set for training + all what's needed for the algorithm
#------------------------

def load_sample(seed, noise_level=0.1):
    setup       = get_setup(seed=seed, noise_level=noise_level, device=device)
    setup_clean = get_setup(seed=seed, noise_level=0.0,         device=device)
    functions   = build_algo_functions(setup, params)
    noisy = setup['noisy'].unsqueeze(1).to(device)
    clean = setup_clean['noisy'].unsqueeze(1).to(device)
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
# Make tendor in the good form + I had a problem of dimension that was addded 
# ============================================================
#I am using a MLP so should be 1D
def pack(blocks):
    fixed = []
    dev = None
    for b in blocks:
        b = b.float()
        if dev is None:
            dev = b.device
        if b.dim() == 5:
            b = b.squeeze(0)
        if b.dim() == 3:
            b = b.unsqueeze(0)
        elif b.dim() == 2:
            b = b.unsqueeze(0).unsqueeze(0)
        fixed.append(b.to(dev))
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
#  DEVIATIONNET - MLP
# ============================================================

class DeviationCNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        # entrée : concat des packs -> (B, N_CH, size, size) par pack, on concat 4 packs + context -> channels = 4*N_CH + 2
        in_ch = 4 * N_CH + 2
        out_ch = 2 * N_CH

        # encoder initial
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, hidden),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(8, hidden),
        )

        # quelques blocs résiduels conv
        res_blocks = []
        for _ in range(4):
            res_blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
                nn.GELU(),
                nn.GroupNorm(8, hidden),
                nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            ))
        self.res_blocks = nn.ModuleList(res_blocks)

        # projection finale
        self.out_conv = nn.Conv2d(hidden, out_ch, kernel_size=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x_bl, p_prev_bl, y_prev_bl, z_prev_bl, delta, n, T):
        # pack les blocs (retourne (B, N_CH, H, W) pour chaque pack)
        x = pack(x_bl)           # (B, N_CH, H, W)
        p = pack(p_prev_bl)
        y = pack(y_prev_bl)
        z = pack(z_prev_bl)

        # contexte : delta et n/T -> on crée deux canaux constants spatiaux
        B = x.shape[0]
        device = x.device
        ctx = torch.tensor([[float(delta), float(n) / float(T)]], device=device, dtype=x.dtype)
        ctx = ctx.expand(B, 2)                      # (B,2)
        ctx = ctx.view(B, 2, 1, 1).expand(B, 2, size, size)  # (B,2,H,W)

        inp = torch.cat([x, p, y, z, ctx], dim=1)   # (B, 4*N_CH + 2, H, W)

        h = self.enc(inp)
        for block in self.res_blocks:
            h = h + block(h)

        out = self.out_conv(h)                      # (B, 2*N_CH, H, W)

        # split en u_raw / v_raw et unpack en listes de blocs
        u_raw = unpack(out[:, :N_CH])
        v_raw = unpack(out[:, N_CH:2*N_CH])

        # sécurité numérique
        u_raw = [torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0) for u in u_raw]
        v_raw = [torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in v_raw]
        return u_raw, v_raw


# ============================================================
#  SAFEGUARDING LAYER
# ============================================================

class SafeguardingLayer(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.p = params
        self.eps = 1e-12

    def forward(self, u_raw_bl, v_raw_bl, delta, n):
        p   = self.p
        lam = float(p.lam(n + 1))
        mu  = float(p.mu(n + 1))
        lpm = lam + mu

        th_hat = float(p.theta_hat(n + 1))
        th_hat = max(th_hat, self.eps)
        th = float(p.theta(n + 1))
        th = max(th, self.eps)
        th_tilde = float(p.theta_tilde(n + 1))

        c_u = lpm * th_tilde / th_hat
        c_v = lpm * th_hat   / th

        norm_u_sq = sum(u.pow(2).sum() for u in u_raw_bl)
        norm_v_sq = sum(v.pow(2).sum() for v in v_raw_bl)

        Q = c_u * norm_u_sq + c_v * norm_v_sq + 1e-12



        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        budget = float(p.zeta) * delta.clamp(min=0.0)

        ratio = (budget / Q)
 
        ratio = ratio.clamp(min=0.0, max=1.0)
        alpha = torch.sqrt(ratio)


        u_bl = [torch.nan_to_num(alpha * u, nan=0.0, posinf=0.0, neginf=0.0) for u in u_raw_bl]
        v_bl = [torch.nan_to_num(alpha * v, nan=0.0, posinf=0.0, neginf=0.0) for v in v_raw_bl]
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

    y = [x[i] + a_n * (y_prev[i] - x[i]) + u[i]
         for i in range(N_BLOCKS)]

    z = [
        x[i]
        + a_n  * (p_prev[i] - x[i])
        + ab_n * (z_prev[i] - p_prev[i])
        + (th_b * gam * p.beta_bar / (th_h + 1e-12)) * u[i]
        + v[i]
        for i in range(N_BLOCKS)
    ]

    Cy          = C(y)
    z_minus_gCy = [z[i] - gam * Cy[i] for i in range(N_BLOCKS)]
    pr          = RA(z_minus_gCy, gam)

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
        self.dev_blocks = DeviationCNN(hidden=64)
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

            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            u_raw, v_raw = self.dev_blocks(x_new, p, y, z, delta, n, self.T)
            u_new, v_new = self.sg_layer(u_raw, v_raw, delta, n)

            x, y_prev, p_prev, z_prev = x_new, y, p, z
            u, v = u_new, v_new

            res = torch.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)
            residuals.append(res)

        return p_prev, residuals

def convergence_loss(residuals, target_iter=20):
    T      = len(residuals)
    device = residuals[0].device
    r0     = residuals[0].detach().clamp(min=1e-8)

    # Terme 1 : log-résidu avec poids croissants
    weights = torch.exp(torch.linspace(0, 2.0, T, device=device))
    log_res = [torch.log(r.clamp(min=1e-8) / r0) for r in residuals]
    base    = sum(w * l for w, l in zip(weights, log_res)) / weights.sum()

    # Terme 2 : pénalité explicite à l'itération cible
    idx     = min(target_iter - 1, T - 1)
    penalty = torch.log(residuals[idx].clamp(min=1e-8) / r0)

    # Terme 3 : régularisation taux de contraction
    rates = [residuals[i+1] / residuals[i].detach().clamp(min=1e-8)
             for i in range(min(target_iter, T-1))]
    reg   = sum(r.clamp(min=0.5) for r in rates) / len(rates)

    return base + 2.0 * penalty + 0.5 * reg
# ============================================================
#  TRAINING END-TO-END
# ============================================================

def train(n_epochs=200, lr=1e-4, T=100):
    train_loss_hist, test_loss_hist = [], []
    epoch_global = 0
    model     = UnrolledFBS(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Training on {n_epochs} epochs...\n")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, valid_count = 0.0, 0

        for noisy, _, _ in train_data:
            optimizer.zero_grad()
            _, residuals = model(noisy)
            loss = convergence_loss(residuals)
            if not torch.isfinite(loss):
                print(f"Warning: non-finite loss at epoch {epoch}, skipping batch")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
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

    print("\nEntrainement termine.")
    return model, train_loss_hist, test_loss_hist

# ============================================================
#  COMPARAISON : ZERO / APPRIS
# ============================================================

def run_zero(noisy, T=1000):
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
        for n in range(T):
            x, y_prev, p_prev, z_prev, res = one_step(
                x, y_prev, p_prev, z_prev, u, v, n
            )
            val = res.item() if torch.is_tensor(res) else float(res)
            residuals.append(val)
            if val < 1e-3:
                break
    return residuals

def run_learned(model, noisy):
    model.eval()
    with torch.no_grad():
        _, residuals = model(noisy)
    return [float(r.item()) for r in residuals]

# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":

    T = 100


    model = UnrolledFBS(T=5)
    model.to(device)
    noisy_example = train_data[0][0]
    with torch.no_grad():
        p_out, residuals = model(noisy_example)
    print("Forward OK, nb residuals:", len(residuals))


    model, train_loss_hist, test_loss_hist = train(n_epochs=200, lr=1e-4, T=T)

    print("\nMethod comparison...")
    noisy_example = test_data[0][0]

    res_zero    = run_zero(noisy_example, T=1000)
    res_learned = run_learned(model, noisy_example)


    
  
    print(f"\nFinal residual:")
    print(f"  Zero    : {res_zero[-1]:.6f}")
    print(f"  Learned  : {res_learned[-1]:.6f}")

    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.semilogy(res_zero,    label="Zero deviations", linewidth=2)
    ax.semilogy(res_learned, label="Learned (CNN)", linewidth=2, linestyle='-.')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual ||p - y|| (log)")
    ax.set_title("Method Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparisonMLP_full.png", dpi=150)
    plt.show()
    print("Figure saved: comparisonMLP_full.png")