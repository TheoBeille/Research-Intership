import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

from Algo_setuptorch_NN_full import get_setup, Params, build_algo_functions

params = Params()

TRAIN_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
TEST_SEEDS  = [8, 9]

size   = 16
SHAPES = [
    (1, 1, size, size),
    (1, 2, size, size),
    (1, 2, size, size),
    (1, 4, size, size),
]
N_CH      = sum(s[1] for s in SHAPES)  # 9
N_BLOCKS  = len(SHAPES)                 # 4
GAMMA_MAX = 0.99 / 0.425
EPS       = 1e-8

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

torch.manual_seed(0)
np.random.seed(0)

# ============================================================
#  DATA
# ============================================================

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

RA = train_data[0][2]['RA']
C  = train_data[0][2]['C']

# ============================================================
#  CONST — version full PyTorch, pas de float/Python
# ============================================================

def const(lambda_n, mu_n, gamma_n, gamma_prev, beta_bar):
    lpm  = lambda_n + mu_n
    a_n  = mu_n  / (lpm + EPS)
    ab_n = (gamma_n * mu_n) / (gamma_prev * lpm + EPS)

    th       = (4.0 - gamma_n * beta_bar) * lpm - 2.0 * lambda_n ** 2
    th_h     = 2.0 * lpm - gamma_n * beta_bar * lambda_n ** 2
    th_b     = lpm - lambda_n ** 2
    th_tilde = lpm * gamma_n * beta_bar

    return a_n, ab_n, th, th_h, th_b, th_tilde

# ============================================================
#  COMPUTE DELTA — version full PyTorch
# ============================================================

def compute_delta(p, x, p_prev, z, z_prev, y, y_prev, u, v,
                  a_n, th, th_h, th_b, gamma, lambda_n, gam_prev, mu_n):
    core = [
        p[i] - x[i]
        + a_n * (x[i] - p_prev[i])
        + (gamma * params.beta_bar * lambda_n**2 / (th_h + EPS)) * u[i]
        - (2 * th_b / (th + EPS)) * v[i]
        for i in range(N_BLOCKS)
    ]
    term1 = th / 2 * sum(c.pow(2).sum() for c in core)

    diff_z  = [(z[i] - p[i]) / (gamma + EPS) - (z_prev[i] - p_prev[i]) / (gam_prev + EPS)
               for i in range(N_BLOCKS)]
    diff_pp = [p[i] - p_prev[i] for i in range(N_BLOCKS)]
    term2   = 2 * mu_n * gamma * sum((diff_z[i] * diff_pp[i]).sum() for i in range(N_BLOCKS))

    diff_py = [(p[i] - y[i]) - (p_prev[i] - y_prev[i]) for i in range(N_BLOCKS)]
    term3   = (mu_n * gamma * params.beta_bar / 2.0) * sum(d.pow(2).sum() for d in diff_py)

    result = term1 + term2 + term3
    return result.clamp(min=0.0)

# ============================================================
#  PACK / UNPACK
# ============================================================

def pack(blocks):
    fixed, dev = [], None
    for b in blocks:
        b = b.float()
        if dev is None: dev = b.device
        if b.dim() == 5: b = b.squeeze(0)
        if b.dim() == 3: b = b.unsqueeze(0)
        elif b.dim() == 2: b = b.unsqueeze(0).unsqueeze(0)
        fixed.append(b.to(dev))
    return torch.cat(fixed, dim=1)

def unpack(tensor):
    out, c = [], 0
    for s in SHAPES:
        out.append(tensor[:, c:c + s[1]])
        c += s[1]
    return out

# ============================================================
#  DEVIATION NET
# ============================================================

class DeviationNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()

        in_features  = 4 * N_CH * size * size + 2
        out_features = 2 * N_CH * size * size + 4  # maps u,v + 4 scalaires

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
        # NE PAS init à zéro — sinon gradients nuls au départ
        nn.init.xavier_uniform_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_bl, p_prev_bl, y_prev_bl, z_prev_bl, delta, n, T):
        inp  = torch.cat([pack(x_bl), pack(p_prev_bl),
                          pack(y_prev_bl), pack(z_prev_bl)], dim=1)
        B    = inp.shape[0]
        flat = inp.view(B, -1)

        context = torch.stack([
            delta.float().expand(B) if delta.dim() == 0 else delta.float().view(B),
            torch.full((B,), n / T, device=flat.device)
            ], dim=1)
        flat = torch.cat([flat, context], dim=1)
        flat = flat.float() 
        out = self.net(flat)  # (B, out_features)

        # maps u, v
        maps  = out[:, :-4].view(B, 2 * N_CH, size, size)
        u_raw = [torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
                 for u in unpack(maps[:, :N_CH])]
        v_raw = [torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                 for v in unpack(maps[:, N_CH:])]

        # scalaires — moyenne sur batch, graphe intact
        s        = out[:, -4:].mean(dim=0)  # (4,)
        lambda_n = torch.nn.functional.softplus(s[0])
        gamma_n  = GAMMA_MAX * torch.sigmoid(s[1])
        mu_n     = torch.nn.functional.softplus(s[2])
        zeta_n   = torch.sigmoid(s[3])

        return u_raw, v_raw, lambda_n, gamma_n, mu_n, zeta_n

# ============================================================
#  SAFEGUARDING
# ============================================================

class SafeguardingLayer(nn.Module):
    def forward(self, u_raw, v_raw, delta, lambda_n, mu_n,
                th, th_h, th_b, th_tilde, zeta_n):
        th_h = th_h.clamp(min=EPS)
        th   = th  .clamp(min=EPS)
        lpm  = lambda_n + mu_n

        c_u = lpm * th_tilde / th_h
        c_v = lpm * th_h     / th

        Q = c_u * sum(u.pow(2).sum() for u in u_raw) \
          + c_v * sum(v.pow(2).sum() for v in v_raw) + EPS

        budget = zeta_n * delta.clamp(min=0.0)
        alpha  = torch.sqrt((budget / Q).clamp(0.0, 1.0))

        u_bl = [torch.nan_to_num(alpha * u, nan=0.0, posinf=0.0, neginf=0.0) for u in u_raw]
        v_bl = [torch.nan_to_num(alpha * v, nan=0.0, posinf=0.0, neginf=0.0) for v in v_raw]
        return u_bl, v_bl

# ============================================================
#  ONE STEP
# ============================================================

def one_step(x, y_prev, p_prev, z_prev, u, v,
             lambda_n, gamma_n, a_n, ab_n, th_h, th_b):
    y = [x[i] + a_n * (y_prev[i] - x[i]) + u[i] for i in range(N_BLOCKS)]

    z = [
        x[i]
        + a_n  * (p_prev[i] - x[i])
        + ab_n * (z_prev[i] - p_prev[i])
        + (th_b * gamma_n * params.beta_bar / (th_h + EPS)) * u[i]
        + v[i]
        for i in range(N_BLOCKS)
    ]

    Cy  = C(y)
    pr  = RA([z[i] - gamma_n * Cy[i] for i in range(N_BLOCKS)], gamma_n)

    x_new = [
        x[i] + lambda_n * (pr[i] - z[i]) + ab_n * lambda_n * (z_prev[i] - p_prev[i])
        for i in range(N_BLOCKS)
    ]

    residual = torch.sqrt(
        sum((pr[i] - y[i]).pow(2).sum() for i in range(N_BLOCKS)) + EPS
    )
    return x_new, y, pr, z, residual

# ============================================================
#  UNROLLED MODEL
# ============================================================

class UnrolledFBS(nn.Module):
    def __init__(self, T=20):
        super().__init__()
        self.T        = T
        self.dev_net  = DeviationNet(hidden=256)
        self.sg_layer = SafeguardingLayer()

    def _init_state(self, noisy):
        dev = noisy.device
        x = [
            noisy.clone(),
            torch.zeros(SHAPES[1], device=dev),
            torch.zeros(SHAPES[2], device=dev),
            torch.zeros(SHAPES[3], device=dev),
        ]
        p = [t.clone() for t in x]
        y = [t.clone() for t in x]
        z = [t.clone() for t in x]
        u = [torch.zeros_like(t) for t in x]
        v = [torch.zeros_like(t) for t in x]
        return x, y, p, z, u, v

    def forward(self, noisy):
        x, y_prev, p_prev, z_prev, u, v = self._init_state(noisy)
        residuals = []

        lam   = torch.tensor(float(params.lam0),  device=noisy.device)
        mu    = torch.tensor(0.0,                  device=noisy.device)
        gamma = torch.tensor(float(params.gamma0), device=noisy.device)
        gamma_prev = gamma.clone()

        for n in range(self.T):
            a_n, ab_n, th, th_h, th_b, th_tilde = const(
                lam, mu, gamma, gamma_prev, params.beta_bar
            )
            x_new, y, p, z, res = one_step(
                x, y_prev, p_prev, z_prev, u, v,
                lam, gamma, a_n, ab_n, th_h, th_b
            )
            delta = compute_delta(
                p, x_new, p_prev, z, z_prev, y, y_prev, u, v,
                a_n, th, th_h, th_b, gamma, lam, gamma_prev, mu
            )
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            u_raw, v_raw, lam, gamma, mu, zeta = self.dev_net(
                x_new, p, y, z, delta, n, self.T
            )

            gamma_prev = gamma.detach()  # évite explosion du graphe sur T long

            u, v = self.sg_layer(
                u_raw, v_raw, delta,
                lam, mu, th, th_h, th_b, th_tilde, zeta
            )
            x, y_prev, p_prev, z_prev = x_new, y, p, z
            residuals.append(res)

        return p_prev, residuals

# ============================================================
#  LOSS
# ============================================================

def convergence_loss(residuals):
    T       = len(residuals)
    weights = torch.linspace(1.0, float(T), T, device=residuals[0].device)
    weights = weights / weights.sum()
    return sum(w * r for w, r in zip(weights, residuals))

# ============================================================
#  TRAIN
# ============================================================

def train(n_epochs=200, lr=1e-4, T=20):
    model     = UnrolledFBS(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_hist, test_hist = [], []

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training on {n_epochs} epochs, T={T}...\n")

    for epoch in range(n_epochs):
        model.train()
        epoch_loss, count = 0.0, 0

        for noisy, _, _ in train_data:
            optimizer.zero_grad()
            _, residuals = model(noisy)
            loss = convergence_loss(residuals)
            if not torch.isfinite(loss):
                print(f"  [epoch {epoch}] loss non-finie, batch ignoré")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            count      += 1

        epoch_loss = epoch_loss / count if count > 0 else float('nan')
        train_hist.append(epoch_loss)

        model.eval()
        with torch.no_grad():
            test_loss = sum(
                convergence_loss(model(noisy)[1]).item()
                for noisy, _, _ in test_data
            ) / len(test_data)
        test_hist.append(test_loss)

        if epoch % 20 == 0:
            print(f"  Epoch {epoch:4d} | Train={epoch_loss:.6f} | Test={test_loss:.6f}")

    print("\nEntrainement termine.")
    return model, train_hist, test_hist

# ============================================================
#  EVALUATION
# ============================================================

def run_zero(noisy, T=1000):
    x      = [noisy.clone(),
               torch.zeros(SHAPES[1], device=device),
               torch.zeros(SHAPES[2], device=device),
               torch.zeros(SHAPES[3], device=device)]
    p_prev = [t.clone() for t in x]
    y_prev = [t.clone() for t in x]
    z_prev = [t.clone() for t in x]
    u = [torch.zeros_like(t) for t in x]
    v = [torch.zeros_like(t) for t in x]

    lam        = float(params.lam0)
    mu         = 0.0
    gamma      = float(params.gamma0)
    gamma_prev = gamma
    residuals  = []

    with torch.no_grad():
        for _ in range(T):
            a_n, ab_n, th, th_h, th_b, _ = const(
                torch.tensor(lam), torch.tensor(mu),
                torch.tensor(gamma), torch.tensor(gamma_prev),
                params.beta_bar
            )
            x_new, y_prev, p_prev, z_prev, res = one_step(
                x, y_prev, p_prev, z_prev, u, v,
                lam, gamma, a_n, ab_n, th_h, th_b
            )
            x = x_new
            val = res.item()
            residuals.append(val)
            if val < 1e-6:
                break
    return residuals

def run_learned(model, noisy):
    model.eval()
    with torch.no_grad():
        _, residuals = model(noisy)
    return [r.item() for r in residuals]

# ============================================================
#  MAIN
# ============================================================

if __name__ == "__main__":
    T = 80
    model, train_hist, test_hist = train(n_epochs=200, lr=1e-4, T=T)

    noisy_example = test_data[0][0]
    res_zero    = run_zero(noisy_example, T=1000)
    res_learned = run_learned(model, noisy_example)

    print(f"\nResidu final:")
    print(f"  Zero   : {res_zero[-1]:.6f}  ({len(res_zero)} iters)")
    print(f"  Appris : {res_learned[-1]:.6f}  ({len(res_learned)} iters)")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(res_zero,    label="Zero deviations", linewidth=2)
    ax.semilogy(res_learned, label="Learned (MLP)",   linewidth=2, linestyle='-.')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual ||p - y|| (log)")
    ax.set_title("Residual convergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("comparisonMLP_full.png", dpi=150)
    plt.show()