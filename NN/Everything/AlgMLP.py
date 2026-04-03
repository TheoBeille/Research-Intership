import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

from Algo_setuptorch_NN_full import get_setup, Params, build_algo_functions

# ============================================================
#  SETUP
# ============================================================

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
N_CH     = sum(s[1] for s in SHAPES)   # 9
N_BLOCKS = len(SHAPES)                  # 4
DIM      = N_CH * size * size
GAMMA_MAX = 0.99 / 0.425


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

torch.manual_seed(0)
np.random.seed(0)

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
const         = functions['const']

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

class DeviationNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()

        in_features  = 4 * N_CH * size * size + 2
        # u maps + v maps + 4 scalaires (lambda, gamma, mu, zeta)
        out_features = 2 * N_CH * size * size + 4

        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x_bl, p_prev_bl, y_prev_bl, z_prev_bl, delta, n, T):
        inp  = torch.cat([pack(x_bl), pack(p_prev_bl),
                          pack(y_prev_bl), pack(z_prev_bl)], dim=1)
        B    = inp.shape[0]
        flat = inp.view(B, -1)

        context = torch.tensor(
            [[float(delta), n / T]], device=flat.device
        ).expand(B, 2)
        flat = torch.cat([flat, context], dim=1)

        out = self.net(flat)   # (B, out_features)

        maps  = out[:, :-4].view(B, 2 * N_CH, size, size)
        u_raw = unpack(maps[:, :N_CH])
        v_raw = unpack(maps[:, N_CH:])
        u_raw = [torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0) for u in u_raw]
        v_raw = [torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0) for v in v_raw]

        s = out[:, -4:]   # (B, 4)

        lambda_n = torch.nn.functional.softplus(s[:, 0]).mean()
        gamma_n  = GAMMA_MAX  * torch.sigmoid(s[:, 1]).mean()   # (,)
        mu_n     = torch.nn.functional.softplus(s[:, 2]).mean()  # (,)
        zeta_n   = torch.sigmoid(s[:, 3]).mean()                 # (,)

        return u_raw, v_raw, lambda_n, gamma_n, mu_n, zeta_n

class SafeguardingLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.eps = 1e-12

    def forward(self, u_raw_bl, v_raw_bl, delta, lambda_n, mu_n,
                th, th_h, th_b, th_tilde, zeta_n):

        th_hat = th_h if isinstance(th_h, torch.Tensor) else torch.tensor(
            float(th_h), device=u_raw_bl[0].device)
        th_    = th   if isinstance(th,   torch.Tensor) else torch.tensor(
            float(th),   device=u_raw_bl[0].device)
        th_hat = th_hat.clamp(min=self.eps)
        th_    = th_.clamp(min=self.eps)

        lpm = lambda_n + mu_n

        c_u = lpm * th_tilde / th_hat
        c_v = lpm * th_hat   / th_

        norm_u_sq = sum(u.pow(2).sum() for u in u_raw_bl)
        norm_v_sq = sum(v.pow(2).sum() for v in v_raw_bl)
        Q = c_u * norm_u_sq + c_v * norm_v_sq + 1e-12

        delta  = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        budget = zeta_n * delta.clamp(min=0.0)
        alpha  = torch.sqrt((budget / Q).clamp(0.0, 1.0))

        u_bl = [torch.nan_to_num(alpha * u, nan=0.0, posinf=0.0, neginf=0.0) for u in u_raw_bl]
        v_bl = [torch.nan_to_num(alpha * v, nan=0.0, posinf=0.0, neginf=0.0) for v in v_raw_bl]
        return u_bl, v_bl

def one_step(x, y_prev, p_prev, z_prev, u, v,
             lambda_n, gamma_n, a_n, ab_n, th_h, th_b):
    eps = 1e-12

    y = [x[i] + a_n * (y_prev[i] - x[i]) + u[i]
         for i in range(N_BLOCKS)]

    z = [
        x[i]
        + a_n  * (p_prev[i] - x[i])
        + ab_n * (z_prev[i] - p_prev[i])
        + (th_b * gamma_n * params.beta_bar / (th_h + eps)) * u[i]
        + v[i]
        for i in range(N_BLOCKS)
    ]

    Cy          = C(y)
    z_minus_gCy = [z[i] - gamma_n * Cy[i] for i in range(N_BLOCKS)]
    pr          = RA(z_minus_gCy, gamma_n)

    x_new = [
        x[i] + lambda_n * (pr[i] - z[i]) + ab_n * lambda_n * (z_prev[i] - p_prev[i])
        for i in range(N_BLOCKS)
    ]

    residual = torch.sqrt(
        sum((pr[i] - y[i]).pow(2).sum() for i in range(N_BLOCKS)) + 1e-12
    )
    return x_new, y, pr, z, residual

class UnrolledFBS(nn.Module):
    def __init__(self, T=20):
        super().__init__()
        self.T          = T
        self.dev_blocks = DeviationNet(hidden=256)
        self.sg_layer   = SafeguardingLayer()

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

        lambda_val = float(params.lam0)
        mu_val     = 0.0
        gamma_val  = float(params.gamma0)
        gamma_prev = gamma_val

        for n in range(self.T):
            a_n, ab_n, th, th_h, th_b, th_tilde = const(
                lambda_val, mu_val, gamma_val, gamma_prev, params.beta_bar
            )

            x_new, y, p, z, res = one_step(
                x, y_prev, p_prev, z_prev, u, v,
                lambda_val, gamma_val, a_n, ab_n, th_h, th_b
            )

            delta = compute_delta(
                p, x_new, p_prev, z, z_prev, y, y_prev, u, v,
                a_n, th, th_h, th_b, gamma_val, lambda_val, gamma_prev, mu_val
            )
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            u_raw, v_raw, lambda_t, gamma_t, mu_t, zeta_t = self.dev_blocks(
                x_new, p, y, z, delta, n, self.T
            )

            gamma_prev = gamma_val

            lambda_val = lambda_t.item()
            gamma_val  = gamma_t.item()
            mu_val     = mu_t.item()

            u_new, v_new = self.sg_layer(
                u_raw, v_raw, delta,
                lambda_t, mu_t,
                th, th_h, th_b, th_tilde, zeta_t
            )

            x, y_prev, p_prev, z_prev = x_new, y, p, z
            u, v = u_new, v_new

            residuals.append(torch.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6))

        return p_prev, residuals

def convergence_loss(residuals):
    T      = len(residuals)
    device = residuals[0].device
    weights = torch.linspace(1.0, float(T), T, device=device)
    weights = weights / weights.sum()
    loss = sum(w * r for w, r in zip(weights, residuals))
    return loss

def train(n_epochs=200, lr=1e-4, T=20):
    train_loss_hist, test_loss_hist = [], []
    model     = UnrolledFBS(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Training on {n_epochs} epochs, T={T}...\n")

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
            print(f"  Epoch {epoch:4d} | Train={epoch_loss:.6f} | Test={test_loss:.6f}")

    print("\nEntrainement termine.")
    return model, train_loss_hist, test_loss_hist

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

    lambda_n   = float(params.lam0)
    mu_n       = 0.0
    gamma_n    = float(params.gamma0)
    gamma_prev = gamma_n

    residuals = []
    with torch.no_grad():
        for n in range(T):
            a_n, ab_n, th, th_h, th_b, _ = const(
                lambda_n, mu_n, gamma_n, gamma_prev, params.beta_bar
            )
            x_new, y_prev, p_prev, z_prev, res = one_step(
                x, y_prev, p_prev, z_prev, u, v,
                lambda_n, gamma_n, a_n, ab_n, th_h, th_b
            )
            x = x_new
            val = res.item() if torch.is_tensor(res) else float(res)
            residuals.append(val)
            if val < 1e-6:
                break
    return residuals

def run_learned(model, noisy):
    model.eval()
    with torch.no_grad():
        _, residuals = model(noisy)
    return [float(r.item()) for r in residuals]

if __name__ == "__main__":

    T = 80   

    model, train_loss_hist, test_loss_hist = train(n_epochs=200, lr=1e-4, T=T)

    print("\nComparaison des methodes...")
    noisy_example = test_data[0][0]

    res_zero    = run_zero(noisy_example, T=1000)
    res_learned = run_learned(model, noisy_example)

    print(f"\nResidu final :")
    print(f"  Zero   : {res_zero[-1]:.6f}  ({len(res_zero)} iterations)")
    print(f"  Appris : {res_learned[-1]:.6f}  ({len(res_learned)} iterations)")

    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

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
    print("Figure sauvegardee : comparisonMLP_full.png")