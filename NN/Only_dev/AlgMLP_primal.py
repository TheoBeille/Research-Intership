




        
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


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
N_CH_LEARNED = sum(s[1] for s in SHAPES[:2])   # = 3
class DeviationNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()

        in_features  = 4 * N_CH * size * size +2
        out_features = 2 * N_CH_LEARNED * size * size 

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

        
        inp = torch.cat([pack(x_bl), pack(p_prev_bl),
                        pack(y_prev_bl), pack(z_prev_bl)], dim=1)
        B    = inp.shape[0]
        flat = inp.view(B, -1)

        context = torch.tensor(
            [[float(delta), n / T]], device=flat.device
        ).expand(B, 2)

        flat = torch.cat([flat, context], dim=1)
        out = self.net(flat)
        out = out.view(B, 2 * N_CH_LEARNED, size, size)

        u_raw = [
            torch.nan_to_num(out[:, 0:1],              nan=0.0, posinf=0.0, neginf=0.0),  # bloc 0
            torch.nan_to_num(out[:, 1:3],              nan=0.0, posinf=0.0, neginf=0.0),  # bloc 1
            torch.zeros(B, SHAPES[2][1], size, size, device=flat.device),                  # bloc 2
            torch.zeros(B, SHAPES[3][1], size, size, device=flat.device),                  # bloc 3
        ]

        # v : même découpage, décalé de N_CH_LEARNED
        v_raw = [
            torch.nan_to_num(out[:, N_CH_LEARNED:N_CH_LEARNED+1], nan=0.0, posinf=0.0, neginf=0.0),  # bloc 0
            torch.nan_to_num(out[:, N_CH_LEARNED+1:N_CH_LEARNED+3], nan=0.0, posinf=0.0, neginf=0.0), # bloc 1
            torch.zeros(B, SHAPES[2][1], size, size, device=flat.device),                               # bloc 2
            torch.zeros(B, SHAPES[3][1], size, size, device=flat.device),                               # bloc 3
        ]
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
#   ITERATION ALGORITHME
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
        self.dev_net = DeviationNet(hidden=256)
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

            u_raw, v_raw = self.dev_net(x_new, p, y, z, delta, n, self.T)
            u_new, v_new = self.sg_layer(u_raw, v_raw, delta, n)

            x, y_prev, p_prev, z_prev = x_new, y, p, z
            u, v = u_new, v_new


            res = torch.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)
            residuals.append(res)

        return p_prev, residuals

def convergence_loss(residuals):
    T      = len(residuals)
    device = residuals[0].device
    weights = torch.linspace(1.0, float(T), T, device=device)
    weights = weights / weights.sum()
    loss = sum(w * r for w, r in zip(weights, residuals))
    return loss
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

    model, train_loss_hist, test_loss_hist = train(n_epochs=200, lr=1e-4, T=T)

    print("\nMethod comparison...")
    noisy_example = test_data[0][0]

    res_zero    = run_zero(noisy_example, T=1000)
    res_learned = run_learned(model, noisy_example)


    
  
    print(f"\nFinal residual:")
    print(f"  Zero    : {res_zero[-1]:.6f}")
    print(f"  Learned : {res_learned[-1]:.6f}")

    plt.close('all')
    fig, ax = plt.subplots(1, 1, figsize=(7, 4))

    ax.semilogy(res_zero,    label="Zero deviations", linewidth=2)
    ax.semilogy(res_learned, label="Learned (MLP)", linewidth=2, linestyle='-.')
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Residual ||p - y|| (log)")
    ax.set_title("Method Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("comparisonMLP_onlydev_primal.png", dpi=150)
    plt.show()
    print("Figure saved: comparisonMLP_onlydev_primal.png")