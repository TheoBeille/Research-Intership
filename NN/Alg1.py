import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Configure matplotlib to use LaTeX fonts
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'

from Algo_setuptorch import get_setup, Params, build_algo_functions

# ============================================================
# 1. Global Configuration
# ============================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

params = Params()

TRAIN_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
TEST_SEEDS  = [8, 9]

size     = 32
SHAPES   = [
    (1, 1, size, size),
    (1, 2, size, size),
    (1, 2, size, size),
    (1, 4, size, size),
]
N_CH     = sum(s[1] for s in SHAPES)   # 9
N_BLOCKS = len(SHAPES)                  # 4

T_COMPARE = 50
T_TRAIN   = 50

# ============================================================
# 2. Data Loading
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
# 3. Shape Utilities
# ============================================================

def ensure_4d(t: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has 4 dimensions."""

def ensure_4d(t: torch.Tensor) -> torch.Tensor:
    while t.dim() > 4:
        t = t.squeeze(0)
    while t.dim() < 4:
        t = t.unsqueeze(0)
    return t

def fix_blocks(blocks):
    return [ensure_4d(b.float()) for b in blocks]

def pack(blocks):
    return torch.cat(fix_blocks(blocks), dim=1)

def unpack(tensor: torch.Tensor):
    tensor = ensure_4d(tensor)
    out, c = [], 0
    for s in SHAPES:
        out.append(tensor[:, c:c + s[1]])
        c += s[1]
    return out

# ============================================================
# 4. Deviation Network
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
        ], dim=1)                                # (B, 36, 32, 32)

        B   = inp.shape[0]
        inp = inp.view(B, -1)                    # (B, 36864)
        out = self.net(inp)                      # (B, 18432)
        out = out.view(B, 2 * N_CH, size, size)  # (B, 18, 32, 32)

        u_raw = unpack(out[:, :N_CH])
        v_raw = unpack(out[:, N_CH:])
        return u_raw, v_raw

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
                + c_v * sum(v.pow(2).sum() for v in dir_v_bl)
                + 1e-12)
        budget = p.zeta * delta.clamp(min=0.0)
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

    z = [
        x[i]
        + a_n  * (p_prev[i] - x[i])
        + ab_n * (z_prev[i] - p_prev[i])
        + (th_b * gam * p.beta_bar / th_h) * u[i]
        + v[i]
        for i in range(N_BLOCKS)
    ]

    Cy          = fix_blocks(C(y))
    z_minus_gCy = [z[i] - gam * Cy[i] for i in range(N_BLOCKS)]
    pr          = fix_blocks(RA(z_minus_gCy, gam))

    x_new = fix_blocks([
        x[i] + lam * (pr[i] - z[i]) + ab_n * lam * (z_prev[i] - p_prev[i])
        for i in range(N_BLOCKS)
    ])

    residual = torch.sqrt(
        sum((pr[i] - y[i]).pow(2).sum() for i in range(N_BLOCKS)) + 1e-12
    )
    return x_new, y, pr, z, residual

# ============================================================
# 7. Init state
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
    def __init__(self, T: int = T_TRAIN):
        super().__init__()
        self.T        = T
        self.dev_net  = DeviationNet(hidden=64)
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
    weights = torch.linspace(1.0, float(T), T, device=residuals[0].device)
    return sum(w * r for w, r in zip(weights, residuals)) / weights.sum()

# ============================================================
# 10. Training
# ============================================================

def train(n_epochs=300, lr=1e-4, T=T_TRAIN):
    model     = UnrolledFBS(T=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    print(f"Training — {n_epochs} epochs | T={T}\n")

    train_loss_hist = []
    test_loss_hist  = []

    for epoch in range(n_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for noisy, _, _ in train_data:
            optimizer.zero_grad()
            _, residuals = model(noisy)
            loss = convergence_loss(residuals)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_data)
        train_loss_hist.append(epoch_loss)

        # --- Test ---
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for noisy, _, _ in test_data:
                _, residuals = model(noisy)
                test_loss += convergence_loss(residuals).item()
            test_loss /= len(test_data)
        test_loss_hist.append(test_loss)

        scheduler.step()

        if epoch % 30 == 0:
            print(f"Epoch {epoch:4d} | Train loss = {epoch_loss:.6f} | Test loss = {test_loss:.6f}")

    print("\nTraining complete.")
    return model, train_loss_hist, test_loss_hist

# ============================================================
# 11. Comparison Methods
# ============================================================

def run_zero(noisy, T=T_COMPARE):
    x, y_prev, p_prev, z_prev, u, v = init_state(noisy)
    residuals = []
    with torch.no_grad():
        for n in range(T):
            x, y_prev, p_prev, z_prev, res = one_step(
                x, y_prev, p_prev, z_prev, u, v, n
            )
            residuals.append(res.item())
    return x[0], residuals


def run_random(noisy, T=T_COMPARE):
    sg = SafeguardingLayer(params).to(device)
    x, y_prev, p_prev, z_prev, u, v = init_state(noisy)
    residuals = []
    with torch.no_grad():
        for n in range(T):
            x_new, y, p, z, res = one_step(x, y_prev, p_prev, z_prev, u, v, n)
            delta = compute_delta(p, x_new, p_prev, z, z_prev, y, y_prev, u, v, n)
            u_raw = [torch.randn_like(t) for t in x]
            v_raw = [torch.randn_like(t) for t in x]
            u, v  = sg(u_raw, v_raw, delta, n)
            x, y_prev, p_prev, z_prev = x_new, y, p, z
            residuals.append(res.item())
    return x[0], residuals


def run_learned(model, noisy, T=T_COMPARE):
    model.eval()
    sg = model.sg_layer
    x, y_prev, p_prev, z_prev, u, v = init_state(noisy)
    residuals = []
    with torch.no_grad():
        for n in range(T):
            x_new, y, p, z, res = one_step(x, y_prev, p_prev, z_prev, u, v, n)
            delta        = compute_delta(p, x_new, p_prev, z, z_prev, y, y_prev, u, v, n)
            dir_u, dir_v = model.dev_net(x_new, p, y, z)
            u, v         = sg(dir_u, dir_v, delta, n)
            x, y_prev, p_prev, z_prev = x_new, y, p, z
            residuals.append(res.item())
    return x[0], residuals

# ============================================================
# 12. Main
# ============================================================

if __name__ == "__main__":

    # --- Entraînement ---
    model, train_loss, test_loss = train(n_epochs=300, lr=1e-4, T=T_TRAIN)

    # --- Comparison on test images ---
    print(f"\nComparison on test images ({T_COMPARE} iterations)..")

    all_zero, all_rand, all_learned = [], [], []

    for noisy, clean, _ in test_data:
        _, r_z = run_zero(noisy,           T=T_COMPARE)
        _, r_r = run_random(noisy,         T=T_COMPARE)
        _, r_l = run_learned(model, noisy, T=T_COMPARE)
        all_zero.append(r_z)
        all_rand.append(r_r)
        all_learned.append(r_l)

    res_zero    = np.mean(all_zero,    axis=0)
    res_random  = np.mean(all_rand,    axis=0)
    res_learned = np.mean(all_learned, axis=0)

    # --- Convergence threshold ---
    threshold = 1e-2
    def iters_to_threshold(res, thr):
        for i, r in enumerate(res):
            if r <= thr:
                return i + 1
        return None

    it_z = iters_to_threshold(res_zero,    threshold)
    it_r = iters_to_threshold(res_random,  threshold)
    it_l = iters_to_threshold(res_learned, threshold)
    print(f"\nIterations to converge to {threshold:.4f}:")
    print(f"  Zero     : {it_z}")
    print(f"  Random   : {it_r}")
    print(f"  Learned  : {it_l}")

    # ============================================================
    # Figures
    # ============================================================

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(train_loss, label=r"Train", linewidth=2)
    axes[0].semilogy(test_loss,  label=r"Test",  linewidth=2, linestyle='--')
    axes[0].axvline(x=150, color='gray', linestyle=':', alpha=0.6, label=r"LR $\div 2$")
    axes[0].set_xlabel(r"Epoch")
    axes[0].set_ylabel(r"Loss (log)")
    axes[0].set_title(r"Training")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    iters = np.arange(1, T_COMPARE + 1)
    axes[1].semilogy(iters, res_zero,    label=r"Zero (baseline)",  linewidth=2.5)
    axes[1].semilogy(iters, res_random,  label=r"Random",           linewidth=2,   linestyle='--')
    axes[1].semilogy(iters, res_learned, label=r"Learned",          linewidth=2,   linestyle='-.')
    axes[1].axhline(y=threshold, color='gray', linestyle=':', alpha=0.7,
                    label=f"Threshold $= {threshold:.4f}$")
    axes[1].set_xlabel(r"Iteration")
    axes[1].set_ylabel(r"Residual $\|p - y\|$ (log)")
    axes[1].set_title(r"Convergence Speed — Test Images")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("convergence_comparison.png", dpi=150)
    plt.show()

    # --- Visualization of first test image ---
    noisy_vis, clean_vis, _ = test_data[0]
    x_z, _ = run_zero(noisy_vis,           T=T_COMPARE)
    x_l, _ = run_learned(model, noisy_vis, T=T_COMPARE)

    fig2, axs2 = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(axs2,
                               [noisy_vis, clean_vis, x_z, x_l],
                               [r"Noisy (test)", r"Clean (ref)", r"Zero", r"Learned"]):
        ax.imshow(img.detach().cpu().numpy().squeeze(), cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    fig2.suptitle(f"Test Image (seed={TEST_SEEDS[0]}) after {T_COMPARE} iterations", fontsize=13)
    fig2.tight_layout()
    fig2.savefig("test_images.png", dpi=150)
    plt.show()

    print("\nFigures saved: convergence_comparison.png, test_images.png")
