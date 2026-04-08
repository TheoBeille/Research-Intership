
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = False

from Algo_setuptorch_NN_full import get_setup, Params, build_algo_functions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")






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
N_CH      = sum(s[1] for s in SHAPES)
N_BLOCKS  = len(SHAPES)
GAMMA_MAX = 8.463922e-01
EPS       = 1e-8






torch.manual_seed(0)
np.random.seed(0)

def load_sample(seed, noise_level=0.1):
    setup       = get_setup(seed=seed, noise_level=noise_level, device=device)
    setup_clean = get_setup(seed=seed, noise_level=0.0,         device=device)
    functions   = build_algo_functions(setup, params)
    noisy = setup['noisy'].unsqueeze(1).to(device)  # (1,1,H,W) -> keep batch dim
    clean = setup_clean['noisy'].unsqueeze(1).to(device)
    # quick checks
    print(f"[load_sample] seed={seed} noisy.shape={noisy.shape} device={noisy.device}")
    return noisy, clean, functions

print("Loading data...")
train_data = [load_sample(s) for s in TRAIN_SEEDS]
test_data  = [load_sample(s) for s in TEST_SEEDS]
print(f"  Train: {len(train_data)} images  |  Test: {len(test_data)} images\n")

# extract functions from first sample and check
functions = train_data[0][2]
RA = functions['RA']
C  = functions['C']
compute_delta_fn = functions.get('compute_delta', None)
const_fn = functions.get('const', None)
print("Functions keys:", list(functions.keys()))

def const(lambda_n, mu_n, gamma_n, gamma_prev, beta_bar):
    return const_fn(lambda_n, mu_n, gamma_n, gamma_prev, beta_bar)

def compute_delta(p, x, p_prev, z, z_prev, y, y_prev, u, v,
                  a_n, th, th_h, th_b, gamma, lambda_n, gam_prev, mu_n):
    if compute_delta_fn is None:
        raise RuntimeError("compute_delta function not found in build_algo_functions")
    return compute_delta_fn(p, x, p_prev, z, z_prev, y, y_prev, u, v,
                            a_n, th, th_h, th_b, gamma, lambda_n, gam_prev, mu_n)

# PACK / UNPACK (unchanged)
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

class DeviationNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        in_features  = 4 * N_CH * size * size + 2
        out_features = 2 * N_CH * size * size + 4
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )
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
        out = self.net(flat)
        maps  = out[:, :-4].view(B, 2 * N_CH, size, size)
        u_raw = [torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
                 for u in unpack(maps[:, :N_CH])]
        v_raw = [torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                 for v in unpack(maps[:, N_CH:])]
        s        = out[:, -4:].mean(dim=0)
        lambda_n = torch.nn.functional.softplus(s[0])
        gamma_n  = GAMMA_MAX * torch.sigmoid(s[1])
        mu_n     = torch.nn.functional.softplus(s[2])
        zeta_n   = torch.sigmoid(s[3])
        return u_raw, v_raw, lambda_n, gamma_n, mu_n, zeta_n



class SafeguardLayer(nn.Module):
    """
    Projection par scaling simple pour garantir la sauvegarde (inegalite (3) de l'alg.1).
    radius^2 = S_next * en / (An1_plus_In1)
    """
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps = eps

    def _pack(self, blocks):
        return torch.cat([b.view(b.shape[0], -1) for b in blocks], dim=1)

    def _unpack(self, flat, template_blocks):
        out = []
        c = 0
        B = flat.shape[0]
        for b in template_blocks:
            n = b.numel() // B
            out.append(flat[:, c:c+n].view_as(b))
            c += n
        return out

    def forward(self, u_blocks, v_blocks, en, An1_plus_In1, S_next=1.0):
        # pack
        u_flat = self._pack(u_blocks)    # (B, Du)
        v_flat = self._pack(v_blocks)    # (B, Dv)
        uv = torch.cat([u_flat, v_flat], dim=1)  # (B, D)

        # radius^2 = S_next * en / (An1_plus_In1 + eps)
        denom = An1_plus_In1.clamp(min=self.eps)
        radius2 = (S_next * en) / denom
        if radius2.dim() == 0:
            radius2 = radius2.view(1).expand(uv.shape[0])
        radius = torch.sqrt(radius2.clamp(min=0.0) + self.eps)  # (B,)

        norms = torch.norm(uv, dim=1)  # (B,)
        scale = (radius / (norms + self.eps)).clamp(max=1.0).view(-1, 1)  # <=1

        uv_safe = uv * scale  # (B, D)

        # unpack
        u_safe = self._unpack(uv_safe[:, :u_flat.shape[1]], u_blocks)
        v_safe = self._unpack(uv_safe[:, u_flat.shape[1]:], v_blocks)

        return u_safe, v_safe, norms, radius





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

class UnrolledFBS(nn.Module):
    def __init__(self, T=20, debug_checks=True):
        super().__init__()
        self.T        = T
        self.dev_net  = DeviationNet(hidden=256)
        self.safeguard = SafeguardLayer()

        self.debug_checks = debug_checks

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

            # checks
            if self.debug_checks and (n % 5 == 0):
                print(f"[iter {n}] residual={res.item():.6e} lam={float(lam):.4e} gamma={float(gamma):.4e}")
                

            delta = compute_delta(
                p, x_new, p_prev, z, z_prev, y, y_prev, u, v,
                a_n, th, th_h, th_b, gamma, lam, gamma_prev, mu
            )
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            u, v, lam, gamma, mu, zeta = self.dev_net(
                x_new, p, y, z, delta, n, self.T
            )
            en = delta.detach() if isinstance(delta, torch.Tensor) else torch.tensor(float(delta), device=noisy.device)
            An1_plus_In1 = torch.tensor(1.0, device=noisy.device)

            u_safe, v_safe, norms_uv, radius_uv = self.safeguard(u, v, en, An1_plus_In1, S_next=1.0)

            # Logging succinct (optionnel, utile pour checks réguliers)
            if n % 5 == 0:
                clipped_frac = float((norms_uv > radius_uv).float().mean().item())
                print(f"  [safeguard] iter={n} ||u,v|| mean={norms_uv.mean().item():.4e} radius mean={radius_uv.mean().item():.4e} clipped_frac={clipped_frac:.3f}")

            # remplacer u,v par les versions sécurisées
            u, v = u_safe, v_safe
            # more checks
            if self.debug_checks and (n % 10 == 0):
                # scalars
                print(f"  dev_net scalars: lam={float(lam):.6f} gamma={float(gamma):.6f} mu={float(mu):.6f} zeta={float(zeta):.6f}")

               

            gamma_prev = gamma.detach()
            x, y_prev, p_prev, z_prev = x_new, y, p, z
            residuals.append(res)

        return p_prev, residuals

def convergence_loss(residuals):
    T       = len(residuals)
    weights = torch.linspace(1.0, float(T), T, device=residuals[0].device)
    weights = weights / weights.sum()
    return sum(w * r for w, r in zip(weights, residuals))

def train(n_epochs=50, lr=1e-4, T=20):
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
                print(f"  [epoch {epoch}] loss non-finie, batch ignoré, loss={loss}")
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

        print(f"  Epoch {epoch:4d} | Train={epoch_loss:.6f} | Test={test_loss:.6f}")

    print("\nTraining finished.")
    return model, train_hist, test_hist

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
        for it in range(T):
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
            if val < 1e-3:
                break
    return residuals

def run_learned(model, noisy):
    model.eval()
    with torch.no_grad():
        _, residuals = model(noisy)
    return [r.item() for r in residuals]

if __name__ == "__main__":
    T = 100
    model, train_hist, test_hist = train(n_epochs=200, lr=1e-4, T=T)

    noisy_example = test_data[0][0]
    res_zero    = run_zero(noisy_example, T=1000)
    res_learned = run_learned(model, noisy_example)

    print(f"\n final:")
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
    plt.savefig("comparisonMLP_full_debug.png", dpi=150)
    plt.show()
