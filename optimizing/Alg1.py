import torch
import numpy as np
import matplotlib.pyplot as plt
from Algo_setuptorch import get_setup, Params, build_algo_functions

setup  = get_setup()
params = Params()
fcts   = build_algo_functions(setup, params)

noisy           = setup['noisy']
gamma           = fcts['gamma']
alpha           = fcts['alpha']
alpha_bar       = fcts['alpha_bar']
theta_hat       = fcts['theta_hat']
theta_bar       = fcts['theta_bar']
RA              = fcts['RA']
C               = fcts['C']
compute_deviations = fcts["compute_deviations"]

shapes = [
    (1, 64, 64),
    (1, 2, 64, 64),
    (1, 2, 64, 64),
    (1, 4, 64, 64),
]

N       = 1000
n_inner = 500
lr      = 1e-3


def one_iteration(state, dir_u, dir_v, n):
    x, y_prev, p_prev, z_prev, u, v = state

    gam  = gamma(n)
    a_n  = alpha(n)
    ab_n = alpha_bar(n)
    th_h = theta_hat(n)
    th_b = theta_bar(n)

    y = [x[i] + a_n*(y_prev[i] - x[i]) + u[i] for i in range(4)]
    z = [(x[i]
          + a_n*(p_prev[i] - x[i])
          + ab_n*(z_prev[i] - p_prev[i])
          + (th_b * gam * params.beta_bar / th_h) * u[i]
          + v[i]) for i in range(4)]

    Cy = C(y)
    z_minus_gCy = [z[i] - gam*Cy[i] for i in range(4)]
    p = RA(z_minus_gCy, gam)

    lam_n = params.lam(n)
    x_new = [x[i] + lam_n*(p[i] - z[i]) + ab_n*lam_n*(z_prev[i] - p_prev[i])
             for i in range(4)]

    residual = torch.sqrt(sum((p[i] - y[i]).norm()**2 for i in range(4)) + 1e-12)

    u_next, v_next = compute_deviations(
        p, x_new, p_prev, z, z_prev,
        y, y_prev, u, v,
        n, dir_u, dir_v
    )

    next_state = [x_new, y, p, z, u_next, v_next]
    return next_state, residual


def make_initial_state():
    x      = [torch.zeros(s) for s in shapes]
    x[0]   = noisy.clone()
    y_prev = [t.clone() for t in x]
    p_prev = [t.clone() for t in x]
    z_prev = [t.clone() for t in x]
    u      = [torch.zeros(s) for s in shapes]
    v      = [torch.zeros(s) for s in shapes]
    return [x, y_prev, p_prev, z_prev, u, v]


def detach_state(state):
    return [[xi.detach().clone() for xi in s] if isinstance(s, list)
            else s for s in state]


# ============================================================
# 1. Zéro directions
# ============================================================
print("=== Zero dirs ===")
state       = make_initial_state()
zero_dirs   = [torch.zeros(s) for s in shapes]
residuals_zero = []

for step in range(N):
    with torch.no_grad():
        next_state, residual = one_iteration(state, zero_dirs, zero_dirs, step)
    residuals_zero.append(residual.item())
    print(f"iter {step:3d} | residual = {residual.item():.6f}")
    state = detach_state(next_state)
    if residual.item() < 1e-3:
        print(f"Converged a iteration {step}")
        break


# ============================================================
# 2. Random Directions 
# ============================================================
print("\n=== Random dirs ===")
state          = make_initial_state()
residuals_rand = []

for step in range(N):
    with torch.no_grad():
        dir_u = [torch.randn(s) for s in shapes]
        dir_v = [torch.randn(s) for s in shapes]
        next_state, residual = one_iteration(state, dir_u, dir_v, step)
    residuals_rand.append(residual.item())
    print(f"iter {step:3d} | residual = {residual.item():.6f}")
    state = detach_state(next_state)
    if residual.item() < 1e-3:
        print(f"Converged at iteration {step}")
        break


# ============================================================
# 3. Optimised Directions 
# ============================================================
print("\n=== OPT dirs ===")
state  = make_initial_state()
residuals_adam = []




for step in range(N):

    dir_u = [torch.randn(s, requires_grad=True) for s in shapes]
    dir_v = [torch.randn(s, requires_grad=True) for s in shapes]

    optimizer = torch.optim.Adam(dir_u + dir_v, lr=1e-3)


    for inner in range(n_inner):
        optimizer.zero_grad()

        next_state, res = one_iteration(state, dir_u, dir_v, step)
        zero_dirs = [torch.zeros_like(d) for d in dir_u]
        next_next_state, residual = one_iteration(
            next_state,
            dir_u,
            dir_v,
            step+1,)
        residual.backward()
        optimizer.step()
        if inner % 100 == 0:
            print(f"    inner {inner} | loss = {residual.item():.6f}")
    # ---- Avance l'état avec les meilleures directions
    with torch.no_grad():
        new_state, _ = one_iteration(state, dir_u, dir_v, step)
        _, residual_next = one_iteration(new_state, zero_dirs, zero_dirs, step + 1)
    print(f"iter {step:3d} | residual n+1 = {residual_next.item():.6f}")
    residuals_adam.append(residual_next.item())
    

    state = detach_state(new_state)

    if residual.item() < 1e-3:
        print(f"Converged at itération {step}")
        break 
    
    

# Plots

plt.figure(figsize=(10, 5))
plt.semilogy(residuals_zero, label="0", linewidth=2)
plt.semilogy(residuals_rand, label="Random", linewidth=2)
plt.semilogy(residuals_adam, label="Optimised", linewidth=2)
plt.xlabel("Iteration")
plt.ylabel("Residual ||p - z||")
plt.title("Comparaison in the choice of deviations ")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("comparison.pdf", dpi=150)
plt.show()
