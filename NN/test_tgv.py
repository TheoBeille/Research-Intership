import torch
import numpy as np
import odl

# =========================
# IMPORTS DE TON CODE
# =========================
from Algo_setuptorch import get_setup
from algorithm.tgv import gradient, sym_grad


# =========================
# TORCH VERSION CORRECTE
# =========================
def tgv_torch(u, w, data, alpha=1e-1, beta=1.0):
    term_data = 0.5 * torch.sum((u - data) ** 2)

    Gu = gradient(u)
    term1 = alpha * torch.sum(torch.abs(Gu - w))

    Ew = sym_grad(w)
    term2 = alpha * beta * torch.sum(torch.abs(Ew))

    return term_data + term1 + term2


# =========================
# ODL VERSION
# =========================

def tgv_odl(u_np, w_np, data_np, setup, alpha=1e-1, beta=1.0):
    U = setup["space"]
    G = setup["D"]
    E = setup["E"]
    A = setup["A"]

    u_odl = U.element(u_np)
    V = G.range
    w_odl = V.element(w_np)
    data_odl = U.element(data_np)

    W = E.range

    l2_norm = odl.solvers.L2NormSquared(A.range).translated(data_odl)
    l1_norm_1 = alpha * odl.solvers.L1Norm(V)
    l1_norm_2 = alpha * beta * odl.solvers.L1Norm(W)

    term_data = 0.5 * l2_norm(A(u_odl))
    term1 = l1_norm_1(G(u_odl) - w_odl)
    term2 = l1_norm_2(E(w_odl))

    return term_data + term1 + term2
# =========================
# MAIN TEST
# =========================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    setup = get_setup(size=128, seed=0, noise_level=0.1, device=device)

    # data bruitée
    data_torch = setup["initial_state"].unsqueeze(1).to(device)

    # test: on choisit u et w à évaluer
    u = data_torch.clone()
    w = torch.randn((1, 2, 128, 128), device=device)

    # Torch
    F_torch = tgv_torch(u, w, data_torch)
    print("F_torch =", F_torch.item())

    # ODL
    u_np = u[0].detach().cpu().squeeze().numpy()
    w_np = w[0].detach().cpu().numpy()
    data_np = data_torch[0].detach().cpu().squeeze().numpy()

    F_odl = tgv_odl(u_np, w_np, data_np, setup)
    print("F_odl =", F_odl)
    print("ratio =", F_odl / F_torch.item())

# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()