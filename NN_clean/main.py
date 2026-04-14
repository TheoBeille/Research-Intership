# main.py

import torch

# Disable cuDNN to avoid compatibility issues
torch.backends.cudnn.enabled = False

from Algo_setuptorch import Params
from data.dataset import build_train_test_data
from algorithm.fbs_step import one_step
from algorithm.unrolled_model import UnrolledFBS
from training.train import train

# ============================
# CONFIG
# ============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

params = Params()

TRAIN_SEEDS = list(range(5))   # petit pour debug
TEST_SEEDS  = [10]

# IMPORTANT → adapte selon ton projet
size = 16
SHAPES = [
    (1, 1, size, size),
    (1, 2, size, size),
    (1, 2, size, size),
    (1, 4, size, size),
]
N_CH = sum(s[1] for s in SHAPES)


def run_zero(noisy, functions, params, shapes, T, device):
    C = functions["C"]
    RA = functions["RA"]

    B = noisy.shape[0]
    x = []
    for i, s in enumerate(shapes):
        _, Cb, H, W = s
        if i == 0:
            x.append(noisy.clone())
        else:
            x.append(torch.zeros((B, Cb, H, W), device=device))

    y_prev = [t.clone() for t in x]
    p_prev = [t.clone() for t in x]
    z_prev = [t.clone() for t in x]
    u = [torch.zeros_like(t) for t in x]
    v = [torch.zeros_like(t) for t in x]

    residuals = []
    with torch.no_grad():
        for n in range(T):
            x, y_prev, p_prev, z_prev, res = one_step(
                x, y_prev, p_prev, z_prev, u, v, n, params, C, RA
            )
            residuals.append(res.item())

    return residuals


def run_learned(model, noisy, functions):
    model.eval()
    with torch.no_grad():
        _, residuals = model(noisy, functions)
    return [r.item() for r in residuals]

