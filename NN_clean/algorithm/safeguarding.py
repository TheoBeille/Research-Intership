# algorithm/safeguarding.py

import torch
import torch.nn as nn


class SafeguardingLayer(nn.Module):
    """
    Safeguarding layer for the learned deviations (u, v).

    It rescales the proposed deviations so that they satisfy the paper's
    admissibility condition, while preserving their direction.
    """

    def __init__(self, params, eps: float = 1e-12, extra_budget: float = 1e-4):
        super().__init__()
        self.params = params
        self.eps = eps
        self.extra_budget = extra_budget

    @staticmethod
    def _sum_sq(blocks):
        return sum(b.pow(2).sum() for b in blocks)

    def forward(self, u_raw, v_raw, delta, n: int):
        p = self.params

        lam = float(p.lam(n + 1))
        mu = float(p.mu(n + 1))
        lpm = lam + mu

        theta_hat = max(float(p.theta_hat(n + 1)), self.eps)
        theta = max(float(p.theta(n + 1)), self.eps)
        theta_tilde = float(p.theta_tilde(n + 1))

        c_u = lpm * theta_tilde / theta_hat
        c_v = lpm * theta_hat / theta

        norm_u_sq = self._sum_sq(u_raw)
        norm_v_sq = self._sum_sq(v_raw)

        Q = c_u * norm_u_sq + c_v * norm_v_sq + self.eps

        delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        budget = float(p.zeta) * (delta.clamp(min=0.0) + self.extra_budget)

        ratio = (budget / Q).clamp(min=0.0, max=1.0)
        alpha = torch.sqrt(ratio)

        u_safe = [
            torch.nan_to_num(alpha * u, nan=0.0, posinf=0.0, neginf=0.0)
            for u in u_raw
        ]
        v_safe = [
            torch.nan_to_num(alpha * v, nan=0.0, posinf=0.0, neginf=0.0)
            for v in v_raw
        ]

        return u_safe, v_safe