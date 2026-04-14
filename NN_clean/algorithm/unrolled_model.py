

import torch
import torch.nn as nn

from algorithm.fbs_step import one_step
from algorithm.normalization import normalize_and_scale
from algorithm.safeguarding import SafeguardingLayer
from models.deviation_net import DeviationNet


class UnrolledFBS(nn.Module):
    """
    Unrolled forward-backward splitting model with learned deviations.

    Structure:
        -> learned raw deviations
        -> normalization / scaling
        -> safeguarding
        -> next iteration
    """

    def __init__(self, params, shapes, n_channels, T=20, net_hidden=32, net_layers=2, alpha=0.99):
        super().__init__()

        self.params = params
        self.shapes = shapes
        self.n_channels = n_channels
        self.n_blocks = len(shapes)
        self.T = T
        self.alpha = alpha

        self.dev_net = DeviationNet(
            n_channels=n_channels,
            hidden=net_hidden,
            n_layers=net_layers,
        )
        self.sg_layer = SafeguardingLayer(params)

    def _init_state(self, noisy):
        """
        Initialize x, y_prev, p_prev, z_prev, u, v.
        """
        dev = noisy.device

        x = [
            noisy.clone(),
            torch.zeros(self.shapes[1], device=dev),
            torch.zeros(self.shapes[2], device=dev),
            torch.zeros(self.shapes[3], device=dev),
        ]

        y_prev = [t.clone() for t in x]
        p_prev = [t.clone() for t in x]
        z_prev = [t.clone() for t in x]

        u = [torch.zeros_like(t) for t in x]
        v = [torch.zeros_like(t) for t in x]

        return x, y_prev, p_prev, z_prev, u, v

    def forward(self, noisy, functions, return_all=False):
        """
        Args:
            noisy: tensor [B,1,H,W]
            functions: dict with keys
                - 'C'
                - 'RA'
                - 'compute_delta_torch'
            return_all: if True, also returns internal trajectories

        Returns:
            p_prev, residuals
            optionally also states/history
        """
        C = functions["C"]
        RA = functions["RA"]
        compute_delta = functions["compute_delta_torch"]

        x, y_prev, p_prev, z_prev, u, v = self._init_state(noisy)

        residuals = []

        if return_all:
            x_hist, y_hist, p_hist, z_hist = [], [], [], []
            u_hist, v_hist, delta_hist = [], [], []

        for n in range(self.T):
        
            x_new, y, p, z, res = one_step(
                x=x,
                y_prev=y_prev,
                p_prev=p_prev,
                z_prev=z_prev,
                u=u,
                v=v,
                n=n,
                params=self.params,
                C=C,
                RA=RA,
            )

            delta = compute_delta(
                p, x_new, p_prev, z, z_prev, y, y_prev, u, v, n
            )
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            # Ensure all tensors are float32
            x_new = [t.float() for t in x_new]
            p = [t.float() for t in p]
            y = [t.float() for t in y]
            z = [t.float() for t in z]

            u_raw, v_raw = self.dev_net(
                x_blocks=x_new,
                p_blocks=p,
                y_blocks=y,
                z_blocks=z,
                noisy=noisy,
                shapes=self.shapes,
            )


            u_scaled, v_scaled = normalize_and_scale(
                u_raw=u_raw,
                v_raw=v_raw,
                delta=delta,
                alpha=self.alpha,
            )


            u, v = self.sg_layer(
                u_raw=u_scaled,
                v_raw=v_scaled,
                delta=delta,
                n=n,
            )


            x, y_prev, p_prev, z_prev = x_new, y, p, z

            res = torch.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)
            residuals.append(res)

            if return_all:
                x_hist.append([t.clone() for t in x])
                y_hist.append([t.clone() for t in y_prev])
                p_hist.append([t.clone() for t in p_prev])
                z_hist.append([t.clone() for t in z_prev])
                u_hist.append([t.clone() for t in u])
                v_hist.append([t.clone() for t in v])
                delta_hist.append(delta.clone() if torch.is_tensor(delta) else delta)

        if return_all:
            history = {
                "x": x_hist,
                "y": y_hist,
                "p": p_hist,
                "z": z_hist,
                "u": u_hist,
                "v": v_hist,
                "delta": delta_hist,
            }
            return p_prev, residuals, history

        return p_prev, residuals