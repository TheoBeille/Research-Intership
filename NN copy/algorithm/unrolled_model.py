

import torch
import torch.nn as nn

from algorithm.fbs_step import one_step
from algorithm.normalization import block_norm_sq

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

        self.dev_net= DeviationNet(
            n_channels=n_channels,
            hidden=net_hidden,
            n_layers=net_layers,
        )


    def _init_state(self, init_state):
        """
        Initialize x, y_prev, p_prev, z_prev, u, v.
        """
        dev = init_state.device
        B = init_state.shape[0]
        x = [
            init_state.clone(),
            torch.zeros((B, *self.shapes[1][1:]), device=dev),
            torch.zeros((B, *self.shapes[2][1:]), device=dev),
            torch.zeros((B, *self.shapes[3][1:]), device=dev),
        ]

        y_prev = [t.clone() for t in x]
        p_prev = [t.clone() for t in x]
        z_prev = [t.clone() for t in x]

        u = [torch.zeros_like(t) for t in x]
        v = [torch.zeros_like(t) for t in x]
        u_prev=[t.clone() for t in u]
        v_prev=[t.clone() for t in v]
        
        return x, y_prev, p_prev, z_prev, u, v,u_prev,v_prev

    def forward(self,initial_state, functions, return_all=False):
        """
        Args:
            initial_state
            functions: dict with keys
                - 'C'
                - 'RA'
                - 'compute_delta_torch'
                -'x0'
            return_all: if True, also returns internal trajectories

        Returns:
            p_prev, residuals
            optionally also states/history
        """
        C = functions["C"]
        RA = functions["RA"]
        K = functions["K"]
        compute_delta = functions["compute_delta_torch"]

        x, y_prev, p_prev, z_prev, u, v,u_prev,v_prev= self._init_state(initial_state)

        residuals = []
        AxCx=[]

        if return_all:
            x_hist, y_hist, p_hist, z_hist = [], [], [], []
            u_hist, v_hist, delta_hist = [], [], []
            
            
        if self.training: 
            T_rand = torch.randint(low=self.T, high=2*self.T, size=(1,)).item() 
        else:
            T_rand = self.T




        for n in range(T_rand):
        
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

          
            x_new = [t.float() for t in x_new]
            p = [t.float() for t in p]
            y = [t.float() for t in y]
            z = [t.float() for t in z]
            
            Cy=C(y)


            u, v = self.dev_net(
                shapes=self.shapes,
                x_blocks=x_new ,    
                p_blocks=p      ,    
                y_blocks=y     ,    
                z_blocks=z    ,     
                u_prev=u_prev ,    
                v_prev=v_prev , 
                Cy=Cy,
            )
          


            x, y_prev, p_prev, z_prev = x_new, y, p, z
            
            u_prev = [u_i.clone() for u_i in u]
            v_prev = [v_i.clone() for v_i in v]

            res = torch.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)
            residuals.append(res)
            Kx = K(x)      # liste de 4 tenseurs
            Cx = C(x)      # liste de 4 tenseurs
            AxCx.append(sum(torch.mean((k + c)**2) for k, c in zip(Kx, Cx)))

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
            return AxCx, residuals, history

        return AxCx, residuals