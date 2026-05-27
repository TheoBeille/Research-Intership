
import numpy as np
import torch
from algorithm.fbs_step import one_step
from algorithm.normalization import block_norm_sq





def run_learned(model, initial_state, clean, functions, T_test=20, return_all=False):
    model.eval()

    C = functions["C"]
    RA = functions["RA"]
  
    compute_delta = functions["compute_delta_torch"]

    x, y_prev, p_prev, z_prev, u, v, u_prev, v_prev = model._init_state(initial_state)

    residuals = []

    AxCx = []

    if return_all:
        x_hist, y_hist, p_hist, z_hist = [], [], [], []
        u_hist, v_hist, delta_hist = [], [], []

    with torch.no_grad():
        for n in range(T_test):
            print(f'iter:{n}')
            


           
         
            x_new, y, p, z, res = one_step(
                x=x, y_prev=y_prev, p_prev=p_prev, z_prev=z_prev,
                u=u, v=v, n=n, params=model.params, C=C, RA=RA,
            )

            delta = compute_delta(p, x, p_prev, z, z_prev, y, y_prev, u, v, n)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

            x_new = [t.float() for t in x_new]
            p = [t.float() for t in p]
            y = [t.float() for t in y]
            z = [t.float() for t in z]

            Cy = C(y)

            u_raw, v_raw = model.dev_nets[n](
                shapes=model.shapes,
                x_blocks=x_new,
                p_blocks=p,
                y_blocks=y,
                z_blocks=z,
                u_prev=u_prev,
                v_prev=v_prev,
                Cy=Cy,
              )
            params = model.params
            lam = float(params.lam(n + 1))
            mu = float(params.mu(n + 1))
            lpm = lam + mu

            theta_hat = float(params.theta_hat(n + 1))
            theta = float(params.theta(n + 1))
            theta_tilde = float(params.theta_tilde(n + 1))

            c_u = lpm * theta_tilde / theta_hat
            c_v = lpm * theta_hat / theta

            norm_u_sq = block_norm_sq(u_raw)
            norm_v_sq = block_norm_sq(v_raw)

            Q = c_u * norm_u_sq + c_v * norm_v_sq
            budget = float(params.zeta) * delta.clamp(min=0.0)

            ratio = torch.sqrt(budget / (Q + 1e-12))
            scale = model.alpha * ratio

            u = [scale * u_i for u_i in u_raw]
            v = [scale * v_i for v_i in v_raw]



                
                
                
            x, y_prev, p_prev, z_prev = x_new, y, p, z
            u_prev = [u_i.clone() for u_i in u]
            v_prev = [v_i.clone() for v_i in v]

            res = torch.nan_to_num(res, nan=1e6, posinf=1e6, neginf=1e6)
            residuals.append(res.item())
            val = functions['kkt_residual_norm'](x)
            
            print(val.item())

            AxCx.append(val.item())
                    
                
            if return_all:
                x_hist.append([t.clone() for t in x])
                y_hist.append([t.clone() for t in y_prev])
                p_hist.append([t.clone() for t in p_prev])
                z_hist.append([t.clone() for t in z_prev])
                u_hist.append([t.clone() for t in u])
                v_hist.append([t.clone() for t in v])
                delta_hist.append(delta.clone())

            
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
        return AxCx,residuals, history
    
    
    

    return   AxCx,residuals


"""             else:
                
                u = [torch.zeros_like(t) for t in x]
                v = [torch.zeros_like(t) for t in x]

                x_new, y, p, z, res = one_step(
                    x=x, y_prev=y_prev, p_prev=p_prev, z_prev=z_prev,
                    u=u, v=v, n=n, params=model.params, C=C, RA=RA,
                ) """
                

def run_zero(initial_state,functions, params, shapes, T, device):
    C = functions["C"]
    RA = functions["RA"]

    B = initial_state.shape[0]

    AxCx = []
    x=[]
    for i, s in enumerate(shapes):
        _, Cb, H, W = s
        if i == 0:
            x.append(initial_state.clone())
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
            print(f"iter:{n}")
            x, y_prev, p_prev, z_prev, res = one_step(
                x, y_prev, p_prev, z_prev, u, v, n, params, C, RA
            )
  
            val = functions['kkt_residual_norm'](x)
            print(val.item())

            AxCx.append(val.item())
            residuals.append(res.item())


    return AxCx,residuals,x


