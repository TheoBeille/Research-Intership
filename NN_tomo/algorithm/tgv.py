

import torch

def tgv(x_final, initial_state, A, noisy, alpha1=0.1, alpha0=0.1, eps=1e-8):
    """
    TGV objective for tomography:
        0.5 ||A u - noisy||^2 + alpha1 TV1 + alpha0 TV2
    """

    u, w1, w2, _ = extract_variables(x_final, initial_state)

    # DATA TERM (tomography)
    Au = A(u)  # forward projection
    data_term = 0.5 * torch.sum((Au - noisy) ** 2)

    # FIRST ORDER
    ux = u[:, :, 1:] - u[:, :, :-1]
    uy = u[:, 1:, :] - u[:, :-1, :]

    ux_crop = ux[:, :-1, :]
    uy_crop = uy[:, :, :-1]
    w1_crop = w1[:, :-1, :-1]
    w2_crop = w2[:, :-1, :-1]

    first_order_term = alpha1 * torch.sum(torch.sqrt(
        (ux_crop - w1_crop) ** 2 +
        (uy_crop - w2_crop) ** 2 +
        eps
    ))

    # SECOND ORDER
    w1_dx = w1[:, :, 1:] - w1[:, :, :-1]
    w1_dy = w1[:, 1:, :] - w1[:, :-1, :]
    w2_dx = w2[:, :, 1:] - w2[:, :, :-1]
    w2_dy = w2[:, 1:, :] - w2[:, :-1, :]

    e11 = w1_dx[:, :-1, :]
    e22 = w2_dy[:, :, :-1]
    e12 = 0.5 * (w1_dy[:, :, :-1] + w2_dx[:, :-1, :])

    second_order_term = alpha0 * torch.sum(torch.sqrt(
        e11 ** 2 + e22 ** 2 + 2 * e12 ** 2 + eps
    ))

    return data_term + first_order_term + second_order_term
def extract_variables(x_final, f_initial):


    u = x_final[0][:, 0, :, :]  # (B, H, W)
    w1 = x_final[1][:, 0, :, :]  # (B, H, W)
    w2 = x_final[1][:, 1, :, :]  # (B, H, W)
    
    
    return u, w1, w2, f_initial

