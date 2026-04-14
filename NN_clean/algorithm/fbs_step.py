

import torch


def one_step(x, y_prev, p_prev, z_prev, u, v, n, params, C, RA):
    """
    One iteration the algorithm.

    Inputs:
        x, y_prev, p_prev, z_prev : lists of tensors
        u, v : lists of tensors (deviations)
        n : iteration index
        params : algorithm parameters
        C : forward operator
        RA : resolvent operator

    Returns:
        x_new, y, p, z, residual
    """

    gam  = params.gamma(n)
    a_n  = params.alpha(n)
    ab_n = params.alpha_bar(n)
    th_b = params.theta_bar(n)
    th_h = params.theta_hat(n)
    lam  = params.lam(n)


    y = [
        x[i] + a_n * (y_prev[i] - x[i]) + u[i]
        for i in range(len(x))
    ]


    z = [
        x[i]
        + a_n  * (p_prev[i] - x[i])
        + ab_n * (z_prev[i] - p_prev[i])
        + (th_b * gam * params.beta_bar / (th_h + 1e-12)) * u[i]
        + v[i]
        for i in range(len(x))
    ]


    Cy = C(y)

    z_minus_gCy = [
        z[i] - gam * Cy[i]
        for i in range(len(x))
    ]


    p = RA(z_minus_gCy, gam)

  
    x_new = [
        x[i]
        + lam * (p[i] - z[i])
        + ab_n * lam * (z_prev[i] - p_prev[i])
        for i in range(len(x))
    ]

    residual = torch.sqrt(
        sum((p[i] - y[i]).pow(2).sum() for i in range(len(x))) + 1e-12
    )

    return x_new, y, p, z, residual