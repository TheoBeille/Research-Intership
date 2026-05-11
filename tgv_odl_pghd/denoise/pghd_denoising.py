"""Total Generalized Variation denoising using PDHG."""

import numpy as np
import odl
import matplotlib.pyplot as plt

from odl.phantom.noise import white_noise
from odl.operator.pspace_ops import (
    ProductSpaceOperator,
    ComponentProjection,
    ReductionOperator,
    BroadcastOperator,
)

def pdhg():
    U = odl.uniform_discr(
        min_pt=[-20, -20], max_pt=[20, 20], shape=[128, 128], dtype="float32"
    )

    A = odl.IdentityOperator(U)

    phantom = odl.phantom.shepp_logan(U, modified=True)
    data = A(phantom)
    data += white_noise(A.range) * np.mean(data.asarray()) * 0.1

    plt.imsave("phantom.png", phantom.asarray(), cmap="gray")
    plt.imsave("data.png", data.asarray(), cmap="gray")

    G = odl.Gradient(U, method="forward", pad_mode="symmetric")
    V = G.range

    Dx = odl.PartialDerivative(U, 0, method="backward", pad_mode="symmetric")
    Dy = odl.PartialDerivative(U, 1, method="backward", pad_mode="symmetric")

    # 2D symmetrized gradient: 3 components
    E = ProductSpaceOperator(
        [
            [Dx, 0],
            [0, Dy],
            [0.5 * Dy, 0.5 * Dx],
        ]
    )
    W = E.range

    domain = odl.ProductSpace(U, V)

    op = BroadcastOperator(
        A * ComponentProjection(domain, 0),
        ReductionOperator(G, odl.ScalingOperator(V, -1)),
        E * ComponentProjection(domain, 1),
    )

    alpha = 1e-1
    beta = 1.0

    f = odl.solvers.ZeroFunctional(domain)
    l2_norm = odl.solvers.L2NormSquared(A.range).translated(data)
    l1_norm_1 = alpha * odl.solvers.L1Norm(V)
    l1_norm_2 = alpha * beta * odl.solvers.L1Norm(W)
    g = odl.solvers.SeparableSum(l2_norm, l1_norm_1, l1_norm_2)

    op_norm = 1.1 * odl.power_method_opnorm(op)

    niter = 500
    tau = 1.0 / op_norm
    sigma = 1.0 / op_norm

    x = op.domain.zero()
    F_values = []

    class CallbackStore(odl.solvers.Callback):
        def __init__(self):
            self.iter = 0

        def __call__(self, x):
            val = f(x) + g(op(x))
            val = float(val)
            F_values.append(val)
            self.iter += 1
            print(f"Iter {self.iter}: F(x) = {val:.6f}")

    callback = (
        odl.solvers.CallbackPrintIteration()
        & CallbackStore()
    )

    odl.solvers.pdhg(
        x, f, g, op,
        niter=niter,
        tau=tau,
        sigma=sigma,
        callback=callback,
    )

    return x, F_values

