"""
Total Generalized Variation tomography using PDHG.
"""

import numpy as np
import matplotlib.pyplot as plt
import odl

from odl.phantom.geometric import tgv_phantom
from odl.phantom.noise import white_noise
from odl.tomo.geometry.parallel import parallel_beam_geometry
from odl.tomo.operators.ray_trafo import RayTransform

from odl.operator.pspace_ops import (
    ProductSpaceOperator,
    ComponentProjection,
    ReductionOperator,
    BroadcastOperator,
)

from odl.solvers.functional.default_functionals import (
    ZeroFunctional,
    L2NormSquared,
    L1Norm,
    SeparableSum,
)

# --- Reconstruction space --- #
U = odl.uniform_discr(
    min_pt=[-20, -20],
    max_pt=[20, 20],
    shape=[300, 300],
    dtype="float32",
)

# --- Geometry / forward operator --- #
geometry = parallel_beam_geometry(U, num_angles=180, det_shape=200)

# Try the common backends in order
A = None
last_err = None
for impl in ("astra_cuda", "astra_cpu", "skimage", None):
    try:
        A = RayTransform(U, geometry, impl=impl)
        print(f"Using RayTransform backend: {impl}")
        break
    except Exception as err:
        last_err = err

if A is None:
    raise RuntimeError(
        "No RayTransform backend available. Install ASTRA toolbox or scikit-image."
    ) from last_err

# --- Generate artificial data --- #
phantom = odl.phantom.shepp_logan(U, modified=True)
plt.imsave("phantom.png", phantom.asarray(), cmap="gray")

data = A(phantom)
data += white_noise(A.range) * np.mean(data.asarray()) * 0.1
plt.imsave("sinogram.png", data.asarray(), cmap="gray")

# --- Set up inverse problem --- #
G = odl.Gradient(U, method="forward", pad_mode="symmetric")
V = G.range

Dx = odl.PartialDerivative(U, 0, method="backward", pad_mode="symmetric")
Dy = odl.PartialDerivative(U, 1, method="backward", pad_mode="symmetric")

# 2D symmetrized gradient for TGV2: 3 components, not 4
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

f = ZeroFunctional(domain)

l2_norm = L2NormSquared(A.range).translated(data)

alpha = 4e-1
beta = 1.0

l1_norm_1 = alpha * L1Norm(V)
l1_norm_2 = alpha * beta * L1Norm(W)

g = SeparableSum(l2_norm, l1_norm_1, l1_norm_2)

# --- PDHG parameters --- #
op_norm = 1.1 * odl.power_method_opnorm(op)
niter = 100
tau = 1.0 / op_norm
sigma = 1.0 / op_norm

callback = odl.solvers.CallbackPrintIteration()

x = op.domain.zero()

odl.solvers.pdhg(
    x, f, g, op,
    niter=niter,
    tau=tau,
    sigma=sigma,
    callback=callback,
)

# --- Save results --- #
plt.imsave("reconstruction.png", x[0].asarray(), cmap="gray")
plt.imsave("derivatives_x.png", x[1][0].asarray(), cmap="gray")
plt.imsave("derivatives_y.png", x[1][1].asarray(), cmap="gray")

print("Saved: phantom.png, sinogram.png, reconstruction.png, derivatives_x.png, derivatives_y.png")