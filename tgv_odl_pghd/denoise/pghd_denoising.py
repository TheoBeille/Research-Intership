"""Total Generalized Variation denoising using PDHG."""

import numpy as np
import odl
import matplotlib.pyplot as plt
from odl.phantom.geometric import tgv_phantom
from odl.phantom.noise import white_noise
from odl.operator.pspace_ops import (
    ProductSpaceOperator,
    ComponentProjection,
    ReductionOperator,
    BroadcastOperator,
)

# --- Set up the forward operator (identity) --- #

U = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[128, 128], dtype="float32"
)

A = odl.IdentityOperator(U)

# --- Generate artificial data --- #

phantom = odl.phantom.shepp_logan(U, modified=True)


data = A(phantom)
data += white_noise(A.range) * np.mean(data) *0.1

plt.imsave("phantom.png", phantom.asarray(), cmap="gray")
plt.imsave("data.png", data.asarray(), cmap="gray")
# --- Set up the inverse problem --- #

G = odl.Gradient(U, method="forward", pad_mode="symmetric")
V = G.range

Dx = odl.PartialDerivative(U, 0, method="backward", pad_mode="symmetric")
Dy = odl.PartialDerivative(U, 1, method="backward", pad_mode="symmetric")

E = ProductSpaceOperator(
    [
        [Dx, 0],
        [0, Dy],
        [0.5 * Dy, 0.5 * Dx],
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
beta = 1

f = odl.solvers.ZeroFunctional(domain)
l2_norm = odl.solvers.L2NormSquared(A.range).translated(data)

l1_norm_1 = alpha * odl.solvers.L1Norm(V)
l1_norm_2 = alpha * beta * odl.solvers.L1Norm(W)

g = odl.solvers.SeparableSum(l2_norm, l1_norm_1, l1_norm_2)

op_norm = 1.1 * odl.power_method_opnorm(op)

niter = 100
tau = 1.0 / op_norm
sigma = 1.0 / op_norm


x = op.domain.zero()


F_values = []

class CallbackStore(odl.solvers.Callback):
    def __init__(self):
        self.iter = 0

    def __call__(self, x):
        val = f(x) + g(op(x))
        F_values.append(val)
        self.iter += 1
        print(f"Iter {self.iter}: F(x) = {val:.6f}")


callback = (
    odl.solvers.CallbackPrintIteration()
    & odl.solvers.CallbackShow(step=10, indices=0)
    & CallbackStore()
)


odl.solvers.pdhg(x, f, g, op, niter=500, tau=tau, sigma=sigma, callback=callback)

plt.imsave("reconstruction.png", x[0].asarray(), cmap="gray")
plt.imsave("derivatives.png", x[1][0].asarray(), cmap="gray")

plt.close()
import matplotlib.pyplot as plt

F_values = np.array(F_values)
F_star = F_values[-1]
F_values=F_values[:100]
iters = np.arange(1, len(F_values) + 1)

gap = F_values - F_star
gap[gap <= 1e-16] = 1e-16  

plt.loglog(iters, gap, label="F(x_n) - F*")


C1 = gap[0] * iters[0]      #  -1
C2 = gap[0] * iters[0]**2   #  -2

ref1 = C1 / iters        
ref2 = C2 / (iters**2)     

plt.loglog(iters, ref1, '--', label="O(1/k)")
plt.loglog(iters, ref2, '--', label="O(1/k²)")

plt.xlabel("Iteration")
plt.ylabel("F(x_n) - F*")
plt.title("Convergence PDHG (log-log)")
plt.legend()
plt.grid(which="both")

plt.savefig("convergence_loglog.png")
plt.show()