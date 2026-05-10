# -*- coding: utf-8 -*-
#
#   PyCUDA primal-dual algorithms for TV/TGV-constrained imaging problems
#
#   Copyright (C) 2013, 2025 Kristian Bredies (kristian.bredies@uni-graz.at)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   If you consider this code useful, please cite:
#
#   Kristian Bredies. Recovering piecewise smooth multichannel images by
#   minimization of convex functionals with total generalized variation
#   penalty. Lecture Notes in Computer Science, 8293:44-77, 2014.

import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import compiler
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel
from numpy import float32, int32, zeros
from common_pycuda import block_size_x, block_size_y, block_, \
    get_grid, gpuarray_copy, visualize, prepare_data, display

# kernel code
kernels = """
    __global__ void tv_update_p(float *u, float *p, float alpha_inv,
                                float tau_d, int nx, int ny, int chans) {
    __shared__ float l_px[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_py[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      float pabs;
      pabs = 0.0f;

      for (int i=0; i<chans; i++) {
        float u00 =              u[(i*ny+y)  *nx+x];
        float u10 = (x < nx-1) ? u[(i*ny+y)  *nx+x+1] : u00;
        float u01 = (y < ny-1) ? u[(i*ny+y+1)*nx+x] : u00;

        float dxp = u10-u00;
        float dyp = u01-u00;

        float px = p[((2*i)  *ny+y)*nx+x] + tau_d*dxp;
        float py = p[((2*i+1)*ny+y)*nx+x] + tau_d*dyp;

        l_px[threadIdx.x][threadIdx.y][i] = px;
        l_py[threadIdx.x][threadIdx.y][i] = py;

        pabs += px*px + py*py;
      }

      pabs = sqrtf(pabs)*alpha_inv;
      pabs = (pabs > 1.0f) ? 1.0f/pabs : 1.0f;

      for (int i=0; i<chans; i++) {
        p[((2*i)  *ny+y)*nx+x] = l_px[threadIdx.x][threadIdx.y][i]*pabs;
        p[((2*i+1)*ny+y)*nx+x] = l_py[threadIdx.x][threadIdx.y][i]*pabs;
      }
    }
    }

    __global__ void tv_update_u(float *u, float *p, float *f,
                                float tau_p1inv, int nx, int ny,
                                int chans) {

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      for (int i=0; i<chans; i++) {
        float div = f[(i*ny+y)*nx+x];
        if (x < nx-1) div += p[((2*i  )*ny+y  )*nx+x];
        if (x > 0)    div -= p[((2*i  )*ny+y  )*nx+x-1];
        if (y < ny-1) div += p[((2*i+1)*ny+y  )*nx+x];
        if (y > 0)    div -= p[((2*i+1)*ny+y-1)*nx+x];

        u[(i*ny+y)*nx+x] = div + tau_p1inv*(u[(i*ny+y)*nx+x] - div);
      }
    }
    }

    __global__ void tv_update_u_norm1(float *u, float *p, float *f,
                                      float tau_p, int nx, int ny,
                                      int chans) {

    float l_u[4];
    float l_f[4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      float acc = 0.0f;

      for (int i=0; i<chans; i++) {
        l_f[i] = f[(i*ny+y)*nx+x];

        float div = 0.0f;

        if (x < nx-1) div += p[((2*i  )*ny+y  )*nx+x];
        if (x > 0)    div -= p[((2*i  )*ny+y  )*nx+x-1];
        if (y < ny-1) div += p[((2*i+1)*ny+y  )*nx+x];
        if (y > 0)    div -= p[((2*i+1)*ny+y-1)*nx+x];

        l_u[i] = u[(i*ny+y)*nx+x] + tau_p*div - l_f[i];
        acc += l_u[i]*l_u[i];
      }

      acc = sqrt(acc);
      if (acc > tau_p) acc = (acc - tau_p)/acc;
      else acc = 0.0f;

      for (int i=0; i<chans; i++)
        u[(i*ny+y)*nx+x] = l_u[i]*acc + l_f[i];
    }
    }

    __global__ void tgv_update_p(float *u, float *w, float *p,
                                 float alpha_inv, float tau_d,
                                 int nx, int ny, int chans) {

    __shared__ float l_px[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_py[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      float pabs;
      pabs = 0.0f;

      for (int i=0; i<chans; i++) {
        float u00 =              u[(i*ny+y)  *nx+x];
        float u10 = (x < nx-1) ? u[(i*ny+y)  *nx+x+1] : u00;
        float u01 = (y < ny-1) ? u[(i*ny+y+1)*nx+x]   : u00;

        float dxp = u10-u00;
        float dyp = u01-u00;

        float px = p[((2*i)  *ny+y)*nx+x]
                   + tau_d*(dxp - w[((2*i)  *ny+y)*nx+x]);
        float py = p[((2*i+1)*ny+y)*nx+x]
                   + tau_d*(dyp - w[((2*i+1)*ny+y)*nx+x]);

        l_px[threadIdx.x][threadIdx.y][i] = px;
        l_py[threadIdx.x][threadIdx.y][i] = py;

        pabs += px*px + py*py;
      }

      pabs = sqrtf(pabs)*alpha_inv;
      pabs = (pabs > 1.0f) ? 1.0f/pabs : 1.0f;

      for (int i=0; i<chans; i++) {
        p[((2*i)  *ny+y)*nx+x] = l_px[threadIdx.x][threadIdx.y][i]*pabs;
        p[((2*i+1)*ny+y)*nx+x] = l_py[threadIdx.x][threadIdx.y][i]*pabs;
      }
    }
    }

    __global__ void tgv_update_q(float *w, float *q, float alpha_inv,
                                 float tau_d, int nx, int ny, int chans) {

    __shared__ float l_qxx[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_qyy[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];
    __shared__ float l_qxy[%(BLOCK_SIZE_X)s][%(BLOCK_SIZE_Y)s][4];

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      float qabs;
      qabs = 0.0f;

      for (int i=0; i<chans; i++) {
        float wx00 =              w[((2*i)*ny+y  )*nx+x];
        float wx10 = (x < nx-1) ? w[((2*i)*ny+y  )*nx+x+1] : wx00;
        float wx01 = (y < ny-1) ? w[((2*i)*ny+y+1)*nx+x]   : wx00;

        float wy00 =              w[((2*i+1)*ny+y  )*nx+x];
        float wy10 = (x < nx-1) ? w[((2*i+1)*ny+y  )*nx+x+1] : wy00;
        float wy01 = (y < ny-1) ? w[((2*i+1)*ny+y+1)*nx+x]   : wy00;

        float wxx = wx10-wx00;
        float wyy = wy01-wy00;
        float wxy = 0.5f*(wx01-wx00+wy10-wy00);

        float qxx = q[((3*i  )*ny+y)*nx+x] + tau_d*wxx;
        float qyy = q[((3*i+1)*ny+y)*nx+x] + tau_d*wyy;
        float qxy = q[((3*i+2)*ny+y)*nx+x] + tau_d*wxy;

        l_qxx[threadIdx.x][threadIdx.y][i] = qxx;
        l_qyy[threadIdx.x][threadIdx.y][i] = qyy;
        l_qxy[threadIdx.x][threadIdx.y][i] = qxy;

        qabs += qxx*qxx + qyy*qyy + 2.0f*qxy*qxy;
      }

      qabs = sqrtf(qabs)*alpha_inv;
      qabs = (qabs > 1.0f) ? 1.0f/qabs : 1.0f;

      for (int i=0; i<chans; i++) {
        q[((3*i)  *ny+y)*nx+x] = l_qxx[threadIdx.x][threadIdx.y][i]*qabs;
        q[((3*i+1)*ny+y)*nx+x] = l_qyy[threadIdx.x][threadIdx.y][i]*qabs;
        q[((3*i+2)*ny+y)*nx+x] = l_qxy[threadIdx.x][threadIdx.y][i]*qabs;
      }
    }
    }

    __global__ void tgv_update_w(float *w, float *p, float *q,
                                 float tau_p, int nx, int ny, int chans) {

    float div1, div2;

    int x = blockIdx.x * %(BLOCK_SIZE_X)s + threadIdx.x;
    int y = blockIdx.y * %(BLOCK_SIZE_Y)s + threadIdx.y;

    if ((x < nx) && (y < ny)) {
      for (int i=0; i<chans; i++) {
        div1 = p[((2*i)*ny+y)*nx+x];
        if (x < nx-1) div1 += q[((3*i  )*ny+y  )*nx+x];
        if (x > 0)    div1 -= q[((3*i  )*ny+y  )*nx+x-1];
        if (y < ny-1) div1 += q[((3*i+2)*ny+y  )*nx+x];
        if (y > 0)    div1 -= q[((3*i+2)*ny+y-1)*nx+x];

        div2 = p[((2*i+1)*ny+y)*nx+x];
        if (x < nx-1) div2 += q[((3*i+2)*ny+y  )*nx+x];
        if (x > 0)    div2 -= q[((3*i+2)*ny+y  )*nx+x-1];
        if (y < ny-1) div2 += q[((3*i+1)*ny+y  )*nx+x];
        if (y > 0)    div2 -= q[((3*i+1)*ny+y-1)*nx+x];

        w[((2*i  )*ny+y)*nx+x] += tau_p*div1;
        w[((2*i+1)*ny+y)*nx+x] += tau_p*div2;
      }
    }
    }
    """
kernels = kernels % {
    'BLOCK_SIZE_X': block_size_x,
    'BLOCK_SIZE_Y': block_size_y,
}

# compile kernels
module = compiler.SourceModule(kernels)
tv_update_p_func = module.get_function("tv_update_p")
tv_update_u_func = module.get_function("tv_update_u")
tv_update_u_norm1_func = module.get_function("tv_update_u_norm1")
tv_update_u_ = ElementwiseKernel("float *u_, float *u",
                                 "u_[i] = 2.0f*u[i] - u_[i]",
                                 "extragradient_update")
tgv_update_p_func = module.get_function("tgv_update_p")
tgv_update_q_func = module.get_function("tgv_update_q")
tgv_update_u_func = module.get_function("tv_update_u")
tgv_update_u_norm1_func = module.get_function("tv_update_u_norm1")
tgv_update_w_func = module.get_function("tgv_update_w")
tgv_update_u_ = tv_update_u_
tgv_update_w_ = tgv_update_u_




import numpy as np

def _diff_x(x):
    dx = np.zeros_like(x)
    dx[:, :-1, ...] = x[:, 1:, ...] - x[:, :-1, ...]
    return dx

def _diff_y(x):
    dy = np.zeros_like(x)
    dy[:-1, :, ...] = x[1:, :, ...] - x[:-1, :, ...]
    return dy

def tgv2_energy_terms_np(u, w, f, alpha1, alpha0, eps=1e-12):
    """
    u : (H, W, C) or (H, W)
    w : (H, W, 2*C)  with w[..., 2*c] = w1 and w[..., 2*c+1] = w2
    f : same shape as u
    Returns:
        data_term, tgv1_term, tgv2_term, total
    """
    u = np.asarray(u, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    f = np.asarray(f, dtype=np.float32)

    if u.ndim == 2:
        u = u[..., None]
    if f.ndim == 2:
        f = f[..., None]
    if w.ndim == 2:
        w = w[..., None]

    if u.shape != f.shape:
        raise ValueError(f"u and f must have same shape, got {u.shape} and {f.shape}")

    H, W, C = u.shape
    if w.shape[:2] != (H, W) or w.shape[2] != 2 * C:
        raise ValueError(f"w must have shape (H, W, 2*C). Got u={u.shape}, w={w.shape}")

    data_term = 0.5 * np.sum((u - f) ** 2)

    ux = _diff_x(u)
    uy = _diff_y(u)

    tgv1 = 0.0
    tgv2 = 0.0

    for c in range(C):
        w1 = w[..., 2 * c]
        w2 = w[..., 2 * c + 1]

        # ||∇u - w||_{2,1}
        tgv1 += np.sum(np.sqrt((ux[..., c] - w1) ** 2 +
                               (uy[..., c] - w2) ** 2 + eps))

        # symmetrized gradient E(w)
        w1x = _diff_x(w1)
        w1y = _diff_y(w1)
        w2x = _diff_x(w2)
        w2y = _diff_y(w2)

        e11 = w1x
        e22 = w2y
        e12 = 0.5 * (w1y + w2x)

        tgv2 += np.sum(np.sqrt(e11 ** 2 + e22 ** 2 + 2.0 * e12 ** 2 + eps))

    tgv1 *= alpha1
    tgv2 *= alpha0
    total = data_term + tgv1 + tgv2
    return data_term, tgv1, tgv2, total





def tv_update_p(u, p, alpha, tau_d):
    tv_update_p_func(u, p, float32(1.0 / alpha), float32(tau_d),
                     int32(u.shape[0]), int32(u.shape[1]), int32(u.shape[2]),
                     block=block_, grid=get_grid(u))


def tv_update_u(u, p, f, tau_p, norm):
    if norm == 2:
        tv_update_u_func(u, p, f, float32(1.0 / (1.0 + tau_p)),
                         int32(u.shape[0]), int32(u.shape[1]),
                         int32(u.shape[2]),
                         block=block_, grid=get_grid(u))

    if norm == 1:
        tv_update_u_norm1_func(u, p, f, float32(tau_p),
                               int32(u.shape[0]), int32(u.shape[1]),
                               int32(u.shape[2]),
                               block=block_, grid=get_grid(u))


def tv_denoise(f, alpha=0.1, norm=2, maxiter=500, vis=-1):
    """Perform total variation denoising of f with regularization parameter
    alpha and maxiter iterations."""

    f = prepare_data(f)
    f_gpu = gpuarray.to_gpu(f)

    u = gpuarray_copy(f_gpu)
    u_ = gpuarray_copy(u)
    p = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')

    L2 = 8.0
    k = 0

    if vis > 0:
        display("Denoising image")
        u_vis = zeros(f.shape, 'float32', order='F')
    else:
        u_vis = None

    while k < maxiter:
        tau_p = 1.0 / (k + 1)
        tau_d = 1.0 / tau_p / L2

        #############
        # dual update
        tv_update_p(u_, p, alpha, tau_d)

        ###############
        # primal update
        cuda.memcpy_dtod(u_.gpudata, u.gpudata, u.nbytes)
        tv_update_u(u, p, f_gpu, tau_p, norm)

        ######################
        # extragradient update
        tv_update_u_(u_, u)

        if (vis > 0) and (k % vis == 0):
            display(".")
            cuda.memcpy_dtoh(u_vis.data, u.gpudata)
            visualize(u_vis)

        k += 1

    if vis > 0:
        display("\n")

    return u.get().squeeze()


def tgv_update_p(u, w, p, tau_d, alpha):
    tgv_update_p_func(u, w, p, float32(1.0 / alpha), float32(tau_d),
                      int32(u.shape[0]), int32(u.shape[1]), int32(u.shape[2]),
                      block=block_, grid=get_grid(u))


def tgv_update_q(w, q, tau_d, alpha):
    tgv_update_q_func(w, q, float32(1.0 / alpha), float32(tau_d),
                      int32(w.shape[0]), int32(w.shape[1]),
                      int32(w.shape[2] / 2),
                      block=block_, grid=get_grid(w))


def tgv_update_u(u, p, f, tau_p, norm):
    if norm == 2:
        tgv_update_u_func(u, p, f, float32(1.0 / (1.0 + tau_p)),
                          int32(u.shape[0]), int32(u.shape[1]),
                          int32(u.shape[2]),
                          block=block_, grid=get_grid(u))
    if norm == 1:
        tgv_update_u_norm1_func(u, p, f, float32(tau_p),
                                int32(u.shape[0]), int32(u.shape[1]),
                                int32(u.shape[2]),
                                block=block_, grid=get_grid(u))


def tgv_update_w(w, p, q, tau_p):
    tgv_update_w_func(w, p, q, float32(tau_p),
                      int32(w.shape[0]), int32(w.shape[1]),
                      int32(w.shape[2] / 2),
                      block=block_, grid=get_grid(w))


def tgv_denoise(f, alpha1=0.1, fac=2.0, norm=2, maxiter=500, vis=-1, return_terms=False):
    """Perform second-order total generalized variation denoising
    of f with regularization parameters (alpha0,alpha1)=(fac*alpha1,alpha1)
    and maxiter iterations.
    """

    # primal variables
    f = prepare_data(f)
    f_cpu = f.copy()
    f_gpu = gpuarray.to_gpu(f)

    u = gpuarray_copy(f_gpu)
    u_ = gpuarray_copy(u)
    w = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')
    w_ = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                        'float32', order='F')

    # dual variables
    p = gpuarray.zeros([u.shape[0], u.shape[1], 2 * u.shape[2]],
                       'float32', order='F')
    q = gpuarray.zeros([u.shape[0], u.shape[1], 3 * u.shape[2]],
                       'float32', order='F')

    alpha0 = fac * alpha1
    L2 = 12.0
    k = 0

    residuals = []
    F_vals = []
    F_data = []
    F_tgv1 = []
    F_tgv2 = []

    if vis > 0:
        display("Denoising image")
        u_vis = zeros(f.shape, 'float32', order='F')
    else:
        u_vis = None

    while k < maxiter:
        tau_p = 1.0 / (k + 1)
        tau_d = 1.0 / tau_p / L2

        # dual update
        tgv_update_p(u_, w_, p, tau_d, alpha1)
        tgv_update_q(w_, q, tau_d, alpha0)

        # primal update
        cuda.memcpy_dtod(u_.gpudata, u.gpudata, u.nbytes)
        cuda.memcpy_dtod(w_.gpudata, w.gpudata, w.nbytes)

        tgv_update_u(u, p, f_gpu, tau_p, norm)
        tgv_update_w(w, p, q, tau_p)

        # extragradient update
        tgv_update_u_(u_, u)
        tgv_update_w_(w_, w)

        if (vis > 0) and (k % vis == 0):
            display(".")
            cuda.memcpy_dtoh(u_vis.data, u.gpudata)
            visualize(u_vis)

        # residual (if you already compute it elsewhere, keep yours)
        # here we just store something stable if needed
        residuals.append(0.0)

        # energy terms on CPU
        u_cpu = u.get()
        w_cpu = w.get()

        data_term, tgv1_term, tgv2_term, total = tgv2_energy_terms_np(
            u_cpu, w_cpu, f_cpu, alpha1=alpha1, alpha0=alpha0
        )

        F_data.append(data_term)
        F_tgv1.append(tgv1_term)
        F_tgv2.append(tgv2_term)
        F_vals.append(total)

        k += 1

    if vis > 0:
        display("\n")

    if return_terms:
        return u.get().squeeze(), F_vals, {
            "data": F_data,
            "tgv1": F_tgv1,
            "tgv2": F_tgv2,
            "total": F_vals,
        }

    return u.get().squeeze(), F_vals