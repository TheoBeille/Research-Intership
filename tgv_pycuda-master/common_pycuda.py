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

import sys
import pycuda.driver as cuda
from pycuda import gpuarray
from numpy import array, ceil, log2, zeros, clip
from matplotlib.pyplot import ion, ioff, imshow, cm, draw, clf

block_size_x = 16
block_size_y = 16
block_ = (block_size_x, block_size_y, 1)


def get_grid(u, offset=0):
    grid = (((u.shape[0 + offset] + block_[0] - 1) // block_[0]),
            ((u.shape[1 + offset] + block_[1] - 1) // block_[1]))
    return grid


def gpuarray_copy(u):
    v = gpuarray.zeros_like(u)
    v.strides = u.strides
    cuda.memcpy_dtod(v.gpudata, u.gpudata, u.nbytes)
    return v


def prepare_data(f):
    if len(f.shape) < 3:
        f = f.reshape(f.shape[0], f.shape[1], 1)
    f = f.astype('float32').copy(order='F')
    return f


def enlarge_next_power_of_2(f):
    # determine larger shape
    new_shape = array(f.shape)
    new_shape[0:2] = pow(2.0, ceil(log2(new_shape[0:2]))).astype('int32')
    if (new_shape == f.shape).all():
        return f

    # zero pad
    offset = (new_shape - array(f.shape)) / 2
    g = zeros(list(new_shape), 'float32', order='F')
    g[offset[0]:offset[0] + f.shape[0], offset[1]:offset[1] + f.shape[1]] = f
    return g


def display(str_):
    sys.stdout.write(str_)
    sys.stdout.flush()


def visualize(u):
    ion()
    clf()
    if u.shape[2] > 1:
        imshow(clip(u, 0, 1))
    else:
        imshow(u.squeeze(), cmap=cm.gray)
    draw()
    ioff()
