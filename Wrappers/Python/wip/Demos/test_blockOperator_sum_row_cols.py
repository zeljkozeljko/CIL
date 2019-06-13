#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 13:10:09 2019

@author: evangelos
"""

from ccpi.optimisation.operators import FiniteDiff, SparseFiniteDiff, BlockOperator, Gradient, Identity
from ccpi.framework import ImageGeometry, AcquisitionGeometry, BlockDataContainer, ImageData
from ccpi.astra.ops import AstraProjectorSimple

from scipy import sparse
import numpy as np

N = 3
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
u = ig.allocate('random_int')

# Compare FiniteDiff with SparseFiniteDiff

DY = FiniteDiff(ig, direction = 0, bnd_cond = 'Neumann')
DX = FiniteDiff(ig, direction = 1, bnd_cond = 'Neumann')

DXu = DX.direct(u)
DYu = DY.direct(u)

DX_sparse = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
DY_sparse = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')

DXu_sparse = DX_sparse.direct(u)
DYu_sparse = DY_sparse.direct(u)

np.testing.assert_array_almost_equal(DYu.as_array(), DYu_sparse.as_array(), decimal=4)
np.testing.assert_array_almost_equal(DXu.as_array(), DXu_sparse.as_array(), decimal=4)

#%%  Tau/ Sigma

Id = Identity(ig)
Grad = Gradient(ig)

A1 = DY_sparse.matrix()
A2 = DX_sparse.matrix()
A3 = Id.matrix()

Block1 = sparse.vstack([A1,A2,A3]).toarray()

# comppute tau using vstack
tau_Block1 = np.reshape(np.abs(Block1).sum(axis=0), ig.shape, 'F')
print(" Dimension of tau_Block1 is {}".format(tau_Block1.shape))

# compute tau using sum_abs_row for each Operator
tau_Block2 = DY_sparse.sum_abs_row().as_array() + DX_sparse.sum_abs_row().as_array() + Id.sum_abs_row().as_array()
np.testing.assert_array_almost_equal(tau_Block1, tau_Block2)

# compute tau using BlockOperator
B = BlockOperator(Grad, Id)
tau_Block3 = B.sum_abs_row()

np.testing.assert_array_almost_equal(tau_Block2, tau_Block3.as_array())
np.testing.assert_array_almost_equal(tau_Block1, tau_Block3.as_array())

# comppute tau using vstack
sigma_Block1a = np.reshape(np.abs(Block1[0:9, 0:18]).sum(axis=1), ig.shape, 'F')
sigma_Block1b = np.reshape(np.abs(Block1[9:18, 0:18]).sum(axis=1), ig.shape, 'F')
sigma_Block1c = np.reshape(np.abs(Block1[18:27, 0:18]).sum(axis=1), ig.shape, 'F')
print(" Dimension of sigma_Block1a is {}".format(sigma_Block1a.shape))
print(" Dimension of sigma_Block1b is {}".format(sigma_Block1b.shape))
print(" Dimension of sigma_Block1c is {}".format(sigma_Block1c.shape))

sigma_Block3 = B.sum_abs_col()

np.testing.assert_array_almost_equal(sigma_Block1a, sigma_Block3.get_item(0).get_item(0).as_array())
np.testing.assert_array_almost_equal(sigma_Block1b, sigma_Block3.get_item(0).get_item(1).as_array())
np.testing.assert_array_almost_equal(sigma_Block1c, sigma_Block3.get_item(1).as_array())

#%%

N = 3
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
u = ig.allocate('random_int')

DX_sparse = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
DY_sparse = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
A1 = DY_sparse.matrix()
A2 = DX_sparse.matrix()

Block1 = np.abs(sparse.vstack([A1,A2]).toarray())

tmp_tau = np.reshape(np.sum(Block1, axis = 0), (N, N), order = 'C')
tmp_sigma = np.reshape(np.sum(Block1, axis = 1), (2, N, N), order = 'C')

tau = 1/tmp_tau
sigma = 1/tmp_sigma
sigma[0][sigma[0]==np.inf] = 0
sigma[1][sigma[1]==np.inf] = 0


G = Gradient(ig)

tmp_tau1 = G.sum_abs_col()
tmp_sigma1 = G.sum_abs_row()

tau1 = 1/tmp_tau1
sigma1 = 1/tmp_sigma1










