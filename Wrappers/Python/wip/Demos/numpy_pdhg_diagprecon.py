# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


""" 
Total Variation Denoising using PDHG algorithm

Problem:     min_x \alpha * ||\nabla x||_{1} + || x - g ||_{2}^{2}

             \nabla: Gradient operator 
             g: Noisy Data with Gaussian Noise
             \alpha: Regularization parameter

"""

from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, PDHG_old

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient, SparseFiniteDiff
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

import scipy.sparse as sp

# Create phantom for TV Gaussian denoising
N = 100

data = np.zeros((N,N))
data[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
data[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 1
data = ImageData(data)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
ag = ig

# Create noisy data. Add Gaussian noise
np.random.seed(10)
noisy_data = data.as_array() + np.random.normal(0, 0.05, size=ig.shape) 

# Regularisation Parameter
alpha = 2

DX_sparse = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
DY_sparse = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')

DX = DX_sparse.matrix()
DY = DX_sparse.matrix()

Block = sp.vstack([DY,DX])
BlockT = Block.transpose()

x_old = np.zeros((np.prod(ig.shape),1))
y_old = np.zeros((2 * np.prod(ig.shape),1))
y = np.zeros((2 * np.prod(ig.shape),1))

xbar = np.zeros((np.prod(ig.shape),1))

niter = 10000

b = np.reshape(noisy_data, (N*N, 1), order = 'F')

sigma = 0.1
tau = 1/(8*sigma)
#%%

for i in range(niter):
            

    y_tmp = y_old +  sigma * Block * xbar
    
    t1 = np.sqrt(y_tmp[0:N*N]**2 + y_tmp[N*N:2*N*N]**2)
    
    y[0:N*N] = y[0:N*N]/np.maximum(1, t1/alpha)
    y[N*N:2*N*N] = y[N*N:2*N*N]/np.maximum(1, t1/alpha)

    x_tmp = x_old - tau * BlockT * y
    
    x = b + (x_tmp - b)/(1+2*tau)
    
    x_bar  = 2 * x - x_old
                                                  
    x_old = x
    y_old = y
    
    
    
sol = np.reshape(x, (N, N), order = 'F')  
plt.imshow(sol)  
    
#    print(i)
    
    
#    p1 = f(operator.direct(x)) + g(x)
#    d1 = - ( f.convex_conjugate(y) + g.convex_conjugate(-1*operator.adjoint(y)) )
#    
#    primal.append(p1)
#    dual.append(d1)
#    pdgap.append(p1-d1)


##%%
#method = '1'
#
#if method == '0':
#
#    # Create operators
#    op1 = Gradient(ig)
#    op2 = Identity(ig, ag)
#
#    # Create BlockOperator
#    operator = BlockOperator(op1, op2, shape=(2,1) ) 
#
#    # Create functions
#      
#    f1 = alpha * MixedL21Norm()
#    f2 = 0.5 * L2NormSquared(b = noisy_data)    
#    f = BlockFunction(f1, f2)  
#                                      
#    g = ZeroFunction()
#    
#else:
#    
#    # Without the "Block Framework"
#    operator =  Gradient(ig)
#    f =  alpha * MixedL21Norm()
#    g =  0.5 * L2NormSquared(b = noisy_data)
#        
#
##%%
################################################################################
## No preconditioning  
#normK = operator.norm()    
#sigma = 1
#tau = 1/(sigma*normK**2)
#
#opt = {'niter':2000, 'memopt': True}
#
#res1, sol_it1, primal1, dual1, pdgap1 = PDHG_old(f, g, operator, tau = tau, sigma = sigma, opt = opt) 
# 
#plt.figure(figsize=(5,5))
#plt.imshow(res1.as_array())
#plt.colorbar()
#plt.show()    
#
##%%
################################################################################
## Diag preconditioning  
#
##from scipy import sparse
##DX_sparse = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
##DY_sparse = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
##Id = Identity(ig)
##
##A1 = DY_sparse.matrix()
##A2 = DY_sparse.matrix()
##B1 = Id.matrix()
##
##Block1a = sparse.vstack([A1,A2]).toarray()
##Block = sparse.vstack([Block1a,B1]).toarray()
#
#
##%%
#
#tau1 = 1.0/operator.sum_abs_row()
#sigma1 = 1.0/operator.sum_abs_col()
#
#opt = {'niter':2000, 'memopt': True}
#
#res2, sol_it2, primal2, dual2, pdgap2 = PDHG_old(f, g, operator, tau = tau1, sigma = sigma1, opt = opt) 
# 
#plt.figure(figsize=(5,5))
#plt.imshow(res2.as_array())
#plt.colorbar()
#plt.show()  
#    
##%%
#
#from cvxpy import *
#from ccpi.optimisation.operators import SparseFiniteDiff
#
#u = Variable(ig.shape)
#    
#DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
#DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
#
## Define Total Variation as a regulariser
#regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
#fidelity = 0.5 * sum_squares(u - noisy_data.as_array())
#
## choose solver
#if 'MOSEK' in installed_solvers():
#    solver = MOSEK
#else:
#    solver = SCS  
#    
#obj =  Minimize( regulariser +  fidelity)
#prob = Problem(obj)
#result = prob.solve(verbose = True, solver = MOSEK)
#
#
##%%
#
#def pobj(x, f, g, operator):
#    
#    return f(operator.direct(x)) + g(x)
#
#relat_obj1 = []
#relat_obj2 = []
#for i in range(opt['niter']):        
#    relat_obj1.append((primal1[i] - obj.value)/obj.value)
#    relat_obj2.append((primal2[i] - obj.value)/obj.value)
#    
#
##%%   
#
#plt.loglog(range(opt['niter']), pdgap1, \
#           range(opt['niter']), pdgap2) 
# 
#
#x = np.linspace(1, opt['niter']+1, opt['niter'])
#plt.loglog(range(opt['niter']), 1/(x**2))    
#    
##plt.loglog(range(opt['niter']), relat_obj1, \
##           range(opt['niter']), relat_obj2) 
## 
##
##x = np.linspace(1, opt['niter']+1, opt['niter'])
##plt.loglog(range(opt['niter']), 1/(x**2))
#
#
#
#
#
##%%
#plt.figure(figsize=(15,15))
#plt.subplot(3,1,1)
#plt.imshow(data.as_array())
#plt.title('Ground Truth')
#plt.colorbar()
#plt.subplot(3,1,2)
#plt.imshow(noisy_data.as_array())
#plt.title('Noisy Data')
#plt.colorbar()
#plt.subplot(3,1,3)
#plt.imshow(pdhg.get_output().as_array())
#plt.title('TV Reconstruction')
#plt.colorbar()
#plt.show()
### 
#plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'GTruth')
#plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'TV reconstruction')
#plt.legend()
#plt.title('Middle Line Profiles')
#plt.show()
#
#
##%% Check with CVX solution
#
#from ccpi.optimisation.operators import SparseFiniteDiff
#
#try:
#    from cvxpy import *
#    cvx_not_installable = True
#except ImportError:
#    cvx_not_installable = False
#
#
#if cvx_not_installable:
#
#    ##Construct problem    
#    u = Variable(ig.shape)
#    
#    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
#    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
#    
#    # Define Total Variation as a regulariser
#    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
#    fidelity = 0.5 * sum_squares(u - noisy_data.as_array())
#    
#    # choose solver
#    if 'MOSEK' in installed_solvers():
#        solver = MOSEK
#    else:
#        solver = SCS  
#        
#    obj =  Minimize( regulariser +  fidelity)
#    prob = Problem(obj)
#    result = prob.solve(verbose = True, solver = MOSEK)
#    
#    diff_cvx = numpy.abs( pdhg.get_output().as_array() - u.value )
#        
#    plt.figure(figsize=(15,15))
#    plt.subplot(3,1,1)
#    plt.imshow(pdhg.get_output().as_array())
#    plt.title('PDHG solution')
#    plt.colorbar()
#    plt.subplot(3,1,2)
#    plt.imshow(u.value)
#    plt.title('CVX solution')
#    plt.colorbar()
#    plt.subplot(3,1,3)
#    plt.imshow(diff_cvx)
#    plt.title('Difference')
#    plt.colorbar()
#    plt.show()    
#    
#    plt.plot(np.linspace(0,N,N), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
#    plt.plot(np.linspace(0,N,N), u.value[int(N/2),:], label = 'CVX')
#    plt.plot(np.linspace(0,N,N), data.as_array()[int(N/2),:], label = 'Truth')
#    
#    plt.legend()
#    plt.title('Middle Line Profiles')
#    plt.show()
#            
#    print('Primal Objective (CVX) {} '.format(obj.value))
#    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))
#
#
#
#
#
