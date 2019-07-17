#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

""" 

Total Variation Denoising using PDHG algorithm:


Problem:     min_{u},   \alpha * ||\nabla u||_{2,1} + Fidelity(u, g)

             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
             
             g: Noisy Data 
                          
             Fidelity =  1) L2NormSquarred ( \frac{1}{2} * || u - g ||_{2}^{2} ) if Noise is Gaussian
                         2) L1Norm ( ||u - g||_{1} )if Noise is Salt & Pepper
                         3) Kullback Leibler (\int u - g * log(u) + Id_{u>0})  if Noise is Poisson
                                                       
             Method = 0 ( PDHG - split ) :  K = [ \nabla,
                                                 Identity]
                          
                                                                    
             Method = 1 (PDHG - explicit ):  K = \nabla    
             
             
             Default: ROF denoising
             noise = Gaussian
             Fidelity = L2NormSquarred 
             method = 0
             
             
"""

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction, L2NormSquared,\
                          KullbackLeibler
from ccpi.framework import TestData
import os
import sys
import time


# create a Gradient Operator with Sobel filter
import cv2
from ccpi.optimisation.operators import LinearOperator, ScaledOperator
from ccpi.framework import BlockGeometry, BlockDataContainer
#cv_gx = cv2.Sobel(data.as_array(),cv2.CV_64F,0,1,ksize=3)
#cv_gy = cv2.Sobel(data.as_array(),cv2.CV_64F,1,0,ksize=3)

class GradientSobel(LinearOperator):
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(LinearOperator, self).__init__() 
                
        self.gm_domain = gm_domain # Domain of Grad Operator
        self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(self.gm_domain.length) ] )
                
        self.bnd_cond = bnd_cond

    def sobelX(self, x, forward=True):
        '''return the gradient from sobel filter on X as numpy array'''
        if forward:
            return cv2.Sobel(x.as_array(),cv2.CV_64F,1, 0, ksize=3)
        else:
            kernel = numpy.asarray([[1,0,-1],[2,0,-2],[1,0,-1]])
            return -1 * cv2.filter2D(x.as_array(), -1, kernel)#, borderType=cv2.BORDER_REPLICATE)
    def sobelY(self, x, forward=True):
        '''return the gradient from sobel filter on Y as numpy array'''
        if forward:
            return cv2.Sobel(x.as_array(),cv2.CV_64F,0, 1,ksize=3)
        else:
            kernel = numpy.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
            return -1 * cv2.filter2D(x.as_array(), -1, kernel)#, borderType=cv2.BORDER_REPLICATE)
    def direct(self, x, out=None):
        
                
        if out is not None:
            
            for i in range(self.gm_range.shape[0]):
                if i == 0:
                    g = self.sobelX(x)
                elif i == 1:
                    g = self.sobelY(x)
                else:
                    raise ValueError('expecting direction in 0,1, got', i)
                out[i].fill(g)

        else:
            tmp = self.gm_range.allocate()        
            for i in range(tmp.shape[0]):
                if i == 0:
                    g = self.sobelX(x)
                elif i == 1:
                    g = self.sobelY(x)
                else:
                    raise ValueError('expecting direction in 0,1, got', i)
                tmp.get_item(i).fill(g)
            return tmp    
        
    def adjoint(self, x, out=None):
        # if out is not None:

        #     tmp = self.gm_domain.allocate()            
        #     for i in range(x.shape[0]):
        #         self.FD.direction=self.ind[i] 
        #         self.FD.adjoint(x.get_item(i), out = tmp)
        #         if i == 0:
        #             out.fill(tmp)
        #         else:
        #             out += tmp
        # else:            
        #     tmp = self.gm_domain.allocate()
        #     for i in range(x.shape[0]):
        #         self.FD.direction=self.ind[i]

        #         tmp += self.FD.adjoint(x.get_item(i))
        #     return tmp    
        if out is not None:

            tmp = self.gm_domain.allocate()            
            for i in range(x.shape[0]):
                if i == 0:
                    g = self.sobelX(x.get_item(i), forward=False)
                    #self.FD.adjoint(x.get_item(i), out = tmp)
                    tmp.fill(g)
                    out.fill(tmp)
                elif i == 1:
                    g = self.sobelY(x.get_item(i), forward=False)
                    tmp.fill(g)
                    out += tmp
                else:
                    raise ValueError('expecting direction in 0,1, got', i)
        else:            
            tmp = self.gm_domain.allocate()
            for i in range(x.shape[0]):
                #self.FD.direction=self.ind[i]

                #tmp += self.FD.adjoint(x.get_item(i))
                if i == 0:
                    g = self.sobelX(x.get_item(i), forward=False)
                    #self.FD.adjoint(x.get_item(i), out = tmp)
                    tmp.fill(g)
                    
                elif i == 1:
                    g = self.sobelY(x.get_item(i), forward=False)
                    y = tmp.copy()
                    y.fill(g)
                    tmp += y
                else:
                    raise ValueError('expecting direction in 0,1, got', i)
            return tmp
            
    
    def domain_geometry(self):
        
        '''Returns domain_geometry of Gradient'''
        
        return self.gm_domain
    
    def range_geometry(self):
        
        '''Returns range_geometry of Gradient'''
        
        return self.gm_range
    
    def __rmul__(self, scalar):
        
        '''Multiplication of Gradient with a scalar        
            
            Returns: ScaledOperator
        '''        
        
        return ScaledOperator(self, scalar) 

# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0
print ("Applying {} noise")

if len(sys.argv) > 2:
    method = sys.argv[2]
else:
    method = '1'
print ("method ", method)


loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SHAPES)
ig = data.geometry
ag = ig

# Create noisy data. 
noises = ['gaussian', 'poisson', 's&p']
noise = noises[which_noise]
if noise == 's&p':
    n1 = TestData.random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2)
elif noise == 'poisson':
    scale = 5
    n1 = TestData.random_noise( data.as_array()/scale, mode = noise, seed = 10)*scale
elif noise == 'gaussian':
    n1 = TestData.random_noise(data.as_array(), mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
noisy_data = ig.allocate()
noisy_data.fill(n1)





if True:
    fdiff = Gradient(ig)
    sobel = GradientSobel(ig)

    g_fdiff = fdiff.direct(noisy_data)
    g_sobel = sobel.direct(noisy_data)
    print (g_sobel.shape)


    plt.subplot(2,2,1)
    plt.imshow(g_fdiff.get_item(0).as_array())
    plt.title('finite diff x')
    plt.subplot(2,2,2)
    plt.imshow(g_fdiff.get_item(1).as_array())
    plt.title('finite diff y')
    
    plt.subplot(2,2,3)
    plt.imshow(g_sobel.get_item(0).as_array())
    plt.title('sobel x')
    plt.subplot(2,2,4)
    plt.imshow(g_sobel.get_item(1).as_array())
    plt.title('sobel y')
    plt.show()
    
    adj_fd = fdiff.adjoint(g_fdiff)
    adj_so = sobel.adjoint(g_sobel)

    plt.subplot(1,2,1)
    plt.imshow(adj_fd.as_array())
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(adj_so.as_array())
    #plt.clim([-1e-7, 1e-7])
    plt.colorbar()
    plt.show()
else:
    # Show Ground Truth and Noisy Data
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(data.as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(noisy_data.as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.show()


# Regularisation Parameter depending on the noise distribution
if noise == 's&p':
    alpha = 0.8
elif noise == 'poisson':
    alpha = 1
elif noise == 'gaussian':
    alpha = .3

# fidelity
if noise == 's&p':
    f2 = L1Norm(b=noisy_data)
elif noise == 'poisson':
    f2 = KullbackLeibler(noisy_data)
elif noise == 'gaussian':
    f2 = 0.5 * L2NormSquared(b=noisy_data)

if method == '0':

    # Create operators
    op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
    op2 = Identity(ig, ag)

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions      
    f = BlockFunction(alpha * MixedL21Norm(), f2) 
    g = ZeroFunction()
    
else:
    
    #operator = Gradient(ig)
    operator = GradientSobel(ig)
    f =  (1e-1 * alpha) * MixedL21Norm()
    g = f2
        



# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 10
pdhg.run(20)

# Show results
plt.figure(figsize=(20,5))
plt.subplot(1,4,1)
plt.imshow(data.subset(channel=0).as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.clim([0,1])
plt.subplot(1,4,2)
plt.imshow(noisy_data.subset(channel=0).as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.clim([0,1])
plt.subplot(1,4,3)
plt.imshow(pdhg.get_output().subset(channel=0).as_array())
plt.title('TV Reconstruction')
plt.colorbar()
plt.clim([0,1])
plt.subplot(1,4,4)
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()


#%% Check with CVX solution

from ccpi.optimisation.operators import SparseFiniteDiff

try:
    from cvxpy import *
    cvx = True
except ImportError:
    cvx = False

if not cvx:
    print("Install CVXPY module to compare with CVX solution")
else:

    ##Construct problem
    u = Variable(ig.shape)
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    
    # Define Total Variation as a regulariser
    regulariser = alpha * sum(norm(vstack([Constant(DX.matrix()) * vec(u), Constant(DY.matrix()) * vec(u)]), 2, axis = 0))
    
    # choose solver
    if 'MOSEK' in installed_solvers():
        solver = MOSEK
    else:
        solver = SCS      

    # fidelity
    if noise == 's&p':
        fidelity = pnorm( u - noisy_data.as_array(),1)
    elif noise == 'poisson':
        fidelity = sum(kl_div(noisy_data.as_array(), u)) 
        solver = SCS
    elif noise == 'gaussian':
        fidelity = 0.5 * sum_squares(noisy_data.as_array() - u)
                
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)

    result = prob.solve(verbose = True, solver = solver, max_iters=10)
    
    diff_cvx = numpy.abs( pdhg.get_output().as_array() - u.value )
        
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(pdhg.get_output().as_array())
    plt.title('PDHG solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(u.value)
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'PDHG')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), u.value[int(ig.shape[0]/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))