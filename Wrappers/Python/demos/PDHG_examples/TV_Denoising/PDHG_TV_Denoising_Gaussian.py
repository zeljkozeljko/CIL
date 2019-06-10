#========================================================================
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#=========================================================================
""" 

Total Variation Denoising using PDHG algorithm:


Problem:     min_{x} \alpha * ||\nabla x||_{2,1} + \frac{1}{2} * || x - g ||_{2}^{2}

             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
             
             g: Noisy Data with Gaussian Noise
                          
             Method = 0 ( PDHG - split ) :  K = [ \nabla,
                                                 Identity]
                          
                                                                    
             Method = 1 (PDHG - explicit ):  K = \nabla  
                                                                
"""

from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction

from ccpi.framework import TestData
import os, sys
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))

# Load Data                      
N = 200
M = 300


# user can change the size of the input data
# you can choose between 
# TestData.PEPPERS 2D + Channel
# TestData.BOAT 2D 
# TestData.CAMERA 2D
# TestData.RESOLUTION_CHART 2D 
# TestData.SIMPLE_PHANTOM_2D 2D
data = loader.load(TestData.BOAT, size=(N,M), scale=(0,1))

ig = data.geometry
ag = ig
                      
# Create Noisy data. Add Gaussian noise
np.random.seed(10)
noisy_data = ImageData( data.as_array() + np.random.normal(0, 0.1, size=data.shape) )

print ("min {} max {}".format(data.as_array().min(), data.as_array().max()))

# Show Ground Truth and Noisy Data
plt.figure()
plt.subplot(1,3,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,3,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(1,3,3)
plt.imshow((data - noisy_data).as_array())
plt.title('diff')
plt.colorbar()

plt.show()

# Regularisation Parameter
alpha = .1

method = '0'

if method == '0':

    # Create operators
    op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
    op2 = Identity(ig, ag)

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions      
    f1 = alpha * MixedL21Norm()
    f2 = 0.5 * L2NormSquared(b = noisy_data)    
    f = BlockFunction(f1, f2)  
                                      
    g = ZeroFunction()
    
else:
    
    # Without the "Block Framework"
    operator =  Gradient(ig)
    f =  alpha * MixedL21Norm()
    g =  0.5 * L2NormSquared(b = noisy_data)
        
# Compute Operator Norm
normK = operator.norm()

# Primal & Dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

# Setup and Run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 10000
pdhg.update_objective_interval = 100
pdhg.run(1000, verbose=True)

# Show Results
plt.figure()
plt.subplot(1,3,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.clim(0,1)
plt.subplot(1,3,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.clim(0,1)
plt.subplot(1,3,3)
plt.imshow(pdhg.get_output().as_array())
plt.title('TV Reconstruction')
plt.clim(0,1)
plt.colorbar()
plt.show()

plt.plot(np.linspace(0,N,M), noisy_data.as_array()[int(N/2),:], label = 'Noisy data')
plt.plot(np.linspace(0,N,M), data.as_array()[int(N/2),:], label = 'GTruth')
plt.plot(np.linspace(0,N,M), pdhg.get_output().as_array()[int(N/2),:], label = 'TV reconstruction')

plt.legend()
plt.title('Middle Line Profiles')
plt.show()


#%% Check with CVX solution

from ccpi.optimisation.operators import SparseFiniteDiff

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False


if cvx_not_installable:

    ##Construct problem    
    u = Variable(ig.shape)
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    
    # Define Total Variation as a regulariser
    regulariser = alpha * sum(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
    fidelity = 0.5 * sum_squares(u - noisy_data.as_array())
    
    # choose solver
    if 'MOSEK' in installed_solvers():
        solver = MOSEK
    else:
        solver = SCS  
        
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = MOSEK)
    
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
    
    plt.plot(np.linspace(0,N,M), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
    plt.plot(np.linspace(0,N,M), u.value[int(N/2),:], label = 'CVX')
    plt.plot(np.linspace(0,N,M), data.as_array()[int(N/2),:], label = 'Truth')
    
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))
