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
Tikhonov reconstruction using Block CGLS 


Problem:     min_u alpha * ||\grad u ||^{2}_{2} + || A u - g ||_{2}^{2}
            
             ==> Block CGLS
             
             min_u || \tilde{A} u - \tilde{g} ||_{2}^{2}
             
             \tilde{A} = [ A,
                           alpha * \grad]
             \tilde{g} = [g,
                          0]

             A: Projection operator
             g: Sinogram

"""


from ccpi.framework import ImageGeometry, ImageData, AcquisitionGeometry, BlockDataContainer, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import CGLS
from ccpi.optimisation.operators import BlockOperator, Gradient, Identity
       
import tomophantom
from tomophantom import TomoP2D
from ccpi.astra.operators import AstraProjectorSimple 
import os

device = input('Available device: GPU==1 / CPU==0 ')

if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'

# Load  Shepp-Logan phantom 
model = 1 # select a model number from the library
N = 64 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2D = TomoP2D.Model(model, N, path_library2D)

ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
data = ImageData(phantom_2D)


detectors = N
angles = np.linspace(0, np.pi, 128, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)

Aop = AstraProjectorSimple(ig, ag, dev)    
sin = Aop.direct(data)
#noisy_data = AcquisitionData( sin.as_array() + np.random.normal(0,1,ag.shape))
noisy_data = AcquisitionData( sin.as_array() )

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

# Setup and run the CGLS algorithm  
alpha = 2
Grad = Gradient(ig)
Id = Identity(ig)

# Form Tikhonov as a Block CGLS structure
op_CGLS = BlockOperator( Aop, alpha * Grad, shape=(2,1))
block_data = BlockDataContainer(noisy_data, Grad.range_geometry().allocate())

x_init = ig.allocate()      
cgls = CGLS(x_init=x_init, operator=op_CGLS, data=block_data)
cgls.max_iteration = 1000
cgls.update_objective_interval = 50
cgls.run(1000, verbose = True)

# Show results
plt.figure(figsize=(5,5))
plt.imshow(cgls.get_output().as_array())
plt.title('CGLS reconstruction')
plt.colorbar()
plt.show()


#%% Check with CVX solution

#from ccpi.optimisation.operators import SparseFiniteDiff
#import astra
#import numpy
#
#try:
#    from cvxpy import *
#    cvx_not_installable = True
#except ImportError:
#    cvx_not_installable = False
#    
#if cvx_not_installable:
#    
#    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
#    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
#    
#    ##Construct problem    
#    u = Variable(N*N)
#
#    # create matrix representation for Astra operator
#    vol_geom = astra.create_vol_geom(N, N)
#    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)
#
#    proj_id = astra.create_projector('line', proj_geom, vol_geom)
#
#    matrix_id = astra.projector.matrix(proj_id)
#
#    ProjMat = astra.matrix.get(matrix_id)
#    
#    tmp = noisy_data.as_array().ravel()
#    
#    fidelity = sum_squares(ProjMat * u - tmp)
#    regulariser = (alpha**2) * sum_squares(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))
#
#    solver = MOSEK
#    obj =  Minimize(fidelity + regulariser)
#    prob = Problem(obj)
#    result = prob.solve(verbose = True, solver = solver)    
#         
#    diff_cvx = numpy.abs( cgls.get_output().as_array() - np.reshape(u.value, (N,N) ))
#           
#    plt.figure(figsize=(15,15))
#    plt.subplot(3,1,1)
#    plt.imshow(cgls.get_output().as_array())
#    plt.title('CGLS solution')
#    plt.colorbar()
#    plt.subplot(3,1,2)
#    plt.imshow(np.reshape(u.value, (N, N)))
#    plt.title('CVX solution')
#    plt.colorbar()
#    plt.subplot(3,1,3)
#    plt.imshow(diff_cvx)
#    plt.title('Diff')
#    plt.colorbar()    
#    plt.show()    
#    
#    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), cgls.get_output().as_array()[int(N/2),:], label = 'CGLS')
#    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), np.reshape(u.value, (N,N) )[int(N/2),:], label = 'CVX')
#    plt.legend()
#    plt.title('Middle Line Profiles')
#    plt.show()
#            
#    print('Primal Objective (CVX) {} '.format(obj.value))
#    print('Primal Objective (CGLS) {} '.format(cgls.objective[-1]))



            
