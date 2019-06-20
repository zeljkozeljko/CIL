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
Example for Least Squares minimisation with CG algorithm  (CGLS)


Problem:     min_u || A u - g ||_{2}^{2}


             A: Projection operator
             g: Sinogram

"""


from ccpi.framework import ImageGeometry, ImageData, \
                    AcquisitionGeometry, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt
from ccpi.optimisation.algorithms import CGLS     
import os
from ccpi.astra.operators import AstraProjectorSimple 
import tomophantom
from tomophantom import TomoP2D

# Load Shepp-Logan phantom 
model = 1 # select a model number from the library
N = 64 
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

#This will generate a N_size x N_size phantom (2D)
phantom_2D = TomoP2D.Model(model, N, path_library2D)

# Create image geometry
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
data = ImageData(phantom_2D)

# Create Acquisition data
detectors = N
angles = np.linspace(0, np.pi, 180, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)

device = input('Available device: GPU==1 / CPU==0 ')
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'

Aop = AstraProjectorSimple(ig, ag, dev)    
sin = Aop.direct(data)
projection_data = AcquisitionData(sin.as_array())

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,1,2)
plt.imshow(projection_data.as_array())
plt.title('Projection Data')
plt.colorbar()
plt.show()

# Setup and run the CGLS algorithm  
x_init = ig.allocate()      
cgls = CGLS(x_init=x_init, operator=Aop, data=projection_data)
cgls.max_iteration = 5000
cgls.update_objective_interval = 100
cgls.run(1000,verbose=True)

plt.figure(figsize=(5,5))
plt.imshow(cgls.get_output().as_array())
plt.title('CGLS reconstruction')
plt.colorbar()
plt.show()

# Check with CVX solution

print('If N > {} : CVX solution will take some time '.format(128))

import astra
import numpy

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False
    
if cvx_not_installable:
    
    ##Construct problem    
    u = Variable(N*N)

    # create matrix representation for Astra operator
    vol_geom = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)

    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)

    ProjMat = astra.matrix.get(matrix_id)
    
    tmp = projection_data.as_array().ravel()
    
    fidelity =  sum_squares(ProjMat * u - tmp)

    solver = MOSEK
    obj =  Minimize(fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = solver)    
         
    diff_cvx = numpy.abs( cgls.get_output().as_array() - np.reshape(u.value, (N,N) ))
           
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(cgls.get_output().as_array())
    plt.title('CGLS solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(np.reshape(u.value, (N, N)))
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Diff')
    plt.colorbar()    
    plt.show()    
    
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), cgls.get_output().as_array()[int(N/2),:], label = 'CGLS')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), np.reshape(u.value, (N,N) )[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (CGLS) {} '.format(cgls.objective[-1]))