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

from ccpi.framework import AcquisitionGeometry, ImageData, ImageGeometry
from ccpi.optimisation.algorithms import FISTA
from ccpi.optimisation.functions import IndicatorBox, ZeroFunction, \
                         L2NormSquared, FunctionOperatorComposition
from ccpi.astra.operators import AstraProjectorSimple

import numpy as np
import matplotlib.pyplot as plt
import tomophantom
from tomophantom import TomoP2D
import os

# Load Shepp-Logan phantom 
model = 1 
N = 64 
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

#This will generate a N_size x N_size phantom (2D)
phantom_2D = TomoP2D.Model(model, N, path_library2D)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
data = ImageData(phantom_2D)


device = input('Available device: GPU==1 / CPU==0 ')
geometry = input('Parallel/Cone : Cone == 1 / Parallel == 0 ')

if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'


if geometry=='0':
    
    detectors = N
    angles_num = N    
    det_w = 1.0
    
    angles = np.linspace(0, np.pi, angles_num, endpoint=False)
    ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             detectors,det_w)
  
elif geometry=='1':
    
    SourceOrig = 200
    OrigDetec = 0
    angles = np.linspace(0,2*np.pi,angles_num)
    ag = AcquisitionGeometry('cone',
                             '2D',
                             angles,
                             detectors,
                             det_w,
                             dist_source_center=SourceOrig, 
                             dist_center_detector=OrigDetec)
else:
    NotImplemented

   
Aop = AstraProjectorSimple(ig, ag, dev)
sin = Aop.direct(data)
back_proj = Aop.adjoint(sin)

f = FunctionOperatorComposition(L2NormSquared(b=sin), Aop)
g = ZeroFunction()

x_init = ig.allocate()
fista = FISTA(x_init=x_init, f=f, g=g)
fista.max_iteration = 1000
fista.update_objective_interval = 100
fista.run(1000, verbose = True)

# Run FISTA for least squares with lower/upper bound 
fista0 = FISTA(x_init=x_init, f=f, g=IndicatorBox(lower=0,upper=1))
fista0.max_iteration = 1000
fista0.update_objective_interval = 100
fista0.run(1000, verbose=True)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.subplot(1,3,2)
plt.imshow(sin.as_array())
plt.title('Projection Data')
plt.subplot(1,3,3)
plt.imshow(back_proj.as_array())
plt.title('BackProjection')
plt.show()

plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(fista.get_output().as_array())
plt.title('FISTA unconstrained')
plt.subplot(1,2,2)
plt.imshow(fista0.get_output().as_array())
plt.title('FISTA constrained')
plt.show()




#%% Check with CVX solution

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
    
    tmp = sin.as_array().ravel()
    
    fidelity = sum_squares(ProjMat * u - tmp)

    solver = MOSEK
    obj =  Minimize(fidelity)
    constraints = [u>=0, u<=1]
    prob = Problem(obj, constraints)
    result = prob.solve(verbose = True, solver = solver)   
         
    diff_cvx = numpy.abs( fista0.get_output().as_array() - np.reshape(u.value, ig.shape ))
           
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(fista0.get_output().as_array())
    plt.title('FISTA solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(np.reshape(u.value, ig.shape))
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), fista0.get_output().as_array()[int(N/2),:], label = 'FISTA')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), np.reshape(u.value, ig.shape )[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
    
    
    print('Primal Objective (FISTA) {} '.format(fista.objective[-1]))
    print('Primal Objective (CVX) {} '.format(obj.value))  