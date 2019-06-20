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

from ccpi.framework import AcquisitionGeometry, BlockDataContainer,AcquisitionData, ImageData, ImageGeometry
from ccpi.optimisation.algorithms import FISTA
from ccpi.optimisation.functions import IndicatorBox, ZeroFunction, \
                         L2NormSquared, FunctionOperatorComposition
from ccpi.optimisation.operators import Gradient, BlockOperator                      
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
phantom_2D = TomoP2D.Model(model, N, path_library2D)
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
data = ImageData(phantom_2D)

device = input('Available device: GPU==1 / CPU==0 ')
geometry = input('Parallel/Cone : Cone == 1 / Parallel == 0 ')

angles_num = 90

if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'


if geometry=='0':
    
    detectors = int(np.sqrt(2)*N)  
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
eta = 0
noisy_data = AcquisitionData(sin.as_array() + np.random.normal(0,1,ag.shape))
back_proj = Aop.adjoint(noisy_data)

# Define Least Squares
f = FunctionOperatorComposition(L2NormSquared(b=noisy_data), Aop)

# Allocate solution
x_init = ig.allocate()

# Run FISTA for least squares
fista = FISTA(x_init = x_init, f = f, g = ZeroFunction())
fista.max_iteration = 10
fista.update_objective_interval = 2
fista.run(100, verbose = True)

# Run FISTA for least squares with lower/upper bound 
fista0 = FISTA(x_init = x_init, f = f, g = IndicatorBox(lower=0,upper=1))
fista0.max_iteration = 10
fista0.update_objective_interval = 2
fista0.run(100, verbose=True)

# Run FISTA for Regularised least squares, with Squared norm of Gradient
alpha = 20
Grad = Gradient(ig)
block_op = BlockOperator( Aop, alpha * Grad, shape=(2,1))
block_data = BlockDataContainer(noisy_data, Grad.range_geometry().allocate())
f1 = FunctionOperatorComposition(L2NormSquared(b=block_data), block_op)

fista1 = FISTA(x_init = x_init, f = f1, g = IndicatorBox(lower=0,upper=1))
fista1.max_iteration = 2000
fista1.update_objective_interval = 200
fista1.run(2000, verbose=True)

#%%
# Show Ground Truth/Noisy Data & BackProjection

from mpl_toolkits.axes_grid1 import make_axes_locatable
def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    return fig.colorbar(mappable, cax=cax)

fig, ax = plt.subplots(1,3)
img1 = ax[0].imshow(data.as_array())
ax[0].set_title('Ground Truth')
colorbar(img1)
img2 = ax[1].imshow(noisy_data.as_array())
ax[1].set_title('Projection Data')
colorbar(img2)
img3 = ax[2].imshow(back_proj.as_array())
ax[2].set_title('BackProjection')
colorbar(img3)
plt.tight_layout(h_pad=1.5)

fig1, ax1 = plt.subplots(1,3)
img4 = ax1[0].imshow(fista.get_output().as_array())
ax1[0].set_title('LS unconstrained')
colorbar(img4)
img5 = ax1[1].imshow(fista0.get_output().as_array())
ax1[1].set_title('LS constrained [0,1]')
colorbar(img5)
img6 = ax1[2].imshow(fista1.get_output().as_array())
ax1[2].set_title('L2-Regularised LS')
colorbar(img6)
plt.tight_layout(h_pad=1.5)



#%% Check with CVX solution

import astra
import numpy
from ccpi.optimisation.operators import SparseFiniteDiff

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False
    
if cvx_not_installable:
    
    ##Construct problem    
    u = Variable(N*N)
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')

    # create matrix representation for Astra operator
    vol_geom = astra.create_vol_geom(N, N)
    proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)

    proj_id = astra.create_projector('line', proj_geom, vol_geom)

    matrix_id = astra.projector.matrix(proj_id)
    
    ProjMat = astra.matrix.get(matrix_id)
    
    tmp = noisy_data.as_array().ravel()
    
    fidelity = sum_squares(ProjMat * u - tmp) 
    
    regulariser = alpha**2 * sum_squares(norm(vstack([DX.matrix() * vec(u), DY.matrix() * vec(u)]), 2, axis = 0))

    solver = MOSEK
    obj =  Minimize(fidelity + regulariser)
    constraints = [u>=0, u<=1]
    prob = Problem(obj, constraints)
    result = prob.solve(verbose = True, solver = solver)   
      
    diff_cvx = numpy.abs( fista1.get_output().as_array() - np.reshape(u.value, ig.shape ))
           
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(fista1.get_output().as_array())
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
    
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), fista1.get_output().as_array()[int(N/2),:], label = 'FISTA')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), np.reshape(u.value, ig.shape )[int(N/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
    
    
    print('Primal Objective (FISTA) {} '.format(fista1.objective[-1]))
    print('Primal Objective (CVX) {} '.format(obj.value))  