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
Compare solutions of FISTA & PDHG 
                   & CGLS  & Astra Built-in algorithms for Least Squares
                   & CVX


Problem:     min_u || A u - g ||_{2}^{2}

             A: Projection operator
             g: Sinogram

"""


from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG, CGLS, FISTA

from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, L1Norm, FunctionOperatorComposition
from ccpi.astra.ops import AstraProjectorSimple
import astra   
import os
import tomophantom
from tomophantom import TomoP2D


# Load Data 
#loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))                       
#N = 50
#M = 50
#data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,M), scale=(0,1))
#ig = data.geometry

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

detectors = N
angles = np.linspace(0, np.pi, N, dtype=np.float32)

device = input('Available device: GPU==1 / CPU==0 ')
ag = AcquisitionGeometry('parallel','2D', angles, detectors)
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
    
Aop = AstraProjectorSimple(ig, ag, dev)
sin = Aop.direct(data)

projection_data = sin 

###############################################################################
# Setup and run Astra CGLS algorithm
vol_geom = astra.create_vol_geom(N, N)
proj_geom = astra.create_proj_geom('parallel', 1.0, detectors, angles)
proj_id = astra.create_projector('line', proj_geom, vol_geom)

# Create a sinogram from a phantom
sinogram_id, sinogram = astra.create_sino(data.as_array(), proj_id)

# Create a data object for the reconstruction
rec_id = astra.data2d.create('-vol', vol_geom)

cgls_astra = astra.astra_dict('CGLS')
cgls_astra['ReconstructionDataId'] = rec_id
cgls_astra['ProjectionDataId'] = sinogram_id
cgls_astra['ProjectorId'] = proj_id

# Create the algorithm object from the configuration structure
alg_id = astra.algorithm.create(cgls_astra)

astra.algorithm.run(alg_id, 1000)

recon_cgls_astra = astra.data2d.get(rec_id)

###############################################################################
# Setup and run the CGLS algorithm  
x_init = ig.allocate()             
cgls = CGLS(x_init=x_init, operator=Aop, data=projection_data)
cgls.max_iteration = 2000
cgls.update_objective_interval = 200
cgls.run(2000, verbose=True)

###############################################################################
# Setup and run the PDHG algorithm 
operator = Aop
f = L2NormSquared(b = projection_data)
g = 0.0001 * L1Norm()

## Compute operator Norm
normK = operator.norm()

## Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)


pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 2000
pdhg.update_objective_interval = 200
pdhg.run(2000, verbose=True)

###############################################################################
# Setup and run the FISTA algorithm 
fidelity = FunctionOperatorComposition(L2NormSquared(b=projection_data), Aop)
regularizer = ZeroFunction()

fista = FISTA(x_init=x_init , f=fidelity, g=regularizer)
fista.max_iteration = 2000
fista.update_objective_interval = 200
fista.run(2000, verbose=True)


#%% Show results

plt.figure(figsize=(10,10))
plt.suptitle('Reconstructions ', fontsize=16)

plt.subplot(2,2,1)
plt.imshow(cgls.get_output().as_array())
plt.colorbar()
plt.title('CGLS reconstruction')

plt.subplot(2,2,2)
plt.imshow(fista.get_output().as_array())
plt.colorbar()
plt.title('FISTA reconstruction')

plt.subplot(2,2,3)
plt.imshow(pdhg.get_output().as_array())
plt.colorbar()
plt.title('PDHG reconstruction')

plt.subplot(2,2,4)
plt.imshow(recon_cgls_astra)
plt.colorbar()
plt.title('CGLS astra')

#% Middle line Profiles
plt.figure(figsize=(10,10))

plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), cgls.get_output().as_array()[int(N/2),:], label = 'CGLS')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), recon_cgls_astra[int(N/2),:], label = 'CGLS_Astra')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(N/2),:], label = 'PDHG')
plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), fista.get_output().as_array()[int(N/2),:], label = 'FISTA')
plt.legend()
plt.title('Middle Line Profiles')
plt.show()

print('Primal Objective (FISTA) {} '.format(fista.objective[-1]))
print('Primal Objective (CGLS) {} '.format(cgls.objective[-1]))
print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))

