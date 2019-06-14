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

Total Variation (Dynamic) Denoising using PDHG algorithm and Tomophantom:


Problem:     min_{x} \alpha * ||\nabla x||_{2,1} + \frac{1}{2} * || x - g ||_{2}^{2}

             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
             
             g: 2D Dynamic noisy data with Gaussian Noise
                          
             K = \nabla  
                                                                
"""

from ccpi.framework import ImageData, ImageGeometry

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient, Identity
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared,  \
                      MixedL21Norm, BlockFunction

import os
import tomophantom
from tomophantom import TomoP2D
import matplotlib.animation as animation

# Create phantom for TV 2D dynamic tomography 
model = 102  # note that the selected model is temporal (2D + time)
N = 64 # set dimension of the phantom

path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
phantom_2Dt = TomoP2D.ModelTemporal(model, N, path_library2D)

# Animate
fig = plt.figure()
ims1 = []
for sl in range(0,np.shape(phantom_2Dt)[0]):    
    im1 = plt.imshow(phantom_2Dt[sl,:,:], animated=True, vmin=0, vmax=1)    
    ims1.append([im1])
ani1 = animation.ArtistAnimation(fig, ims1, interval=500, blit = True,
                                repeat_delay=10)
plt.show() 

# Setup geometries    
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, channels = np.shape(phantom_2Dt)[0])
data = ImageData(phantom_2Dt, geometry=ig)
ag = ig
                      
# Create noisy data. Apply Gaussian noise
np.random.seed(10)
noisy_data = ImageData( data.as_array() + np.random.normal(0, 0.25, size=ig.shape) )

# time-frames index
tindex = [8, 16, 24]

fig1, axes1 = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
plt.subplot(1,3,1)
plt.imshow(noisy_data.as_array()[tindex[0],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[0]))
plt.subplot(1,3,2)
plt.imshow(noisy_data.as_array()[tindex[1],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[1]))
plt.subplot(1,3,3)
plt.imshow(noisy_data.as_array()[tindex[2],:,:])
plt.axis('off')
plt.title('Time {}'.format(tindex[2]))

fig1.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)
plt.tight_layout()
plt.show()   

# Regularisation Parameter
alpha = 0.3

# Create Gradient operators with different Space - Channel correlation
op1 = Gradient(ig, correlation='Space') # No gradient in temporal direction
op2 = Gradient(ig, correlation='SpaceChannels') # SpatioTemporal Gradient
op3 = Identity(ig, ag)

# Create BlockOperator
operator1 = BlockOperator(op1, op3, shape=(2,1) ) 
operator2 = BlockOperator(op2, op3, shape=(2,1) )

# Create functions     
f1 = alpha * MixedL21Norm()
f2 = 0.5 * L2NormSquared(b = noisy_data)    
f = BlockFunction(f1, f2)                                                                          
g = ZeroFunction()
    
# Compute operator Norm
normK1 = operator1.norm()
normK2 = operator2.norm()


# Primal & dual stepsizes
sigma1 = 1
tau1 = 1/(sigma1*normK1**2)

sigma2 = 1
tau2 = 1/(sigma2*normK2**2)

# Setup and run the PDHG algorithm
pdhg1 = PDHG(f=f,g=g,operator=operator1, tau=tau1, sigma=sigma1)
pdhg1.max_iteration = 2000
pdhg1.update_objective_interval = 200
pdhg1.run(1000)

# Setup and run the PDHG algorithm
pdhg2 = PDHG(f=f,g=g,operator=operator2, tau=tau2, sigma=sigma2)
pdhg2.max_iteration = 2000
pdhg2.update_objective_interval = 200
pdhg2.run(1000)

#%%
tindex = [8, 16, 24]
fig2, axes2 = plt.subplots(nrows=3, ncols=3, figsize=(10, 10))

# Ground Truth
axes2[0, 0].imshow(phantom_2Dt[tindex[0],:,:])
axes2[0, 0].set_title('Time {}'.format(tindex[0]))
axes2[0,0].set_ylabel('Ground Truth')


axes2[0, 1].imshow(phantom_2Dt[tindex[1],:,:])
axes2[0, 1].set_title('Time {}'.format(tindex[1]))
axes2[0,1].axis('off')

axes2[0, 2].imshow(phantom_2Dt[tindex[2],:,:])
axes2[0, 2].set_title('Time {}'.format(tindex[2]))
axes2[0,2].axis('off')

# Space Correlation 
axes2[1, 0].imshow(pdhg1.get_output().as_array()[tindex[0],:,:], vmin=0, vmax =1)
axes2[1,0].set_ylabel('Space \n Correlation')

axes2[1, 1].imshow(pdhg1.get_output().as_array()[tindex[1],:,:])
axes2[1,1].axis('off')
axes2[1, 2].imshow(pdhg1.get_output().as_array()[tindex[2],:,:])
axes2[1,2].axis('off')

# Space Channel Correlation 
axes2[2, 0].imshow(pdhg2.get_output().as_array()[tindex[0],:,:])
axes2[2,0].set_ylabel('SpaceTime \n Correlation')

axes2[2, 1].imshow(pdhg2.get_output().as_array()[tindex[1],:,:])
axes2[2,1].axis('off')

axes2[2, 2].imshow(pdhg2.get_output().as_array()[tindex[2],:,:])
axes2[2,2].axis('off')


fig2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

plt.tight_layout()
plt.show()

#%%
# Animate
fig3 = plt.figure()

sol1 = pdhg1.get_output().as_array()
sol2 = pdhg2.get_output().as_array()

ims1 = []
ims2 = []
ims3 = []

for sl in range(0,np.shape(phantom_2Dt)[0]):
    
    plt.subplot(1,3,1)
    im1 = plt.imshow(phantom_2Dt[sl,:,:], animated=True)
    plt.title('Ground truth')
    
    plt.subplot(1,3,2)
    im2 = plt.imshow(sol1[sl,:,:], animated=True) 
    plt.title('Space \n Correlation')
    
    plt.subplot(1,3,3)
    im3 = plt.imshow(sol2[sl,:,:], animated=True)  
    plt.title('Space-Time \n Correlation')
    
    ims1.append([im1])
    ims2.append([im2])   
    ims3.append([im3])       

    
    
ani1 = animation.ArtistAnimation(fig3, ims1, interval=500,
                                repeat_delay=10)

ani2 = animation.ArtistAnimation(fig3, ims2, interval=500,
                                repeat_delay=10)

ani3 = animation.ArtistAnimation(fig3, ims3, interval=500,
                                repeat_delay=10)

plt.show() 








