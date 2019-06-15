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

from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, KullbackLeibler, \
                      MixedL21Norm, BlockFunction

from ccpi.astra.operators import AstraProjectorMC, AstraProjectorSimple

import os
import tomophantom
from tomophantom import TomoP2D
import matplotlib.animation as animation

# Select device
device = input('Available device: GPU==1 / CPU==0 ')
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'

# Create phantom for TV 2D dynamic tomography 

model = 102  # note that the selected model is temporal (2D + time)
N = 50 # set dimension of the phantom
# one can specify an exact path to the parameters file
# path_library2D = '../../../PhantomLibrary/models/Phantom2DLibrary.dat'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
#This will generate a N_size x N_size x Time frames phantom (2D + time)

phantom_2Dt = TomoP2D.ModelTemporal(model, N, path_library2D)[0:-1:2]


fig = plt.figure()
ims1 = []
for sl in range(0,np.shape(phantom_2Dt)[0]):    
    im1 = plt.imshow(phantom_2Dt[sl,:,:], animated=True, vmin=0, vmax=1)    
    ims1.append([im1])
ani1 = animation.ArtistAnimation(fig, ims1, interval=500,repeat_delay=10)
plt.show() 

    
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, channels = np.shape(phantom_2Dt)[0])
data = ImageData(phantom_2Dt, geometry=ig)

detectors = N
angles = np.linspace(0,np.pi,180)

ag = AcquisitionGeometry('parallel','2D', angles, detectors, channels = np.shape(phantom_2Dt)[0])
Aop = AstraProjectorMC(ig, ag, dev)
sin = Aop.direct(data)

scale = 2
n1 = scale * np.random.poisson(sin.as_array()/scale)
noisy_data = AcquisitionData(n1, ag)

# Regularisation Parameter
alpha = 10

# Create operators
#op1 = Gradient(ig)
op1 = Gradient(ig, correlation='SpaceChannels')
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Create functions
      
f1 = alpha * MixedL21Norm()
f2 = KullbackLeibler(noisy_data)    
f = BlockFunction(f1, f2)  
                                      
g = ZeroFunction()
    
# Compute operator Norm
igtmp = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
agtmp = AcquisitionGeometry('parallel','2D', angles, detectors)
Atmp = AstraProjectorSimple(igtmp, agtmp, dev)
normK = np.sqrt(op1.norm()**2 + Atmp.norm()**2)

# Primal & dual stepsizes
sigma = 5
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma, memopt=True)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
pdhg.run(1000)


tindex = [0, int(phantom_2Dt.shape[0]/2), phantom_2Dt.shape[0]-1]
fig2, axes2 = plt.subplots(nrows=2, ncols=3, figsize=(5, 5))

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
axes2[1, 0].imshow(pdhg.get_output().as_array()[tindex[0],:,:])
axes2[1,0].set_ylabel('TV reconstruction')

axes2[1, 1].imshow(pdhg.get_output().as_array()[tindex[1],:,:])
axes2[1,1].axis('off')
axes2[1, 2].imshow(pdhg.get_output().as_array()[tindex[2],:,:])
axes2[1,2].axis('off')

fig2.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.8,
                    wspace=0.02, hspace=0.02)

plt.tight_layout()
plt.show()

# Animate
fig3 = plt.figure()

sol1 = pdhg.get_output().as_array()

ims1 = []
ims2 = []

for sl in range(0,np.shape(phantom_2Dt)[0]):
    
    plt.subplot(1,2,1)
    im1 = plt.imshow(phantom_2Dt[sl,:,:], animated=True)
    plt.title('Ground truth')
    
    plt.subplot(1,2,2)
    im2 = plt.imshow(sol1[sl,:,:], animated=True) 
    plt.title('TV reconstruction')
        
    ims1.append([im1])
    ims2.append([im2])       

    
ani1 = animation.ArtistAnimation(fig3, ims1, interval=500,
                                repeat_delay=10)

ani2 = animation.ArtistAnimation(fig3, ims2, interval=500,
                                repeat_delay=10)


plt.show() 


