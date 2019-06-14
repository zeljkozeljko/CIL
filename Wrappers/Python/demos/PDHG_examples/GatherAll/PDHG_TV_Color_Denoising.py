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
Total Variation Color Denoising using PDHG algorithm:
    
Problem:     min_u  \alpha * ||\nabla u||_{2,1} + ||x-g||_{1}
             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
             
             g: Noisy Data with Salt & Pepper Noise
             
             K = \nabla    
                          
"""

import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import Gradient, BlockOperator, FiniteDiff
from ccpi.optimisation.functions import MixedL21Norm, L2NormSquared, BlockFunction, L1Norm, KullbackLeibler                     
from ccpi.framework import TestData, ImageGeometry
import os, sys
if int(numpy.version.version.split('.')[1]) > 12:
    from skimage.util import random_noise
else:
    from demoutil import random_noise

# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 0
print ("Applying {} noise")

loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.PEPPERS, size=(256,256))
ig = data.geometry
ag = ig

# Create noisy data. 
noises = ['gaussian', 'poisson', 's&p']
noise = noises[which_noise]
if noise == 's&p':
    n1 = random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2)
elif noise == 'poisson':
    scale = 5
    n1 = random_noise( data.as_array()/scale, mode = noise, seed = 10)*scale
elif noise == 'gaussian':
    n1 = random_noise(data.as_array(), mode = noise, seed = 10)
else:
    raise ValueError('Unsupported Noise ', noise)
noisy_data = ig.allocate()
noisy_data.fill(n1)


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

operator = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)

f1 =  alpha * MixedL21Norm()

# fidelity
if noise == 's&p':
    g = L1Norm(b=noisy_data)
elif noise == 'poisson':
    g = KullbackLeibler(noisy_data)
elif noise == 'gaussian':
    g = 0.5 * L2NormSquared(b=noisy_data)
            
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 1
tau = 1/(sigma*normK**2)

# Setup and run the PDHG algorithm
pdhg1 = PDHG(f=f1,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg1.max_iteration = 2000
pdhg1.update_objective_interval = 200
pdhg1.run(1000)

# Show results
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.subplot(1,3,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.subplot(1,3,3)
plt.imshow(pdhg1.get_output().as_array())
plt.title('TV Reconstruction')
plt.show()

