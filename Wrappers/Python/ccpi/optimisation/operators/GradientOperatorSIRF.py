# -*- coding: utf-8 -*-
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ccpi.optimisation.operators import LinearOperator, Gradient, \
                                 FiniteDiff, ScaledOperator
from ccpi.framework import BlockGeometry, ImageGeometry
import numpy


class GradientSIRF(LinearOperator):
    
                        
    def __init__(self, gm_domain, bnd_cond = 'Neumann', **kwargs):
        
        super(LinearOperator, self).__init__() 
                
        self.gm_domain = gm_domain # Domain of Grad Operator
        
        self.correlation = kwargs.get('correlation',Gradient.CORRELATION_SPACE)
        
        if self.correlation==Gradient.CORRELATION_SPACE:
            # SIRF ImageData has maximum 3 Dimensions
            self.gm_range = BlockGeometry(*[self.gm_domain for _ in range(len(self.gm_domain.shape)) ] )
            
            if len(self.gm_domain.shape) == 3:                
                # 3D
                # expected Grad_order = ['direction_z', 'direction_y', 'direction_x']
                expected_order = [ImageGeometry.VERTICAL, ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]
                self.ind = [2,1,0]
            else:
                # 2D
                expected_order = [ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X]    
                self.ind = [1,0]
                
            if self.gm_domain.shape[0] == 1:
                self.tmp_ind = 1
            else:
                self.tmp_ind = 0
                
            
        self.bnd_cond = bnd_cond 
        
        # Call FiniteDiff operator        
        self.FD = FiniteDiff(self.gm_domain, direction = 0, bnd_cond = self.bnd_cond)
                                                         
    def direct(self, x, out=None):
        
                
        if out is not None:
            
            for i in range(self.gm_range.shape[0]):
                self.FD.direction = self.ind[i]
                self.FD.direct(x, out = out[i])
        else:
            tmp = self.gm_range.allocate()        
            for i in range(tmp.shape[0]):
                self.FD.direction=self.ind[i]
                tmp.get_item(i).fill(self.FD.direct(x))
            return tmp    
        
    def adjoint(self, x, out=None):
        
        if out is not None:

            tmp = self.gm_domain.allocate()            
            for i in range(x.shape[0] - self.tmp_ind):
                self.FD.direction=self.ind[i] 
                self.FD.adjoint(x.get_item(i), out = tmp)
                if i == 0:
                    out.fill(tmp)
                else:
                    out += tmp
        else:            
            tmp = self.gm_domain.allocate()
            for i in range(x.shape[0] - self.tmp_ind):
                self.FD.direction=self.ind[i]

                tmp += self.FD.adjoint(x.get_item(i))
            return tmp    
            
    def calculate_norm(self):
        return numpy.sqrt(8)

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