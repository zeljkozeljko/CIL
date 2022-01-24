# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


from cil.optimisation.functions import Function, BlockFunction, MixedL21Norm, ZeroFunction, L2NormSquared
from cil.optimisation.operators import GradientOperator, BlockOperator,IdentityOperator, ZeroOperator, SymmetrisedGradientOperator
from cil.optimisation.algorithms import PDHG

class TotalGeneralisedVariation(Function):
            
    def __init__(self,
                 alpha = 1.0,
                 beta = 2.0,
                 max_iteration=100, 
                 correlation = "Space",
                 backend = "c",
                 split = False,
                 verbose = 0):
        
        super(TotalGeneralisedVariation, self).__init__(L = None)

        # regularisation parameters for TGV
        self.alpha = alpha
        self.beta = beta
                
        # Iterations for PDHG_TGV
        self.iterations = max_iteration
        
        # correlation space or spacechannels
        self.correlation = correlation

        # backend for the gradient
        self.backend = backend        
        
        # splitting Gradient
        self.split = split
                        
        # parameters to set up PDHG algorithm
        self.f1 = self.alpha * MixedL21Norm()
        self.f2 = self.beta * MixedL21Norm()
        self.f = BlockFunction(self.f1, self.f2)              
        self.g2 = ZeroFunction()     
        
        self.verbose = verbose

    def __call__(self, x):
        
        if not hasattr(self, 'pdhg'):   
            return 0.0        
        else:              
            # Compute alpha * || Du - w || + beta * ||Ew||, 
            # where (u,w) are solutions coming from the proximal method below.

            # An alternative option is to use 
            # self.f(self.pdhg.y_tmp) where y_tmp contains the same information as
            # self.pdhg.operator.direct(self.pdhg.solution)
            
            # However, this only works if we use pdhg.update_objective() and that means
            # that we need to compute also the adjoint operation for both Gradient and the SymmetrizedGrafientOperator.

            tmp = self.f(self.pdhg.operator.direct(self.pdhg.solution))
            return tmp
        
    def proximal(self, x, tau, out = None, gamma_g=None):
        
        if not hasattr(self, 'domain'):
            self.domain = x.geometry
            
        if not hasattr(self, 'operator'):
            
            self.Gradient = GradientOperator(self.domain, correlation = self.correlation, backend = self.backend)  
            self.SymGradient = SymmetrisedGradientOperator(self.Gradient.range, correlation = self.correlation, backend = self.backend)  
            self.ZeroOperator = ZeroOperator(self.domain, self.SymGradient.range)
            self.IdentityOperator = - IdentityOperator(self.Gradient.range)

            #    BlockOperator = [ Gradient      - Identity  ]
            #                    [ ZeroOperator   SymGradient] 
            self.operator = BlockOperator(self.Gradient, self.IdentityOperator, 
                                               self.ZeroOperator, self.SymGradient, 
                                               shape=(2,2))    

        if not all(hasattr(self, attr) for attr in ["g1", "g"]):
            self.g1 = L2NormSquared(b = x)
            self.g = BlockFunction(self.g1, self.g2)
            
        # setup PDHG    
        self.alpha *=tau
        self.beta *=tau
        
        # configure PDHG only once. This has the advantage of warm-starting in 
        # the case where this proximal method is used as an inner solver inside an algorithm.
        # That means we run .proximal for 100 iterations for one iteration of the outer algorithm,
        # and in the next iteration, we run 100 iterations of the inner solver, but we begin where we stopped before.

        if not hasattr(self, 'pdhg'):            
            self.pdhg = PDHG(f = self.f, g=self.g, operator = self.operator,
                       update_objective_interval = self.iterations,
                       max_iteration = self.iterations, gamma_g = gamma_g)

        # Avoid using pdhg.run() because of print messages (configure PDHG)
        for _ in range(self.iterations):
            self.pdhg.__next__()

        # need to reset, iteration attribute for pdhg
        self.pdhg.iteration=0
                    
        if out is None:
            return self.pdhg.solution[0]
        else:
            out.fill(self.pdhg.solution[0])

    def convex_conjugate(self,x):        
        return 0.0