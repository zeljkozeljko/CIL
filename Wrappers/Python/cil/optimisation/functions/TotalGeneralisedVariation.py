

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

        if not hasattr(self, 'domain'):
            self.domain = x.geometry
            
        if not hasattr(self, 'operator'):
            
            self.Gradient = GradientOperator(self.domain, correlation = self.correlation, backend = self.backend)  
            self.SymGradient = SymmetrisedGradientOperator(self.gradient.range, correlation = self.correlation, backend = self.backend)  
            self.ZeroOperator = ZeroOperator(self.domain, self.sym_gradient.range)
            self.IdentityOperator = - IdentityOperator(self.gradient.range)

            #    BlockOperator = [ Gradient      - Identity  ]
            #                    [ ZeroOperator   SymGradient] 
            self.BlockOperator = BlockOperator(self.Gradient, self.IdentityOperator, self.ZeroOperator, self.SymGradient, shape=(2,2))         

        res = self.BlockOperator.direct(x)

        return res


        
    def proximal(self, x, tau, out = None):
        
        if not hasattr(self, 'domain'):
            self.domain = x.geometry
            
        if not hasattr(self, 'operator'):
            
            self.Gradient = GradientOperator(self.domain, correlation = self.correlation, backend = self.backend)  
            self.SymGradient = SymmetrisedGradientOperator(self.gradient.range, correlation = self.correlation, backend = self.backend)  
            self.ZeroOperator = ZeroOperator(self.domain, self.sym_gradient.range)
            self.IdentityOperator = - IdentityOperator(self.gradient.range)

            #    BlockOperator = [ Gradient      - Identity  ]
            #                    [ ZeroOperator   SymGradient] 
            self.BlockOperator = BlockOperator(self.Gradient, self.IdentityOperator, self.ZeroOperator, self.SymGradient, shape=(2,2))    

        if not all(hasattr(self, attr) for attr in ["g1", "g"]):
            self.g1 = 0.5*L2NormSquared(b = x)
            self.g = BlockFunction(tau*self.g1, tau*self.g2) 
                    
        # if not hasattr(self, 'pdhg'):

        #TODO warm start is better, use initial
        # self.initial not self.pdhg.initial
        self.pdhg = PDHG(f = self.f, g=self.g, operator = self.BlockOp,
                   update_objective_interval = self.iterations, 
                   max_iteration = self.iterations)

        self.pdhg.run(verbose=self.verbose)        
        return self.pdhg.solution

    def convex_conjugate(self,x):        
        return 0.0    