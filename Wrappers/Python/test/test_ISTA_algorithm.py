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


from cil.optimisation.algorithms import ISTA
from cil.optimisation.operators import MatrixOperator
from cil.framework import VectorData
from cil.optimisation.functions import LeastSquares, ZeroFunction

import numpy as np


import unittest

class TestISTA(unittest.TestCase):

    def setUp(self):
        
        n = 100
        m = 1000

        A = np.random.uniform(0,10, (m, n)).astype('float32')
        b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32')

        self.Aop = MatrixOperator(A)
        self.bop = VectorData(b) 

        self.f = LeastSquares(self.Aop, b=self.bop, c=0.5)
        self.g = ZeroFunction()

        self.ig = self.Aop.domain

        self.initial = self.ig.allocate()
  
    def tearDown(self):
        pass   

    def test_signature(self):

        # check required arguments (initial, f, g)
        with np.testing.assert_raises(TypeError):
            ista = ISTA(f = self.f, g = self.g)

        with np.testing.assert_raises(TypeError):
            ista = ISTA(initial = self.initial, f = self.f)            

        with np.testing.assert_raises(TypeError):
            ista = ISTA(initial = self.initial, g = self.g) 

        # ista no step-size
        ista = ISTA(initial = self.initial, f = self.f, g = self.g)  
        np.testing.assert_equal(ista.step_size, 1./self.f.L)

        # ista step-size
        tmp_step_size = 10.
        ista = ISTA(initial = self.initial, f = self.f, g = self.g, step_size=tmp_step_size)  
        np.testing.assert_equal(ista.step_size, tmp_step_size)    

        # check initialisation
        self.assertTrue( id(ista.x)!=id(ista.initial) )   
        self.assertTrue( id(ista.x_old)!=id(ista.initial))              

    def test_update(self):

        # ista run 1 iteration

        tmp_initial = self.ig.allocate()
        ista = ISTA(initial = tmp_initial, f = self.f, g = self.g)  
        ista.run(1)

        x = tmp_initial.copy()
        x_old = tmp_initial.copy()

        for _ in range(11):         
            x = ista.g.proximal(x_old - (1./ista.f.L) * ista.f.gradient(x_old), (1./ista.f.L))
            x_old.fill(x)


        np.testing.assert_allclose(ista.solution.array, x.array, atol=1e-3)      
    
        # # check objective
        # res1 = ista.objective[-1]
        # res2 = self.f(x) + self.g(x)
        # self.assertTrue( res1==res2) 
