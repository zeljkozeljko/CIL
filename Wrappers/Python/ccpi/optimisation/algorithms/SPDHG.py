# -*- coding: utf-8 -*-
# Copyright 2020 Science Technology Facilities Council
# Copyright 2020 University of Manchester
# Copyright 2020 University of Bath
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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from ccpi.optimisation.algorithms import Algorithm, DataContainerWithHistory
from ccpi.optimisation.operators import BlockOperator
from ccpi.optimisation.functions import BlockFunction
import numpy as np

def select_subset(self, index):
    self.current.geometry.subset_id = index
    self.previous.geometry.subset_id = index
DataContainerWithHistory.__getitem__ = select_subset

    
class SPDHGOperator(BlockOperator):
    def __init__(self, *args, **kwargs):
        super(SPDHGOperator, self).__init__(*args, **kwargs)
        self.num_physical_subsets = kwargs.get('num_physical_subsets', 1)
        index_of_operator = []
        for i, op in enumerate(self.operators):
            if self.is_subset_operator(op):
                for j in range(self.num_physical_subsets):
                    index_of_operator.append(i)
            else:
                index_of_operator.append(i)
        self.index_of_operator = index_of_operator

    def is_subset_operator(self, operator):
        if hasattr(operator, 'is_subset_operator'):
            return operator.is_subset_operator
        return False

    
    @property
    def max_operator_index(self):
        '''returns the maximum number of operators taking into account the physical subsets'''
        return len(self.operators) + self.num_physical_subsets - 1

    def __getitem__(self, index):
        '''returns the appropriate operator'''
        if index > self.max_operator_index:
            raise ValueError('Index out of range {}. Max {}'\
                .format(index, self.max_operator_index))
        operator = self.operators[self.index_of_operator[index]]
        if self.is_subset_operator(operator):
            # select subset from index
            physical_subset_index = index - self.index_of_operator[index]
            operator.select_subset(physical_subset_index, self.num_physical_subsets)
        return operator

    def __len__(self):
        '''alias of max_operator_index'''
        return self.max_operator_index

class SPDHGFunction(BlockFunction):
    def __init__(self, functions, operators):
        super(SPDHGFunction, self).__init__(*functions)
        self.functions = functions
        # goes in sync with the operator
        # save a reference to operators
        self.operators = operators
    def __getitem__(self, index):
        # if at index there is a subset operator, the selected function is
        # obtained by first to select the subset in the function and then 
        # return the function at list_index
        # otherwise just returns the function at list_index

        list_index = self.operators.index_of_operator[index]
        operator = self.operators[index]
        
        if operator.is_subset_operator(operator):
            physical_subset_index = operator.subset_id
            num_physical_subsets = operator.num_subsets
            # select the function and select the physical subset
            self.functions[list_index]\
                .select_subset(physical_subset_index, num_physical_subsets)
        return self.functions[list_index]

class SPDHGFactory(object):
    @staticmethod
    def get_algorithm(f, g, operator, tau=None, sigma=None, \
                      x_init=None, prob=None, gamma=1., norms=None,\
                      max_iteration=1, update_objective_interval=1,
                      data = None,
                      num_physical_subsets=1, physical_subsets_method='stagger'):
        '''Creates an instance of SPDHG configured with the parameters passed by the user

        '''
        rho = 0.99
        if isinstance(operator, BlockOperator):
            # convert to SPDHOperator
            K = SPDHGOperator(*operator.operators, 
                               num_physical_subsets=num_physical_subsets)
        else:
            K = SPDHGOperator(operator)
        F = SPDHGFunction(f, K)
        
        # get the number of dual subsets
        num_dual_subsets = len(K)

        if prob is None:
            prob = []
            equal_probability = 1/len(K.operators)

            for i in range(num_dual_subsets):
                op = K[i]
                if K.is_subset_operator(op):
                    prob.append(equal_probability/num_physical_subsets) 
                else:
                    prob.append(equal_probability)
        else:
            if len(prob) != len(K):
                raise ValueError('prob length is wrong. Expecting {} got {}'\
                    .forma(len(K), len(prob)))
        
        
        if sigma is None:
            if norms is None:
                # Compute norm of each sub-operator       
                norms = [K[i].norm() for i in range(num_dual_subsets)]
            
            sigma = [gamma * rho / ni for ni in norms] 
        if tau is None:
            tau = min( [ pi / ( si * ni**2 ) \
                for pi, ni, si in zip(prob, norms, sigma)] ) 
            tau *= (rho / gamma)

        # we should be good now
        algo = SPDHG(max_iteration=max_iteration, 
           update_objective_interval=update_objective_interval)
        
        #reset dataset to 1 subset
        if data.num_subsets > 1:
            #save pre-selected subset?
            data.generate_subsets(1,physical_subset_method)

        algo.set_up(F, g, K, tau=tau, sigma=tau, \
                x_init=x_init, prob=prob, gamma=gamma, norms=norms)
        
        # generate physical subsets
        data.generate_subsets(num_physical_subsets, physical_subsets_method)
        algo.y.override_subsets (data.geometry)
        algo.y_old.override_subsets (data.geometry)
        return algo
        


class SPDHG(Algorithm):
    r'''Stochastic Primal Dual Hybrid Gradient
    
    Problem: 
    
    .. math::
    
      \min_{x} f(Kx) + g(x) = \min_{x} \sum f_i(K_i x) + g(x)
        
    :param operator: BlockOperator of Linear Operators
    :param f: BlockFunction, each function with "simple" proximal of its conjugate 
    :param g: Convex function with "simple" proximal 
    :param sigma=(sigma_i): List of Step size parameters for Dual problem
    :param tau: Step size parameter for Primal problem
    :param x_init: Initial guess ( Default x_init = 0)
    :param prob: List of probabilities
        
    Remark: Convergence is guaranted provided that [2, eq. (12)]:
        
    .. math:: 
    
      \|\sigma[i]^{1/2} * K[i] * tau^{1/2} \|^2  < p_i for all i
      
    Remark: Notation for primal and dual step-sizes are reversed with comparison
            to PDGH.py
            
    Remark: this code implements serial sampling only, as presented in [2]
            (to be extended to more general case of [1] as future work)             
            
    References:
        
        [1]"Stochastic primal-dual hybrid gradient algorithm with arbitrary 
        sampling and imaging applications",
        Chambolle, Antonin, Matthias J. Ehrhardt, Peter Richtárik, and Carola-Bibiane Schonlieb,
        SIAM Journal on Optimization 28, no. 4 (2018): 2783-2808.   
         
        [2]"Faster PET reconstruction with non-smooth priors by randomization and preconditioning",
        Matthias J Ehrhardt, Pawel Markiewicz and Carola-Bibiane Schönlieb,
        Physics in Medicine & Biology, Volume 64, Number 22, 2019.
        
        
    '''
    def __init__(self, f=None, g=None, operator=None, 
                       tau=None, sigma=None, norms=None,
                       x_init=None, gamma=1.,
                       prob=None, 
                       use_axpby=True, 
                       **kwargs):
        '''SPDHG algorithm creator

        Parameters
        :param operator: BlockOperator of Linear Operators
        :param f: BlockFunction, each function with "simple" proximal of its conjugate 
        :param g: Convex function with "simple" proximal 
        :param sigma=(sigma_i): List of Step size parameters for Dual problem
        :param tau: Step size parameter for Primal problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param prob: List of probabilities for each operator
        :param gamma: parameter controlling the trade-off between the primal and dual step sizes
        :param use_axpby: whether to use axpby or not
        :param norms: norms of the operators in operator
        :type norms: list, default None
        '''
        super(SPDHG, self).__init__(**kwargs)
        self._use_axpby = use_axpby
        if f is not None and operator is not None and g is not None:
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma, 
                        x_init=x_init, prob=prob, gamma=gamma, norms=norms)
    
    def set_up(self, f, g, operator, tau=None, sigma=None, \
                x_init=None, prob=None, gamma=1., norms=None):
        '''initialisation of the algorithm

        :param operator: BlockOperator of Linear Operators
        :param f: BlockFunction, each function with "simple" proximal of its conjugate.
        :param g: Convex function with "simple" proximal 
        :param sigma: list of Step size parameters for dual problem
        :param tau: Step size parameter for primal problem
        :param x_init: Initial guess ( Default x_init = 0)
        :param prob: List of probabilities
        :param norms: List of norm of the operators in operator.
        '''
        print("{} setting up".format(self.__class__.__name__, ))
                    
        # algorithmic parameters
        self.f = f
        self.g = g
        self.operator = operator
        self.tau = tau
        self.sigma = sigma
        self.prob = prob
        self.num_dual_subsets = len(self.operator)
        self.gamma = gamma
        self.rho = .99
        
        if self.prob is None:
            self.prob = [1/self.num_dual_subsets] * self.num_dual_subsets

        
        if self.sigma is None:
            if norms is None:
                # Compute norm of each sub-operator       
                norms = [operator[i].norm() for i in range(self.num_dual_subsets)]
            self.norms = norms
            self.sigma = [self.gamma * self.rho / ni for ni in norms] 
        if self.tau is None:
            self.tau = min( [ pi / ( si * ni**2 ) for pi, ni, si in zip(self.prob, norms, self.sigma)] ) 
            self.tau *= (self.rho / self.gamma)

        # initialize primal variable 
        if x_init is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = x_init.copy()
        
        self.x_tmp = self.operator.domain_geometry().allocate(0)
        
        # initialize dual variable to 0
        self._y = DataContainerWithHistory(operator.range_geometry(), 0)
        
        # initialize variable z corresponding to back-projected dual variable
        self.z = operator.domain_geometry().allocate(0)
        self.zbar= operator.domain_geometry().allocate(0)
        # relaxation parameter
        self.theta = 1
        self.update_objective()
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))
    def update(self):
        # Gradient descent for the primal variable
        # x_tmp = x - tau * zbar
        if self._use_axpby:
            self.x.axpby(1., -self.tau, self.zbar, out=self.x_tmp)
        else:
            self.zbar.multiply(self.tau, out=self.x_tmp)
            self.x.subtract(self.x_tmp, out=self.x_tmp)
            
        self.g.proximal(self.x_tmp, self.tau, out=self.x)
        
        # Choose subset
        i = int(np.random.choice(len(self.sigma), 1, p=self.prob))
        
        # save previous iteration
        self.save_previous_iteration(i)
        
        # Gradient ascent for the dual variable
        # y[i] = y_old[i] + sigma[i] * K[i] x
        self.operator[i].direct(self.x, out=self.y[i])
        if self._use_axpby:
            self.y[i].axpby(self.sigma[i], 1., self.y_old[i], out=self.y[i])
        else:
            self.y[i].multiply(self.sigma[i], out=self.y[i])
            self.y[i].add(self.y_old[i], out=self.y[i])
            
        self.f[i].proximal_conjugate(self.y[i], self.sigma[i], out=self.y[i])
        
        # Back-project
        # x_tmp = K[i]^*(y[i] - y_old[i])
        self.y[i].subtract(self.y_old[i], out=self.y_old[i])
        self.operator[i].adjoint(self.y_old[i], out = self.x_tmp)
        # Update backprojected dual variable and extrapolate
        # zbar = z + (1 + theta/p[i]) x_tmp

        # z = z + x_tmp
        self.z.add(self.x_tmp, out =self.z)
        # zbar = z + (theta/p[i]) * x_tmp
        if self._use_axpby:
            self.z.axpby(1., self.theta / self.prob[i], self.x_tmp, out = self.zbar)
        else:
            self.x_tmp.multiply(self.theta / self.prob[i], out=self.x_tmp)
            self.z.add(self.x_tmp, out=self.zbar)
        
    def update_objective(self):
         p1 = self.f(self.operator.direct(self.x)) + self.g(self.x)
         d1 = -(self.f.convex_conjugate(self.y) + self.g.convex_conjugate(-1*self.operator.adjoint(self.y)))
#
         self.loss.append([p1, d1, p1-d1])

    @property
    def objective(self):
         '''alias of loss'''
         return [x[0] for x in self.loss]
    @property
    def dual_objective(self):
        return [x[1] for x in self.loss]
    
    @property
    def primal_dual_gap(self):
        return [x[2] for x in self.loss]
    @property
    def y(self):
        return self._y.current
    @property
    def y_old(self): # change to previous
        return self._y.previous
    def save_previous_iteration(self, index):
        # swaps the reference in the BlockDataContainers
        self._y.current.containers , self._y.previous.containers = \
            swap_element_from_tuples( self._y.current.containers, self._y.previous.containers, index )


def swap_element_from_tuples(tuple1, tuple2, index):
    '''swap element at index from tuple1 and tuple2, returns 2 new tuples'''
    a = tuple1[index]
    b = tuple2[index]
    ca = create_and_replace_element_in_tuple(tuple1, index, b)
    cb = create_and_replace_element_in_tuple(tuple2, index, a)
    return ca,cb
    
def create_and_replace_element_in_tuple(dtuple, index, new_element):
    '''replace element at index in a tuple with new_element by creating a list and returning a new tuple'''
    dlist = []
    for i in range(len(dtuple)):
        if i == index:
            dlist.append(new_element)
        else:
            dlist.append(dtuple[i])
    return tuple(dlist)
    

