# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:02:25 2022

@author: dv516
"""

import numpy as np
from typing import List, Tuple
from Functions.test_functions import f_d2, f_d3
import pybobyqa

def optimizer_dummy(f, N_x: int, bounds: List[Tuple[float]], N: int = 100) -> (float, List[float]):
# '''
# Optimizer aims to optimize a black-box function 'f' using the dimensionality
# 'N_x', and box-'bounds' on the decision vector
# Input:
# f: function: taking as input a list of size N_x and outputing a float
# N_x: int: number of dimensions
# N: int: optional: Evaluation budget
# bounds: List of size N where each element i is a tuple conisting of 2 floats
# (lower, upper) serving as box-bounds on the ith element of x
# Return:
# tuple: 1st element: lowest value found for f, f_min
# 2nd element: list/array of size N_x giving the decision variables
# associated with f_min
# '''
    if N_x != len(bounds):
        raise ValueError('Nbr of variables N_x does not match length of bounds')
    bounds = np.array(bounds)
   
    x0 = np.array([(b[0] + b[1])/2 for b in bounds]) 
   
    sol = pybobyqa.solve(f, x0, bounds=bounds.T, maxfun=N, seek_global_minimum=True) 
    
    return sol.f, sol.x


N_x = 2
bounds = [(-2.0, 2.0) for i in range(N_x)]
test1 = optimizer_dummy(f_d2, N_x, bounds, N=10000)
N_x = 3
bounds = [(-2.0, 2.0) for i in range(N_x)]
test2 = optimizer_dummy(f_d3, N_x, bounds, N=10000)

np.testing.assert_array_less(test1[0], 1e-3)
np.testing.assert_array_less(test2[0], 1e-3)









