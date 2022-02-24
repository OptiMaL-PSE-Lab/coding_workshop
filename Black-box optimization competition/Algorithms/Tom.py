# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:02:25 2022

@author: dv516
"""

import numpy as np
from typing import List, Tuple

from Functions.test_functions import f_d2, f_d3

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
    ### Your code here
    iterations = 5
    surrogate_size = int(N/iterations)
    x_current = np.mean(bounds,axis=1)
    x_ranges = bounds[:,1]-bounds[:,0]

    def LHS(bounds,p):
        d = len(bounds)
        sample = np.zeros((p,len(bounds)))
        for i in range(0,d):
            sample[:,i] = np.linspace(bounds[i,0],bounds[i,1],p)
            np.random.shuffle(sample[:,i])
        return sample

    for iter in range(iterations):
        bounds_sample = np.array([x_current-0.5*x_ranges,x_current+0.5*x_ranges])


        samples = LHS(bounds_sample.T,surrogate_size)

        f_sampled = []
        for sample in samples:
            f_sampled.append(f(sample))

        f_sampled = (np.array(f_sampled)-min(f_sampled))/(max(f_sampled)-min(f_sampled))
        x_current = samples[np.argsort(f_sampled)[0]]

        x_ranges *= 0.5

###

    return f(x_current), x_current

N_x = 2
bounds = [(-2.0, 2.0) for i in range(N_x)]
test1 = optimizer_dummy(f_d2, N_x, bounds, N=10000)
N_x = 3
bounds = [(-2.0, 2.0) for i in range(N_x)]
test2 = optimizer_dummy(f_d3, N_x, bounds, N=10000)

np.testing.assert_array_less(test1[0], 1e-3)
np.testing.assert_array_less(test2[0], 1e-3)









