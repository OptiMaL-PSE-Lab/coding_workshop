# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:02:25 2022

@author: dv516
"""

import numpy as np
from typing import List, Tuple
from Functions.test_functions import f_d2, f_d3

import random
import time

def ParticleSwarmOptimization(low_bound, high_bound, dimension, function, number_particles, cognitive_constant, social_constant, max_iteration, max_plot):
  fig, (plot1, plot2) = plt.subplots(2, figsize=(20,20))
  plot1.set_xlabel('Number of Iterations')
  plot1.set_ylabel('Value of Objective Function')
  plot1.set_yscale("log")
  plot2.set_xlabel('Time (s)')
  plot2.set_ylabel('Value of Objective Function')
  plot2.set_yscale("log")
  plot = 0

  while plot < max_plot:
    start = time.time()
    particles = np.random.uniform(low_bound, high_bound, size=(number_particles, dimension))
    particle_best_input = np.random.uniform(1000, 1000000, size=(number_particles, dimension))
    particle_best_output = np.random.uniform(1000, 1000000, size=(number_particles, 1))
    global x_axis
    global y_axis
    global time_axis
    global global_best_output
    x_axis     = []
    y_axis     = []
    time_axis  = []

    global_best_input = np.random.uniform(low=1000, high=1000000, size=(dimension,))
    global_best_output = 1e99

    iteration = 0
    c1 = cognitive_constant
    c2 = social_constant
    velocity = np.random.uniform(0, 2, size=(number_particles, dimension))

    while iteration < max_iteration:

      particles_output = np.empty(number_particles)

      for i in range(number_particles):
        particles_output[i] = function(particles[i])
    
        if particles_output[i] < particle_best_output[i]:
          particle_best_output[i] = particles_output[i]
          particle_best_input[i] = particles[i]
        if particles_output[i] < global_best_output:
          global_best_output = particles_output[i]
          global_best_input = np.copy(particles[i])
  
      inertia = 0.9 - (iteration/max_iteration) * 0.5

      for i in range(number_particles):
        r1 = random.uniform(0,1)
        r2 = random.uniform(0,1)
        velocity[i] = (inertia*velocity[i]) + (c1*r1*(particle_best_input[i]-particles[i])) + (c2*r2*(global_best_input-particles[i]))
        particles[i] = particles[i] + velocity[i]

      iteration = iteration + 1
      iteration_end = time.time()
      iteration_time = iteration_end - start
      time_axis.append(iteration_time)
      x_axis.append(iteration)
      y_axis.append(global_best_output)

    print('Input Value:', global_best_input)
    print('Output Value from Function:', global_best_output)
    end = time.time()
    total_time = end-start
    print('Total time:', total_time, 'seconds')
    function_counter = number_particles + 1 + (max_iteration * number_particles)
    print('Function Evaluations Counter:', function_counter)
    plot1.plot(x_axis,y_axis,'r')
    plot2.plot(time_axis,y_axis,'b')
    plot = plot + 1

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









