# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:02:25 2022

@author: dv516
"""

from Functions.test_functions import f_d2, f_d3

# import sobol_seq
import numpy as np
from typing import List, Tuple

def gimme_it(f, N_x: int, bounds: List[Tuple[float]], N: int = 100) -> (float, List[float]):
    '''
    Optimizer aims to optimize a black-box function 'f' using the dimensionality 
    'N_x', and box-'bounds' on the decision vector 
    Input:
    f: function: taking as input a list of size N_x and outputing a float
    N_x: int: number of dimensions
    N: int: optional: Evaluation budget
    bounds: List of size N where each element i is a tuple conisting of 2 floats 
            (lower, upper) serving as box-bounds on the ith element of x
    Return:
    tuple: 1st element: lowest value found for f, f_min
           2nd element: list/array of size N_x giving the decision variables
                        associated with f_min
    '''
    if N_x != len(bounds): 
        raise ValueError('Nbr of variables N_x does not match length of bounds')

    ### Your code here
    # print('# --------------- Running Gimme It optimizer v0.4.2.0 --------------- #')
    # print('Noisy sobol sampling with space reduction')
    # print('Submitted late... sorry\n')
    # print('P.S. Need to think of better exit conditions though as right now it is ONLY dependent on space_reduction.')
    # print('So all problems will complete at the same number of iterations unless space_reduction is changed.\n')
    
    from scipy.stats import qmc
    import numpy.random as random

    # ---------------------------------------- Options ---------------------------------------- #
    # Default is to minimize (1), set direction to -1 to maximise
    direction = 1
    sobol_power = 7 # Size of Sobol sampling per "iteration", default is 2**7 = 128

    space_reduction = 70 # Reduce searching space to this amount in % of the previous space in each iteration
    space_tolerance = 1e-6 # if the best solution is within this difference to the previous iteration then we stop

    noise = True # adds noise based on current bounds on to the inputs if on
    noise_strength = 1 # max +- noise on inputs in % based on current bounds
    # -------------------------------------------------------------------------------- #

    ubd = np.array([max(i) for i in bounds]) # upper bounds
    lbd = np.array([min(i) for i in bounds]) # lower bounds
    current_x_space = lbd - ubd                      # space of parameters allowed based on the bounds
    cb = np.array([lbd, ubd])                # set initial current bounds

    history = {}
    bv_hist = []

    tolerance_space_flag = False
    for iteration in range(N):
        sampler = qmc.Sobol(d = N_x, scramble = False) # initialize the sampler
        inputs = qmc.scale(sampler.random_base2(m = sobol_power), cb[0], cb[1]) # get the inputs
        if noise:
            # Create noisy input
            noisy_inputs = []
            for i in range(inputs.shape[1]):
                noisy_inputs.append(inputs[:, i] + current_x_space[i]*noise_strength/100*(2*random.rand(inputs.shape[0]) - 1))
            noisy_inputs = np.array(noisy_inputs).T

            # Check if created inputs is within bounds, if not then clip
            for i, ninput in enumerate(noisy_inputs):
                if (ninput > cb[1]).any():
                    for k, input_x in enumerate(ninput):
                        if input_x > cb[1][k]:
                            noisy_inputs[i][k] = cb[1][k]
                if (ninput < cb[0]).any():
                    for k, input_x in enumerate(ninput):
                        if input_x < cb[0][k]:
                            noisy_inputs[i][k] = cb[0][k]
            inputs = noisy_inputs
        outputs = direction*f(inputs.T) # evaluate f to get the outputs
        best_sample = {'value': outputs.min(), 'index': outputs.argmin(), 'x': inputs[outputs.argmin()]}
        bx = best_sample['x']
        bv = best_sample['value']

        reduced_x_space = (cb[1] - cb[0])*space_reduction/100 # calculate the reduced space based on space_reduction
        nb = np.array([bx - reduced_x_space/2, bx + reduced_x_space/2]) # center new space around best solution

        # Check that new bounds is within original bounds, if not then shift the space while keeping the same size
        for i in range(N_x):
            if nb.T[i][0] < bounds[i][0]: # check lower bounds
                nb.T[i][1] += bounds[i][0] - nb.T[i][0] # shift upper bounds
                nb.T[i][0] = bounds[i][0]               # clip lower bounds
            if nb.T[i][1] > bounds[i][1]: # check upper bounds
                nb.T[i][0] += bounds[i][1] - nb.T[i][1] # shift upper bounds
                nb.T[i][1] = bounds[i][1]               # clip upper bounds
        cb = nb
        current_x_space = reduced_x_space

        history[iteration] = best_sample # collect history data
        bv_hist.append(bv)

        # Termination condition based on the space size
        if reduced_x_space.mean() <= space_tolerance:
            tolerance_space_flag = True
            break
    total_iterations = len(history.keys())
    total_function_evals = total_iterations*2**sobol_power
    # if tolerance_space_flag:
    #     print('# ----- Normal completion ----- #')
    #     print('Probably NOT the optimal value, but I hope its good enough! :)')
    #     print(f'Space tolerance set: {space_tolerance}')
    #     print(f'Exited with a mean space difference of: {reduced_x_space.mean()}\n')
    # else:
    #     print('# ----- Space did not converge ----- #')
    #     print('Oh no :(')
    #     print(f'Did not exit with the space tolerance set: {space_tolerance}')
    #     print(f'Exited with a mean space difference of: {reduced_x_space.mean()}')
    #     print(f'Try reducing the space_reduction parameter or increasing maximum iterations\n')

    # print(f'Total number of iterations: {total_iterations}')
    # print(f'Total function evaluations: {total_function_evals}')
    # print(f'Objective: {bv}')
    # print(f'x: {bx}')

    # Plot objective value per iteration
    # fig = plt.figure()
    # plt.plot(bv_hist)
    # plt.ylabel('Objective value')
    # plt.xlabel('Iteration')
    # plt.show()
    
    x = bx
    ###

    return f(x), x



N_x = 2
bounds = [(-2.0, 2.0) for i in range(N_x)] # problem with bounds
test1 = gimme_it(f_d2, N_x, bounds, 100)
N_x = 3
bounds = [(-2.0, 2.0) for i in range(N_x)]
test2 = gimme_it(f_d3, N_x, bounds, 100)

np.testing.assert_array_less(test1[0], 1e-3)
np.testing.assert_array_less(test2[0], 1e-3)





