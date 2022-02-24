# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:02:25 2022

@author: dv516
"""

from sklearn.gaussian_process import GaussianProcessRegressor
from typing import List, Tuple
import random
import numpy as np
from scipy.stats import norm
from Functions.test_functions import f_d2, f_d3


def find_max(exp_improvements, x_samples):
    # find x that leads to best exp improvement
    max_exp_improvement = -1E10
    index_max_exp_improvement = 0
    for index, exp_improvement in enumerate(exp_improvements):
        if exp_improvement > max_exp_improvement:
            max_exp_improvement = exp_improvement
            index_max_exp_improvement = index
    best_x = x_samples[index_max_exp_improvement]
    return best_x


def acquision_function(surrogate_model, bounds, N_x):
    """
    estimate the likelihood of a sample being worth evaluating

    mean_y_pred: prediction of the new y
    sigma_y_pred: standard deviation of the prediction of the new y
    max_y_pred: max of the predictions of the new y

    return: expected improvement
    """

    # print("Generate random points for acquisition function")
    X = generate_random_points(bounds, N_x)

    exp_improvements = []

    for x in X:
        # print("Predicting y for x {} in surrogate".format(x))
        mean_y_pred, sigma_y_pred = surrogate_model.predict(np.array(x).reshape(1, -1), return_std=True)
        max_y_pred = np.max(mean_y_pred)

        # print("Computing z for surrogate prediction x{}".format(x))
        z = (mean_y_pred - max_y_pred) / sigma_y_pred

        if sigma_y_pred <= 0:
            # print("sigma of predicted y is {}, which is below 0".format(sigma_y_pred))
            exp_improvement = 0
        else:
            # print("sigma of predicted y is {}, which is superior than 0".format(sigma_y_pred))
            exp_improvement = (mean_y_pred - max_y_pred) * norm.cdf(z) - sigma_y_pred * norm.pdf(z)

        # print("Appending expected improvement of {}".format(exp_improvement))
        exp_improvements.append(exp_improvement)

    best_x = find_max(exp_improvements, X)
    # print("Returning best x {}".format(best_x))
    return best_x


def surrogate_function(x, y):
    """
    fit a gaussian process model to test a range of candidate samples
    """
    surrogate_model = GaussianProcessRegressor()
    # print("Starting to fit the model")
    surrogate_model.fit(np.array(x), np.array(y))
    return surrogate_model


def sample_gen_points(X, f):
    ys = []
    for x in X:
        y = f(x)
        ys.append(y)
    return ys


def generate_random_points(bounds, N_x, num_points_to_gen=100) -> (List[float], List[float]):
    xs = []
    for _ in range(num_points_to_gen):
        x = []
        for n in range(N_x):
            ### changed this line
            x.append(np.random.uniform(bounds[n][0], bounds[n][1]))
            ###
        xs.append(x)
    return xs


def bayesian_optimiser(f, N_x: int, bounds: List[Tuple[float, float]], N: int = 100) -> (float, List[float]):
    '''
    Optimiser aims to optimize a black-box function 'f' using the dimensionality
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

    X = generate_random_points(bounds, N_x)
    Y = sample_gen_points(X, f)

    surrogate_model = surrogate_function(X, Y)

    #best_result = [-1E30,-1E30]
    #best_y = -1E30
    #best_y_index = 0

    for n in range(N):
        x_best_surrogate = acquision_function(surrogate_model, bounds, N_x)
        y_best_surrogate = f(x_best_surrogate)

        # print("Appending best x {}:".format(x_best_surrogate))
        X.append(x_best_surrogate)
        Y.append(y_best_surrogate)

        # print("Updating surrogate model:")
        surrogate_model = surrogate_function(X, Y)

        # compute max of surrogate
        # if max == previous: exit
    # print("Computing the max of sample points:")
    
    ## changed this
    index_max = np.argmax(Y)
    ##
    best_y = Y[index_max]
    best_x = X[index_max]

    return best_y, best_x


f_d2max = lambda x: -f_d2(x)
f_d3max = lambda x: -f_d3(x)

# N_x = 2
# bounds = [(-2.0, 2.0) for i in range(N_x)]
# test1 = bayesian_optimiser(f_d2max, N_x, bounds, N=100)
# N_x = 3
# bounds = [(-2.0, 2.0) for i in range(N_x)]
# test2 = bayesian_optimiser(f_d3max, N_x, bounds, N=100)


# np.testing.assert_array_less(-test1[0], 1e-3)
# np.testing.assert_array_less(-test2[0], 1e-3)









