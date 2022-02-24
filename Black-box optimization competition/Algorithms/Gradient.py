# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:02:25 2022

@author: dv516
"""

import numpy as np
import sobol_seq


from Functions.test_functions import f_d2, f_d3

######################################
# Forward finite differences 
######################################

def forward_finite_diff(f, x):
    Delta = np.sqrt(np.finfo(float).eps) #step-size is taken as the square root of the machine precision
    n     = np.shape(x)[0]
    x     = x.reshape((n,1))
    dX    = np.zeros((n,1))
    
    for j in range(n):
        x_d_f    = np.copy(x)
        x_d_f[j] = x_d_f[j] + Delta
        dX[j]    = (f(x_d_f) - f(x))/Delta

    return dX

#############################
# Line search function
#############################

def line_search_f(direction, x, f, lr):
    '''
    function that determines optimal step with linesearch
    Note: f and lr must be given
    '''
    old_f = f(x); new_f = old_f + 1.
    ls_i  = 0   ; lr_i  = 2.*lr
    while new_f>old_f and ls_i<8:
        lr_i  = lr_i/2.
        x_i   = x - lr_i*direction 
        new_f = f(x_i)
        ls_i += 1
    
    return x_i, ls_i


#############################
# Approximating Hessian
#############################

def Hk_f(x, x_past, grad_i, grad_i_past, Hk_past, Imatrix):
    '''
    function that approximates the Hessian
    '''
    sk  = x - x_past 
    yk  = grad_i - grad_i_past
    rho = 1./(yk.T@sk+1e-7)

    Hinv = (Imatrix-rho*sk@yk.T)@Hk_past@(Imatrix-rho*yk@sk.T) + rho*sk@sk.T
    
    return Hinv

#############################
# First step 
#############################

def BFGS_step1(f, x, x0, n, grad_f, Imatrix):
    '''
    function that computes the first step for BFGS, because there is no Hk_past
    in the first interation, for x_past or grad_i_past. For this a steepest descent
    step is taken.
    '''
    grad_i      = grad_f(f,x)
    x           = x - 1e-5*grad_i
    # past values
    x_past      = x0.reshape((n,1))
    grad_i_past = grad_i
    # new gradient
    grad_i      = grad_f(f,x)
    # sk, yk, rho
    sk          = x - x_past 
    yk          = grad_i - grad_i_past
    # rho         = 1./(yk.T@sk+1e-7)
    # initial guess for H0
    Hk_past     = ((yk.T@sk)/(yk.T@yk))*Imatrix
    
    return Hk_past, grad_i_past, x_past, grad_i

###################################
# multistart
###################################

def x0_startf(bounds, n_s, N_x):
    '''
    Give starting points
    array([[0.  , 2.  , 7.5 ],
       [0.5 , 1.  , 8.75]])
    '''
    bounds_l = np.array([[ bounds[n_ix][1]-bounds[n_ix][0] ] for n_ix in range(len(bounds))])
    sobol_l  = sobol_seq.i4_sobol_generate(N_x, n_s)
    lb_l     = np.array([[bounds[i][0] for i in range(len(bounds))]])
    x0_start = lb_l  + sobol_l*bounds_l.T
    
    return x0_start

###################################
# BFGS for 'global search'
###################################

def BFGS_gs(f, N_x, bounds, N=1e6, ns=5, grad_f=forward_finite_diff, lr=2., max_iter=1e4, grad_tol=1e-4):
    '''
    Optimization algorithm: BFGS for global search with linesearch.
    Function evaluations are counted as if the algorithm was smartly coded - which it is not :)
    '''

    ns=max(5,int(N_x*0.02))


    # evaluate starting points
    x0_candidates = x0_startf(bounds, ns, N_x)
    f_l = []; 
    for xii in range(ns):
        f_l.append(f(x0_candidates[xii]))
    f_eval      = ns
    best_point  = ['none',1e15]    

    # multi-starting point loop
    while len(f_l)>=1 and f_eval<=N:
        minindex      = np.argmin(f_l)
        x0            = x0_candidates[minindex]
        #pop
        x0_candidates = x0_candidates.tolist()
        f_l.pop(minindex); x0_candidates.pop(minindex)
        x0_candidates = np.asarray(x0_candidates)
    
        # initialize problem
        n       = np.shape(x0)[0]
        x       = np.copy(x0); x = x.reshape((n,1))
        iter_i  = 0
        Imatrix = np.identity(n)
               
        # first step: gradient descent
        # compute gradient   
        grad_i  = grad_f(f,x)
        f_eval += N_x
        Hk_past, grad_i_past, x_past, grad_i = BFGS_step1(f, x, x0, n, grad_f, Imatrix)
        f_eval                              += 1
        
        # optimization loop
        while np.sum(np.abs(grad_i)) > grad_tol and iter_i < max_iter:    

            # compute gradient   
            grad_i  = grad_f(f,x)
            f_eval += N_x
            # compute Hessian
            Hinv    = Hk_f(x, x_past, grad_i, grad_i_past, Hk_past, Imatrix)
            x_past  = x
            # step direction
            Df_i    = Hinv@grad_i 

            # No linearch
            #x_i = x - Df_i
            # line-search
            x_i, ls_i = line_search_f(Df_i, x, f, lr)
            f_eval   += ls_i

            # record past points and gradients
            grad_i_past = grad_i
            Hk_past     = Hinv

            x = x_i                
            iter_i += 1 

            if best_point[1] > f(x):
                best_point = [x, f(x)]
            # best_point  = ['none',1e15]
    
        
        # print('f_eval   ===== ',f_eval)
        # print('x      ===== ',x) 
        # print('grad_i ===== ',grad_i)
        # print('f(x) ===== ',f(x))
        # print('f(x) best ===== ',best_point[1],'\n')
    
        
        
    return best_point[1], x # changed this

# N_x = 2
# bounds = [(-2.0, 2.0) for i in range(N_x)]
# test1 = BFGS_gs(f_d2, N_x, bounds, N=10000)
# N_x = 3
# bounds = [(-2.0, 2.0) for i in range(N_x)]
# test2 = BFGS_gs(f_d3, N_x, bounds, N=10000)

# np.testing.assert_array_less(test1[0], 1e-3)
# np.testing.assert_array_less(test2[0], 1e-3)





