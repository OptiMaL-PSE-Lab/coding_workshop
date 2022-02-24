# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 20:02:25 2022

@author: dv516
"""

import numpy as np
import random
# import scipy.integrate as scp
import numpy.random as rnd
import time
# import matplotlib.pyplot as plt
# import numpy.random as rnd
# import copy
import numpy as np
from Functions.test_functions import f_d2, f_d3
# from typing import List, Tuple


class INDIVIDUAL():

    def __init__(self, Dim, Bound, VBound, func):
        ### Init the var
        self.var = list()
        for i in range(Dim):
            # self.var.append(Bound[0] + (Bound[1] - Bound[0]) * random.uniform(0, 1))
            self.var.append(random.uniform(Bound[i][0], Bound[i][1]))

        self.fit = 1E30
        self.std = 0
        self.localBestVar = self.var
        self.localBestFit = self.fit
        self.localBestStd = self.std
        self.rank = 0
        self.vector = list()
        self.mut_strength = list()
        for dimno in range(Dim):
            # self.vector.append(0)
            self.vector.append(random.uniform(VBound[0], VBound[1]))
            self.mut_strength.append(np.random.uniform(0, 1))

        self.func = func

    def Calculate(self):
        # self.fit = self.func(self.var)
        self.fit, self.std = self.func(self.var)

        if (self.fit < self.localBestFit):
            self.localBestVar = self.var
            self.localBestFit = self.fit
            self.localBestStd = self.std

    def CalRank(self, rank):
        self.rank = rank

    def Rank(self, selfAll): # what's the meaning of this rank?
        Elem = list()
        for indi in selfAll:
            Elem.append(indi.localBestFit)

        ElemSorted = sorted(Elem)

        for indi in selfAll:
            for no in range(len(ElemSorted)):
                if indi.localBestFit == ElemSorted[no]:
                    indi.rank = no
                    ElemSorted[no] = max(ElemSorted) + 1

                    break

def solveSA(PO, GBest, funcName, iterMax, dim, interval, T_Bound, T, rate): # what is PO?
    ####x1 = np.array(self.x_seed)  # Initial x
    ###T = self.T_max
    
    countSA = 0
    for indi in PO:
        x1 = np.array(GBest.var)
        
        for i in range(iterMax):
            y, xstd = funcName(x1)
            delta_x = []
            for idex in range(dim):
                delta_x.append(np.random.uniform(-1, 1))
            xNew = getNewList(x1, delta_x, interval)

            yNew, xNewstd = funcName(xNew)
            countSA += 1
            delta_y = yNew - y

            if delta_y < 0:
                indi.var = xNew
                indi.fit = yNew
                indi.std = xNewstd
            else:
                P = p_min(delta_y, T)
                if P > np.random.uniform(0, 1):
                    indi.var = xNew
                    indi.fit = yNew
                    indi.std = xNewstd
                else:
                    indi.var = x1
                    indi.fit = y
                    indi.std = xstd
                    
            # x1 = deal_min(x1, xNew, delta_y, T)
            # indi.var = x1
            # indi.fit = funcName(indi.var)
            
            ###更新个体
            if indi.fit < indi.localBestFit:
                indi.localBestFit = indi.fit
                indi.localBestVar = indi.var
                indi.localBestStd = indi.std
                if indi.fit < GBest.fit:
                    GBest.fit = indi.fit
                    GBest.var = indi.var
                    GBest.std = indi.std
        T *= rate

        ###return x_best, funcName(x_best), value, time1, evalSA
    return T, PO, GBest, countSA

def getNewList(list1, list2, bounds): # what's this for?
    new_list = []
    for idx in range(len(list1)):
        if list1[idx] + list2[idx] >= bounds[idx][1]:
            new_list.append(bounds[idx][1])
        elif list1[idx] + list2[idx] <= bounds[idx][0]:
            new_list.append(bounds[idx][0])
        else:
            new_list.append(list1[idx] + list2[idx])
    return np.array(new_list)

def p_min(delta, T):
    probability = np.exp(-delta / T)
    return probability


"""
PSO
"""
def solvePSO(PO, GBest, funcName, Bound, iterMax, dim, c1, c2, VBound):
    nPop = len(PO)
    PopPosition = np.zeros([nPop, dim])
    PopCost = np.zeros([nPop, 1])
    Popstd = np.zeros([nPop, 1])
    Popvel = np.zeros([nPop, dim])
    for no in range(nPop):
        PopPosition[no][:] = PO[no].var
        PopCost[no][0] = PO[no].fit
        Popstd[no][0] = PO[no].std
        Popvel[no][:] = PO[no].vector
        # PopPosition[no][:] = PO[no].localBestVar
        # PopCost[no][0] = PO[no].localBestFit
        # Popstd[no][0] = PO[no].localBestStd
        # Popvel[no][:] = PO[no].vector    
    c3 = c1 + c2
    k = 2 / (abs(2 - c3 - np.sqrt((c3 ** 2) - (4 * c3))))  # creating velocity weighting factor
    iter = 1
    countPSO =0
    while iter <= iterMax:
        # for indi in PO:
        for no in range(len(PO)):    

            for i in range(dim):
                
                # PO[no].vector[i] = k * (PO[no].vector[i] + c1 * np.random.uniform(0, 1) * (PO[no].localBestVar[i] - PO[no].var[i]) + \
                #     c2 * np.random.uniform(0, 1) * (GBest.var[i] - PO[no].var[i]))
                Popvel[no][i] = k * (Popvel[no][i] + c1 * np.random.uniform(0, 1) * (PO[no].localBestVar[i] - PopPosition[no][i]) + \
                    c2 * np.random.uniform(0, 1) * (GBest.var[i] - PopPosition[no][i]))

                # if (PO[no].vector[i] > VBound[1]):
                #     PO[no].vector[i] = VBound[1]
                # if (PO[no].vector[i] < VBound[0]):
                #     PO[no].vector[i] = VBound[0]
                if (Popvel[no][i] > VBound[1]): # how to determine velocity bound? why we need to update v?
                    Popvel[no][i] = VBound[1]
                if (Popvel[no][i] < VBound[0]):
                    Popvel[no][i] = VBound[0]

                # PO[no].var[i] = PO[no].var[i] + PO[no].vector[i]
                PopPosition[no][i] = PopPosition[no][i] + Popvel[no][i]


                # if PO[no].var[i] > Bound[i][1]:
                #     PO[no].var[i] = Bound[i][1]
                # if PO[no].var[i] < Bound[i][0]:
                #     PO[no].var[i] = Bound[i][0]
                if PopPosition[no][i] > Bound[i][1]:
                    PopPosition[no][i] = Bound[i][1]
                if PopPosition[no][i] < Bound[i][0]:
                    PopPosition[no][i] = Bound[i][0]

            # indi.Calculate()
            # PO[no].fit, PO[no].std = funcName(PO[no].var)
            countPSO += 1 
            PopCost[no][0], Popstd[no][0] = funcName(PopPosition[no])
            
            PO[no].var = PopPosition[no][:]
            PO[no].fit = PopCost[no][0]
            PO[no].std = Popstd[no][0]
            PO[no].vector = Popvel[no][:]
            if PO[no].fit < PO[no].localBestFit:
                PO[no].localBestFit = PO[no].fit
                PO[no].localBestVar = PO[no].var
                PO[no].localBestStd = PO[no].std
                if PO[no].localBestFit < GBest.fit:
                    GBest.fit = PO[no].localBestFit
                    GBest.var = PO[no].localBestVar
                    GBest.std = PO[no].localBestStd
            # if PO[no].fit < PO[no].localBestFit:
            #     PO[no].localBestFit = PO[no].fit
            #     PO[no].localBestVar = PO[no].var
            #     PO[no].localBestStd = PO[no].std
            #     if PO[no].fit < GBest.fit:
            #         GBest.fit = PO[no].fit
            #         GBest.var = PO[no].var
            #         GBest.std = PO[no].std
    
        iter += 1

    return PO, GBest ,countPSO

"""
ABC
"""
def solveABC(PO, GBest, funcName, Bound, iterMax, dim, a, nOnLooker, L):
    nPop = len(PO)

    PopPosition = np.zeros([nPop, dim])
    PopCost = np.zeros([nPop, 1])
    Popstd = np.zeros([nPop, 1])
    for no in range(nPop):
        PopPosition[no][:] = PO[no].localBestVar
        PopCost[no][0] = PO[no].localBestFit
        Popstd[no][0] = PO[no].localBestStd

    Probability = np.zeros([nPop, 1])
    BestSol = GBest.var
    BestCost = GBest.fit
    BestStd = GBest.std
    Mine = np.zeros([nPop, 1])

    # start = time.perf_counter()
    # time1 = [0]
    # eval = [0]
    iter = 1
    countABC = 0
    while iter <= iterMax:
        

        # employed bees

        # Find the next source of honey
        for i in range(nPop):
            while True:
                k = np.random.randint(0, nPop)
                if k != i:
                    break
            phi = a * (-1 + 2 * np.random.rand(dim))
            NewPosition = PopPosition[i] + phi * (PopPosition[i] - PopPosition[k])
            # NewPosition[:] = np.clip(NewPosition, *Bound)
            for idx in range(dim):
                if NewPosition[idx] > Bound[idx][1]:
                    NewPosition[idx] = Bound[idx][1]
                if NewPosition[idx] < Bound[idx][0]:
                    NewPosition[idx] = Bound[idx][0]

            # Make greedy choices
            NewCost, NewStd = funcName(NewPosition)
            countABC += 1
            if NewCost < PopCost[i][0]:
                PopPosition[i] = NewPosition
                PopCost[i][0] = NewCost
                Popstd[i][0] = NewStd
            else:
                Mine[i][0] = Mine[i][0] + 1

        # Follower bees

        # 计算选择概率矩阵
        Mean = np.mean(PopCost)
        for i in range(nPop):
            Probability[i][0] = np.exp(-PopCost[i][0] / Mean)
        Probability = Probability / np.sum(Probability)
        CumProb = np.cumsum(Probability)

        for k in range(nOnLooker):

            # Implementation of Roulette Selection Method
            m = 0
            for i in range(nPop):
                m = m + CumProb[i]
                if m >= np.random.rand(1):
                    break

            # Repeated employed bee operations
            while True:
                k = np.random.randint(0, nPop)
                if k != i:
                    break
            phi = a * (-1 + 2 * np.random.rand(dim))
            NewPosition = PopPosition[i] + phi * (PopPosition[i] - PopPosition[k])
            # NewPosition[:] = np.clip(NewPosition, *Bound)
            for idx in range(dim):
                if NewPosition[idx] > Bound[idx][1]:
                    NewPosition[idx] = Bound[idx][1]
                if NewPosition[idx] < Bound[idx][0]:
                    NewPosition[idx] = Bound[idx][0]

            # Make greedy choices
            NewCost, NewStd = funcName(NewPosition)
            countABC += 1
            if NewCost < PopCost[i][0]:
                PopPosition[i] = NewPosition
                PopCost[i][0] = NewCost
                Popstd[i][0] = NewStd
            else:
                Mine[i][0] = Mine[i][0] + 1

        # Scouter bees
        for i in range(nPop):
            if Mine[i][0] >= L:
                PopPosition[i] = getMatrix(Bound, dim)
                PopCost[i][0], Popstd[i][0] = funcName(PopPosition[i])
                countABC += 1
                Mine[i][0] = 0

        # Save the historical best solution
        for i in range(nPop):
            if PopCost[i][0] < BestCost:
                BestCost = PopCost[i][0]
                BestSol = PopPosition[i]
                BestStd = Popstd[i][0]

        # save best solution
        # end = time.perf_counter()
        # time1.append(end - start)
        # eval.append(count)

        iter += 1

    for no in range(nPop):
        PO[no].var = PopPosition[no][:]
        PO[no].fit = PopCost[no][0]
        PO[no].std = Popstd[no][0]
        if PO[no].fit < PO[no].localBestFit:
            PO[no].localBestFit = PO[no].fit
            PO[no].localBestVar = PO[no].var
            PO[no].localBestStd = PO[no].std
            if PO[no].localBestFit < GBest.fit:
                GBest.fit = PO[no].localBestFit
                GBest.var = PO[no].localBestVar
                GBest.std = PO[no].localBestStd

    ###evaluation = [sum(eval[:x]) for x in range(1, len(eval) + 1)]
    ###return BestSol[-1], BestCost[-1], BestCost, time1, evaluation
    return PO, GBest , countABC

def getMatrix(Bound, Dim):

    result = []
    for i in range(Dim):    
        result.append(random.uniform(Bound[i][0], Bound[i][1]))
    # for i in range(n):
    #     item_list = []
    #     for j in range(m):
    #         item_list.append(random.uniform(low, up))
    #     result.append(item_list)

    return result  # if require list then change into "return result"


def solveES(PO, GBest, funcName, Bound, iterMax, dim, N_KID):
    ###def solve(pop):

    nPop = len(PO)

    var = np.zeros([nPop, dim])
    mut_strength = np.zeros([nPop, dim])
    fitness = np.zeros([nPop,1])
    std = np.zeros([nPop,1])
    for no in range(nPop):
        var[no][:] = PO[no].localBestVar
        mut_strength[no][:] = PO[no].mut_strength
        fitness[no] = PO[no].localBestFit
        std[no] = PO[no].localBestStd
    """
    pop = dict(DNA=np.random.uniform(DNA_BOUND[0], DNA_BOUND[1], size=(1, DNA_SIZE)).repeat(POP_SIZE, axis=0),
               # initialize the pop DNA values
               mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))
    """

    pop = dict(DNA = var, mut_strength = mut_strength, fitness = fitness, std = std)


    # value = [funcName(pop['DNA'])[0]]
    # start = time.perf_counter()
    # time1 = [0]
    # evalES = [0]
    countES = 0
    iter = 1
    ###for _ in range(N_GENERATIONS):
    while iter <= iterMax:
        iter += 1
        # countES += 1
        # ES part
        kids = make_kid(pop, N_KID, dim, Bound,nPop)
        pop = kill_bad(pop, kids, funcName, nPop)  # keep some good parent for elitism

        # save best solution
        # end = time.perf_counter()
        # time1.append(end - start)
        # value.append(funcName(pop['DNA'])[0])
        # evalES.append(countES * (nPop + N_KID))

    # objSelCh = list()
    # for no in range(nPop):
    #     objSelCh.append(funcName(pop['DNA'][no]))

    for no in range(nPop):
        PO[no].var = pop['DNA'][no]
        # PO[no].fit = objSelCh[no]
        PO[no].fit = pop['fitness'][no][0]
        PO[no].mut_strength = pop['mut_strength'][no]
        PO[no].std = pop['std'][no][0]
        if PO[no].fit < PO[no].localBestFit:
            PO[no].localBestFit = PO[no].fit
            PO[no].localBestVar = PO[no].var
            PO[no].localBestStd = PO[no].std
            if PO[no].localBestFit < GBest.fit:
                GBest.fit = PO[no].localBestFit
                GBest.var = PO[no].localBestVar
                GBest.std = PO[no].localBestStd
    countES = iterMax*(N_KID)
    return PO, GBest , countES


def make_kid(pop, n_kid, dim, Bound,nPop):
    # generate empty kid holder
    kids = {'DNA': np.empty((n_kid, dim)), 'fitness': np.empty((n_kid, 1)), 'std': np.empty((n_kid, 1))}
    kids['mut_strength'] = np.empty_like(kids['DNA'])
    for kv, ks in zip(kids['DNA'], kids['mut_strength']):
        # crossover (roughly half p1 and half p2)
        p1, p2 = np.random.choice(np.arange(nPop), size=2, replace=False)
        cp = np.random.randint(0, 2, dim, dtype=np.bool_)  # crossover points
        kv[cp] = pop['DNA'][p1, cp]
        kv[~cp] = pop['DNA'][p2, ~cp]
        ks[cp] = pop['mut_strength'][p1, cp]
        ks[~cp] = pop['mut_strength'][p2, ~cp]

        # mutate (change DNA based on normal distribution)
        ks[:] = np.maximum(ks + (np.random.rand(*ks.shape) - 0.5), 0.)  # must > 0
        kv += ks * np.random.randn(*kv.shape)
        # kv[:] = np.clip(kv, *Bound)  # clip the mutated value
        for idx in range(dim):
            kv[idx] = np.clip(kv[idx], Bound[idx][0],Bound[idx][1])
    return kids


def kill_bad(pop, kids, funcName, Popsize):
    # put pop and kids together
    for key in ['DNA', 'mut_strength', 'fitness','std']:
        pop[key] = np.vstack((pop[key], kids[key]))

    for i in range(len(pop['DNA'])):
        pop['fitness'][i], pop['std'][i] = funcName(pop['DNA'][i])
    
    # calculate global fitness
    # fitness = get_fitness(F(pop['DNA'], funcName))
    fitness = get_fitness(pop['fitness'])

    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][:Popsize]  # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength','fitness','std']:
        pop[key] = pop[key][good_idx]
    return pop


def get_fitness(pred): return pred.flatten()

def optimizer_dummy(f, N_x, bound, N=100):

#     if N_x != len(bound):
#         raise ValueError('Nbr of variables N_x does not match length of bounds')
        
    
  ### Your code here
#     x = [np.mean(bounds[i]) for i in range(N_x)]
    dimensions= N_x
    dimension_bounds= bound   # bounds are important here
    bounds=[0]*dimensions     #creating n_d dimensional bounds
    for i in range(dimensions):
        bounds[i]=dimension_bounds

    PopSize = int(N/10) # changed these
    MaxIter = N-PopSize # changed these

    numRerank = 5  #How many iterations to reclassify 
    numRebound = 20 #How many iterations to reduce the search space
    delta = 0.35 #Determines the size of the reduced search space  

    """
    Parameters of Simulated annealing
    """
    T_Bound = [1, 200]
    T = T_Bound[1]
    rate =  0.9
    """
    Parameters of PSO
    """
    VBound = [-(dimension_bounds[1]-dimension_bounds[0])*0.75, (dimension_bounds[1]-dimension_bounds[0])*0.75]
    c1=2.8
    c2=1.3

    """
    Parameters of ABC
    """
    nOnLooker = 10
    L = np.around(0.6 * dimensions * PopSize)
    a = 1
    """
    Parameters of ES
    """
    N_KID = 40


    """Initialize the population"""
    GBest = INDIVIDUAL(dimensions, bounds, VBound,f)
    PO = list()
    for i in range(PopSize):
        PO.append(INDIVIDUAL(dimensions, bounds, VBound,f))
        PO[i].Calculate()
        if (PO[i].fit < GBest.fit):
            GBest.var = PO[i].var
            GBest.fit = PO[i].fit
            GBest.std = PO[i].std

    PO[0].Rank(PO)


    """iteration starts"""

    iter = 0
    meanhybrid = [GBest.fit]
    stdhybrid = [GBest.std]
    evalhybrid = [0]
    count = 0
    start = time.perf_counter()
    timehybrid = [0]
    while iter <= MaxIter:
    #--Here you can adjust the population size of each algorithm and try different combinations--#
        if iter % numRerank == 0:
            PO0 = list()
            # PO1 = list()
            # PO2 = list()
            PO3 = list()
            for no in range(len(PO)):
                if PO[no].rank <= 19:
                    PO0.append(PO[no])
                # if (PO[no].rank > 20) and (PO[no].rank <= 21):
                #     PO1.append(PO[no])
                # if (PO[no].rank > 21) and (PO[no].rank <= 30):
                #     PO2.append(PO[no])
                if (PO[no].rank > 19) and (PO[no].rank <= 80):
                    PO3.append(PO[no])

        # T, PO1, GBest , countSA = bAlg.solveSA(PO1, GBest, Rosenbrock, 1, dimensions, bounds, T_Bound, T, rate)
        # print('iter',iter,GBest.fit)
        # PO2, GBest ,countABC= bAlg.solveABC(PO2, GBest, Rosenbrock, bounds, 1, dimensions, a, nOnLooker, L)
        # print('iter',iter,GBest.fit)
        PO0, GBest ,countES= solveES(PO0, GBest,f, bounds, 1, dimensions, N_KID)
        # print('iter',iter,GBest.fit)
        PO3, GBest, countPSO = solvePSO(PO3, GBest,f, bounds, 1, dimensions, c1, c2,  VBound)
        # print('iter',iter,GBest.fit)
        if iter % numRerank == numRerank-1:  #?
            # PO = PO0 + PO1 + PO2 + PO3
            PO = PO0+ PO3
            PO[0].Rank(PO)

        if iter % numRebound == numRebound-1:
            for i in range(dimensions):
                position = list()
                lower = 0
                upper = 0
                for j in range(PopSize):
                    position.append(PO[j].localBestVar[i]) 
                # lower =  min(position) + (min(position) - GBest.var[i]) * delta
                lower =  min(position) + abs((min(position) - GBest.var[i])) * delta
                upper =  max(position) - (max(position) - GBest.var[i]) * delta
                bounds[i] = [lower,upper]


        end = time.perf_counter()
        timehybrid.append(end - start)
        # count += countSA + countPSO + countABC +countES
        count +=  countPSO  +countES
        evalhybrid.append(count)
        meanhybrid.append(GBest.fit)
        stdhybrid.append(GBest.std)
        iter += 1

    # print("Best solution: ", GBest.var)
    # print("Best value: ", GBest.fit)

    ################
    # plt.figure()
    # plt.plot(evalhybrid , meanhybrid)
    # plt.gca().fill_between(evalhybrid, np.array(meanhybrid) - np.array(stdhybrid), 
    #                        np.array(meanhybrid) + np.array(stdhybrid), 
    #                        color='C0', alpha=0.2)
    # plt.xlabel('number of function evaluation')
    # plt.ylabel('Objective function value')
    # plt.show()


  ###

    return GBest.fit,GBest.var


N_x = 2
bounds = [(-2.0, 2.0) for i in range(N_x)] # problem with bounds
test1 = optimizer_dummy(f_d2, N_x, bounds, 100)
N_x = 3
bounds = [(-2.0, 2.0) for i in range(N_x)]
test2 = optimizer_dummy(f_d3, N_x, bounds, 100)

np.testing.assert_array_less(test1[0], 1e-3)
np.testing.assert_array_less(test2[0], 1e-3)


