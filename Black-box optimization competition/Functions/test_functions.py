# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 21:12:19 2022

@author: dv516
"""

def f_d2(x):
    f = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2

    return f

def f_d3(x):
    
    f = (1-x[0])**2 + 100*(x[1]-x[0]**2)**2 + x[2]**2

    return f