# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 22:13:22 2022

@author: dipto
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import random

def Func(x):
    return x*x

w = 0.7
phi_1 = 1.2
phi_2 = 0.8
r1 = 0.4
r2 = 0.7

X = np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*8]) for _ in range(10)])
print(X)
V = np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*3]) for _ in range(10)]) 
print(V)
P_i = X
P_g = 100000.0
position =-1
for i in range(10):
    f = Func(X[i])
    if(f<P_g):
        P_g = f
        position = i


for t in range(1000):
    for i in range(10):
        V[i] = w*V[i] + phi_1*r1*(P_i[i]-X[i]) + phi_2*r2*(P_g-X[i])
    print(V)
    for i in range(10):
        X[i]+= V[i]
    print(X)
    for i in range(10):
        if Func(P_i[i])>Func(X[i]) :
            P_i[i] = X[i]
        if Func(X[i])<P_g:
            P_g = Func(X[i])
            position = i
print(P_g)
print(np.round(P_g))
            
    