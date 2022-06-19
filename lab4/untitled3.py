# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 23:33:26 2022

@author: dipto
"""

import numpy as np
import math
import random
import matplotlib.pyplot as plt

def Fx(X):
    return X**2

w = 0.7
Q1 = 1.2
Q2 = 0.8
r1= 0.4
r2 = 0.7

X = np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*8]) for _ in range(10)])
print(X)
V = np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*3]) for _ in range(10)]) 
P_i = X
P_g = 100000.0
position =-1
for i in range(10):
    f = Fx(X[i])
    if(f<P_g):
        P_g = f
        position = i

for t in range(5000):
    V = w*V + Q1*r1*(P_i-X) + Q2*r2*(P_g-X)
    X= X+V
    for i in range(10):
        if Fx(P_i[i])>Fx(X[i]) :
            P_i[i] = X[i]
        if Fx(X[i])<P_g:
            P_g = Fx(X[i])
            position = i
print(P_g)
print(np.round(P_g))