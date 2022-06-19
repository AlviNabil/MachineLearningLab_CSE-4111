# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 19:43:40 2022

@author: dipto
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jun  7 18:49:01 2022

@author: Sunanda Das
"""

import numpy as np
import math
import random

def fx(x):
    return x**2

w = 0.7

c1 = 1.2

c2 = 0.8

r1 = 0.4

r2 = 0.7



#x = np.zeros(10,np.float32)

px = np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*8]) for _ in range(10)])

#print(x)
    



#print(px)

#xc = np.array(100,np.float32)

pv =  np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*3]) for _ in range(10)])

#print(pv)

gb = 100000.0

gbp = -1

for i in range(10):
    f = fx(px[i])
    if(f<gb):
        gb = f
        gbp = i

#print(gbp)
        
xb = px




    
        
#print(xb)\

#print(w*pv)
#
#print(pv)

pos = -1

for x in range(1000):
    wv = np.zeros(10,np.float32)
    for i in range(10):
        wv[i] = w*pv[i]
    t1 = np.zeros(10,np.float32)
    for i in range(10):
        t1[i] = c1*r1*(xb[i]-px[i])
#    t1 = xb-px
#    t1 = c1*r1*t1
    t2  = np.zeros(10,np.float32)
    
    for i in range(10):
        t2[i] = c2*r2*(gb-px[i])
    for i in range(10):
        pv[i] = wv[i]+t1[i]+t2[i]
    print(pv)
    for i in range(10):
        px[i]+=pv[i]
    print(px)
    for i in range(10):
        f1 = fx(px[i])
        f2 = fx(xb[i])
        if f1 < f2:
            xb[i] = px[i]
        if f1<gb:
            gb=f1
            pos = i
        
#print(pv)
            
print(gb)

print(np.round(gb))
    
    
    