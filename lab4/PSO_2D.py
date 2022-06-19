# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 20:02:17 2022

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

def fx(x,y):
    return x**2+y**2

def dist(x1,y1,x2,y2):
    a = (x1-x2)**2+(y1-y2)**2
    a = np.sqrt(a)
    return a

w = 0.7

c1 = 1.2

c2 = 0.8

r1 = 0.4

r2 = 0.7



#x = np.zeros(10,np.float32)

px = np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*8]) for _ in range(10)])
py = np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*8]) for _ in range(10)])

#print(x)
    



#print(px)

#xc = np.array(100,np.float32)

vx =  np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*3]) for _ in range(10)])
vy =  np.array([np.array([(-1)** (bool(random.getrandbits(1))) * random.random()*3]) for _ in range(10)])

#print(pv)

gbx = 1
gby = 1

gb = 10000000.0

for i in range(10):
    f = fx(px[i],py[i])
    if f < gb:
        gbx = px[i]
        gby = py[i]





#print(gbp)
        
xb = px
yb = py





    
        
#print(xb)

#print(w*pv)
#
#print(pv)

#pos = -1

for x in range(8000):
   wvx = np.zeros(10,np.float32)
   wvy = np.zeros(10,np.float32)
   
   for i in range(10):
       wvx[i] = vx[i]*w
   for i in range(10):
       wvy[i] = vy[i]*w
   t1 = np.zeros(10,np.float32)
   for i in range(10):
       t1[i] = c1*r1*dist(xb[i],yb[i],px[i],py[i])
#   for i in range(10):
   t2 = np.zeros(10,np.float32)
   for i in range(10):
       t2[i] = c2*r2*dist(gbx,gby,px[i],py[i])
   for i in range(10):
       vx[i] = wvx[i]+t1[i]+t2[i]
       vy[i] = wvy[i]+t1[i]+t2[i]
   for i in range(10):
       px[i]+=vx[i]
   for i in range(10):
       py[i]+=vy[i]
   for i in range(10):
       f = fx(px[i],py[i])
       if f < fx(xb[i],yb[i]):
           xb[i] = px[i]
           yb[i] = py[i]
       if gb>f:
           gb=f
       
       
   
   
   
   

        
#print(pv)
            
print(gb)

print(np.round(gb))
    
    
    