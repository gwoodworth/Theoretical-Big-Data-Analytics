#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:53:38 2019

@author: glenwoodworth
"""

import matplotlib.pyplot as plt
import numpy as np
n = 10
p = 15
Llist=[]
MR = []
L=0
for k in range(100):
    L=L+.01
    X = np.matrix(np.random.uniform(0,1, (n,p)))
    Y = np.matrix(np.random.uniform(0,1, (n,1)))
    betaR = np.linalg.inv(X.T*X+L*np.eye(p))*(X.T*Y)
    r = np.mean(betaR)
    MR.append(r)
    Llist.append(L)
plt.figure()
plt.xlabel('Lambda')
plt.ylabel('BetaR')
plt.plot(Llist,MR)


