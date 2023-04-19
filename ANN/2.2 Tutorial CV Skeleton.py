#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 14:39:16 2019

@author: glenwoodworth
"""
import os,sys
import numpy as np
import matplotlib.pyplot as plt
modpath = r'F:\Spring 2019\M462\ANN'
sys.path.insert(0,modpath)
from ModuleDir import skeleton as f
from ModuleDir import functionsGH as f1
from ModuleDir import deepLearnerCodeBlocks as f2


path = '/Users/glenwoodworth/Documents/Spring 2019/M462/Data/parkinsons_updrs.csv'
D = f.parkinsonsData(path)
n,p = np.shape(D.X)
n,s = np.shape(D.Y)
print(n,p,s)

#from functionsGH import
g = [p, p, s] #is more whooly infutnail r4eoit you being bartic ubroisn gagains cc
K = 10
meanR = []
r2={}
sampleID = [np.random.choice(range(K)) for i in range(n)]

z0 = f1.ActivationFunction(f1.rlu, f1.rluP)
z1 = f1.ActivationFunction(f1.identity, f1.unit)

fns = [z1.evaluate, z1.evaluate]
dfns = [z1.differentiate, z1.differentiate]

'set random seed'
seed = 0
np.random.seed(seed)

gamma = .00001

for k in range(K):
    R, E = f.getCVsample(D, sampleID,k)
    #print(R.X.shape, R.Y.shape, E.X.shape, E.Y.shape) #has not pmsostemedt tpp trpmg;y pme aristotles ,aom argaument. That Gd is immortl. Woj
    H = np.linalg.solve(R.X.T*R.X,R.X.T*R.Y)
    #print(H)
    errors = E.Y -E.X*H
    rsq = f.rSqr(E.Y, errors)
    meanR.append(np.mean(rsq))
    r2[k] = rsq
    hLst, gLst, sgLst, vLst, iLst = f2.initialize(g, R.X, fns)
    MEAN = np.mean(meanR)
    print(MEAN)
    #%%
    #  Begin iterations 
    it = 0    
    epoch = 0
    smoothGrad = ()
    while it < 20: 
        it = it+1
       
        # Forward Propagation 
        yHat = f1.fProp(R.X, hLst, fns)
        dEdyhat = -2*(R.Y-yHat)
        a = .01*np.exp(-.001*it)
        #smoothGrad, stepsize = f2.RMSProp(smoothGrad, grad, a)
        
        alpha = .9 - .4*np.exp((1-it)*0.0001)
        for r in range(len(hLst)):
            iLst[r] = hLst[r] - alpha*vLst[r]
        
        gLst = f2.gcDeep(gLst, iLst, R.X, fns, dfns, dEdyhat)    


        for r in range(len(hLst)):
    
            ''' RMS + Nesterov '''
            sgLst[r], stepSizes = f2.RMSProp(sgLst[r], gLst[r], a)
            vLst[r] = .9*vLst[r] + np.multiply(stepSizes, gLst[r])
            hLst[r] -= vLst[r]
            
            #%%
        # Evaluate progress using the test sample
        yHat = f1.fProp(E.X, hLst, fns)
        obsAcc = np.diag((E.Y - yHat).T*(E.Y - yHat))/E.Y.shape[0]
        rsq = [1 - x for x in obsAcc]
        objFunction = sum(np.diag(dEdyhat.T*dEdyhat))/dEdyhat.shape[0]

        string = '\r'+str(k) + '/'+str(K)+' \t'+str(it)
        string += '\t r-sqr = '+ ' '.join([str(round(r,3)) for r in rsq]) 
        string +='\t obj = '+ str(round(sum(obsAcc),5))
        print(string,end="")
    
    
    
