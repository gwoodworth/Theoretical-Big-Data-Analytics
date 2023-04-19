#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 15:03:36 2019

@author: glenwoodworth
"""

import numpy as np
modpath = r'F:\Spring 2019\M462\ANN'
from ModuleDir import skeleton as f
from ModuleDir import functionsGH as f1
from ModuleDir import deepLearnerCodeBlocks as f2
from collections import namedtuple
dataSet = namedtuple('data','X Y labels meanY stdY')
dataTuple = namedtuple('variables','X Y mean stdDev')   

path = '/Users/glenwoodworth/Documents/Spring 2019/M462/Data/parkinsons_updrs.csv'
D = f.parkinsonsData(path)
n,p = np.shape(D.X)
n,s = np.shape(D.Y)
print(n,p,s)
def getSample(D, sampleID, k):
    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i]==k]
    rIndex = [i for i in range(n) if sampleID[i]!=k]
    
    partition = namedtuple('data','R E')
    data = namedtuple('data', 'X Y labels')
    
    if D.labels is None:
        split = partition(data(D.X[rIndex,:], D.Y[rIndex,:], None),
                          data(D.X[sIndex,:], D.Y[sIndex,:], None))
    else:
        split = partition(data(D.X[rIndex,:], D.Y[rIndex,:],
                               [D.labels[i] for i in rIndex]),
                          data(D.X[sIndex,:], D.Y[sIndex,:],
                               [D.labels[i] for i in sIndex]))
    return split

def fProp(A, hLst, fns, dfns, zpLst):
    ''' Forward propagation assuming one coefficient matrix '''
    n = len(A)
    X0 = np.ones((n,1))
    AH = A * hLst[0]
    X1 = fns[0](AH)
    A1 = np.hstack((X0,X1))
    X2 = fns[1](A1*hLst[1])
    A2 = np.hstack((X0,X2))
    yHat.append(fns[2](A2*hLst[2]))
    zpLst.append(dfns[0](AH))
    zpLst.append(dfns[1](A1*hLst[1]))
    zpLst.append(dfns[2](A2*hLst[2]))
    xLst.append(AH)
    xLst.append(X1)
    xLst.append(X2)
    return xLst, zpLst, yHat


g = [p, p, p, p, p, s] 
K = 10
meanR = []
r2={}
sampleID = [np.random.choice(range(K)) for i in range(n)]
#%%
z0 = f1.ActivationFunction(f1.rlu, f1.rluP)
z1 = f1.ActivationFunction(f1.tanh, f1.tanhP)
z2 = f1.ActivationFunction(f1.identity, f1.unit)


fns = [z0.evaluate, z1.evaluate,z2.evaluate]
dfns = [z0.differentiate,z1.differentiate, z2.differentiate]
#%%
L1 = .005
L2 = .005

'set random seed'
seed = 0
np.random.seed(seed)

batSize = 300
batProgress = 0

for k in range(K):
    #R, E = f.getCVsample(D, sampleID, k)
    C = getSample(D, sampleID,k)
    n = np.shape(C.R.Y)[0] #cross -validation training set size
    nBatSamples = int(n/batSize) # number of batch samples
    batchID = [np.random.choice(range(nBatSamples)) for i in range(n)]
    B = getSample(C.R, batchID, 0) #draw the first batch sample B.E
    batSize = np.shape(B.E.X)[0]
    batprogress = 0
    nProcessed = 0
    #print(R.X.shape, R.Y.shape, E.X.shape, E.Y.shape) #has not pmsostemedt tpp trpmg;y pme aristotles ,aom argaument. That Gd is immortl. Woj
    
   
    #H = np.linalg.solve(R.X.T*R.X,R.X.T*R.Y)
    #print(H)
    #assign each CV training observation a batch sample
   
   
  
    initialLst = f2.initialize(g,B.E.X, fns, dfns)
    yHat, xLst, hLst, gLst, zpLst, vLst, iLst, sgLst = initialLst
#    errors = E.Y -E.X*H
#    rsq = f.rSqr(E.Y, errors)
#    meanR.append(np.mean(rsq))
#    r2[k] = rsq
#    hLst, gLst, sgLst, vLst, iLst = f2.initialize(g, R.X, fns)
#    MEAN = np.mean(meanR)
#    print(MEAN)
    #%%
    #  Begin iterations 
    it = 0    
    epoch = 0
    smoothGrad = ()
    for it in range(5000): 
        epoch = int(nProcessed/n)
        xLst, zpLst, yHat = fProp(xLst, hLst, fns, dfns, zpLst)

        dEdyhat = -2*(B.R.Y[:,1]*yHat)
        a = .01*np.exp(-.001*it)
        
        alpha = .9 - .4*np.exp((1-it)*0.0001)
        for r in range(len(hLst)):
            iLst[r] = hLst[r] - alpha*vLst[r]
        
        gLst, penalty = f2.gcDeepReg(gLst, iLst, B.R.X, fns, dfns, dEdyhat, L1, L2)    


        for r in range(len(hLst)):
    
            ''' RMS + Nesterov '''
            sgLst[r], stepSizes = f2.RMSProp(sgLst[r], gLst[r], a)
            vLst[r] = .9*vLst[r] + np.multiply(stepSizes, gLst[r])
            hLst[r] -= vLst[r]
#%%
            #Determine number of observations used to date
        batProgress += np.dim(B.E.X)[0]
        nProcessed += np.dim(B.E.X)[0]
        #Test epoch is complete
        #If so, create a new batch sample index
        if batProgress >= n:
            #shuffle the batch sample indexes in place
            np.random.shuffle(batchID)
            batProgress = 0 #reset the counter
        #determine the batch sample for the next iteration and get it
        label = (it+1)%nBatSamples
        B = f.getSample(C.R, batchID, label)
        xLst[0] = B.E.X  
        #%%
        # Evaluate progress using the test sample
        #yHat = f1.fProp(E.X, hLst, fns)
        obsAcc = np.diag((C.E - yHat).T*(C.E - yHat))/C.E.shape[0]
        rsq = [1 - x for x in obsAcc]
        objFunction = sum(np.diag(dEdyhat.T*dEdyhat))/dEdyhat.shape[0] + penalty

        string = '\r'+str(k) + '\t' +str(K)+ '\t'+str(it)
        string += '\t r-sqr = '+ ' '.join([str(round(r,3)) for r in rsq]) 
        string +='\t obj = '+ str(round(sum(obsAcc),5))
        print(string,end="")
    
