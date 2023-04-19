#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 06:52:08 2019

@author: glenwoodworth
"""


import numpy as np
from collections import namedtuple
dataSet = namedtuple('data','X Y labels meanY stdY')
dataTuple = namedtuple('variables','X Y mean stdDev')   

modpath = r'F:\Spring 2019\M462\ANN'
from ModuleDir import skeleton as f
from ModuleDir import functionsGH as f1
from ModuleDir import deepLearnerCodeBlocksOriginal as f2

def parkinsonsData(path):
    
    dataSet = namedtuple('data','X Y meanY stdY labels')
    records =  open(path,'r').read().split('\n')
    variables = records[0].split(',')
    
    iX = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    iY = [4, 5]
    
    print('\nPredictor variables:')    
    for i in range(len(iX)) : 
        print(iX[i], variables[iX[i]])
    print('\nTarget variables:')    
    for i in range(len(iY)) : 
        print(iY[i], variables[iY[i]])
        
    n = len(records)-1
    p = len(iX) 
    
    try:
        s = len(iY)
    except(TypeError):
        s = 1
    
    Y = np.matrix(np.zeros(shape = (n, s)))
    X = np.matrix(np.ones(shape = (n, p )))
    for i, j in enumerate(np.arange(1,n+1,1)):
        lst = records[j].split(',')
        for k in range(s):
            Y[i,k] = float(lst[iY[k]])
        for k in range(p):
            X[i,k] = lst[iX[k]]    
    
    s = np.std(Y, axis=0)            
    m = np.mean(Y, axis = 0)    
    
    ''' Normalize the targets and predictors '''
    Y = (Y - m)/s
    
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)            

    data = dataSet(X, Y, m, s, None)
    return data

path = '/Users/glenwoodworth/Documents/Spring 2019/M462/Data/parkinsons_updrs.csv'
D = parkinsonsData(path)
n,p = np.shape(D.X)
n,s = np.shape(D.Y)
print(n,p,s)

def rlu(x):
    return max([0,x])    
def rluP(x):
    return int(x > 0) 

def identity(X):
    return X

def unit(X):
    return 1
def tanh(x):
    return np.tanh(x)
def tanhP(x):
    return 1-np.tanh(x)**2

class ActivationFunction(object):
    def __init__(self, function, derivative):    
        self.function = function
        self.derivative = derivative
            
    def differentiate(self, X):
        shape = np.shape(X)
        A = np.matrix(np.zeros(shape))
        for i in range(shape[0]):    
            for j in range(shape[1]):  
                A[i,j] = self.derivative(X[i,j])
        return A
    
    def evaluate(self, X):
        shape = np.shape(X)
        A = np.matrix(np.zeros(shape))
        for i in range(shape[0]):    
            for j in range(shape[1]):  
                A[i,j] = self.function(X[i,j])
        return A

def augment(X, value):
    n, _ = np.shape(X)
    return np.hstack((value*np.ones(shape= (n,1)), X ))    

def getSample(D, sampleID, k):
    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i] == k]
    rIndex = [i for i in range(n) if sampleID[i] != k]
    partition = namedtuple('data','R E')
    data = namedtuple('data','X Y labels')
    if D.labels is None:
        split = partition(data(D.X[rIndex,:], D.Y[rIndex,:], None ),
                          data(D.X[sIndex,:], D.Y[sIndex,:], None ))
    else:
        split = partition(data(D.X[rIndex,:], D.Y[rIndex,:],
                               [D.labels[i] for i in rIndex]),
                            data(D.X[sIndex,:], D.Y[sIndex,:],
                                 [D.labels[i] for i in sIndex]) )
                               
    return split

def initialize(g, X, fns, dfns):
    
    u = .1
    xLst = []     
    hLst = []
    gLst = []
    sgLst = []
    vLst = []
    iLst = []
    zpLst = []
    [], [], [], [], [], [], [] = xLst , hLst, gLst, sgLst, vLst, iLst, zpLst
    ''' m is the number of mappings between layers '''
    m = len(g) - 1
    for k in range(0, m-1):
        
        shape = (g[k+1] + 1, g[k+2]) if k > 0 else (g[k+1], g[k+2])
            
        gLst.append(np.matrix(np.random.uniform(0, u, shape )))
        hLst.append(np.matrix(np.random.uniform(-u, u, shape )))
        if k > 0:
            n,m = np.shape(X)
            print(n,m)
            X0 = np.ones((n,1))
            A = np.hstack((X,X0))
        else: A = X
        
        X = fns[k-1](A * hLst[k-1])
        xLst.append(X)
        zpLst.append(dfns[k](A*hLst[k]))
        
        vLst = [H.copy() for H in hLst]
        sgLst = [abs(H.copy()) for H in hLst]
        iLst = [H.copy() for H in hLst]
    A = augment(xLst[m], 1)
    yHat = fns[m](A * hLst[m])
    return yHat, xLst, hLst, gLst, zpLst, vLst, iLst, sgLst

def fProp(A, hLst, fns, dfns, zpLst):
    ''' Forward propagation assuming one coefficient matrix '''
     
    u = .1
    xLst = []     
    zpLst = []
    [], [], [], [], [], [], [] = xLst , hLst, gLst, sgLst, vLst, iLst, zpLst
    ''' m is the number of mappings between layers '''
    m = len(g) - 1
    for k in range(0, m-1):
        shape = (g[k+1] + 1, g[k+2]) if k > 0 else (g[k+1], g[k+2])
        hLst.append(np.matrix(np.random.uniform(-u, u, shape )))
        
        X = fns[k](A * hLst[k])
        A = augment(X, 1) if k > 0 else X
        xLst.append(X)
        zpLst.append(dfns[k](A*hLst[k]))
        
    A = augment(xLst[m], 1)
    yHat = fns[m](A * hLst[m])
    return xLst, zpLst, yHat

def gcDeepReg(gLst, hLst, X, fns, dfns, dEdyhat, L1, L2):    
    aLst = []        
    zPrimeLst = []
    m = len(hLst)
    s = dEdyhat.shape[1]
    penalty = 0
    for k in range(m):
        if k > 0:
            n,m = np.shape(X)
            print(n,m)
            X0 = np.ones((n,1))
            A = np.hstack((X,X0))
        else: A = X
        
        aLst.append(A)
        X = fns[k](X * hLst[k])
        zPrimeLst.append(dfns[k](A*hLst[k]))

    for r in range(m-1, -1, -1):
        I, J = hLst[r].shape
        E = np.matrix(np.zeros( ( I, J)))   
        AH = aLst[r]*hLst[r]
        
        for i in range(I):
            for j in range(J):
                E[i, j] = 1
                W = np.multiply(aLst[r]*E, dfns[r](AH) )
                E[i, j] = 0
                Z = W
                for k in range(r+1,m): 
                    Z = np.multiply(Z*hLst[k][1:,:], zPrimeLst[k] )
                penalty += L2*hLst[r][i,j]**2 + L1*abs(hLst[r][i,j])
                dPenalty = 2*L2*hLst[r][i,j] + L1*np.sign(hLst[r][i,j])
                gLst[r][i, j] = np.sum([Z[:, c].T*dEdyhat[:, c] for c in range(s)]) +dPenalty
    return gLst, penalty  
def RMSProp(smoothGrad, grad, a):
    lamb = .9
    smoothGrad= lamb*smoothGrad + (1 - lamb)*np.power(grad,2)
    return smoothGrad, a/np.sqrt(smoothGrad + 1e-6)

g = [p, p, p, p, p, s] 
K = 10
meanR = []
r2={}
sampleID = [np.random.choice(range(K)) for i in range(n)]

z0 = ActivationFunction(rlu, rluP)
z1 = ActivationFunction(tanh, tanhP)
z2 = ActivationFunction(identity, unit)


fns = [z0.evaluate, z1.evaluate,z2.evaluate]
dfns = [z0.evaluate,z1.differentiate, z2.differentiate]

'set random seed'
seed = 0
np.random.seed(seed)

batSize = 300
batProgress = 0

L1=0.005
L2=0.005
n = 100
for k in range(K):
    C = getSample(D, sampleID,k)
    n,m = np.shape(C.R.Y)
    nBatSamples = int(n/batSize)
    batchID = [np.random.choice(range(nBatSamples)) for i in range(n)]
    B = getSample(C.R, batchID, 0)
    batSize = np.shape(B.E.X)[0]
    batProgress = 0
    nProcessed = 0
    #print(R.X.shape, R.Y.shape, E.X.shape, E.Y.shape) #has not pmsostemedt tpp trpmg;y pme aristotles ,aom argaument. That Gd is immortl. Woj
    #H = np.linalg.solve(C.T*C.R,C.R.T*C.Y)
    #print(H)
    #errors = C.Y -C.X*H
    #rsq = f.rSqr(C.Y, errors)
    #meanR.append(np.mean(rsq))
    #r2[k] = rsq
    initialLst = initialize(g, B.E.X, fns, dfns)
    yHat, xLst, hLst, gLst, zpLst, vLst, iLst, sgLst = initialLst
    #MEAN = np.mean(meanR)
    #print(MEAN)
    #%%
    #  Begin iterations 
    it = 0    
    epoch = 0
    smoothGrad = ()
   
    
    for it in range(5000): 
        epoch = int(nProcessed/n)
        xLst, zpLst, yHat = fProp(xLst, hLst, fns,dfns, zpLst)
        
        # Forward Propagation 
        #yHat = f1.fProp(C.X, hLst, fns)
        dEdyhat = -2*(B.E.Y-yHat)
        a = .01*np.exp(-.001*it)
        #smoothGrad, stepsize = f2.RMSProp(smoothGrad, grad, a)
        
        alpha = .9 - .4*np.exp((1-it)*0.0001)
        for r in range(len(hLst)):
            iLst[r] = hLst[r] - alpha*vLst[r]
        #%%
        gLst, penalty = gcDeepReg(gLst, iLst, C.X, fns, dfns, dEdyhat, L1, L2)    


        for r in range(len(hLst)):
    
            ''' RMS + Nesterov '''
            sgLst[r], stepSizes = RMSProp(sgLst[r], gLst[r], a)
            vLst[r] = .9*vLst[r] + np.multiply(stepSizes, gLst[r])
            hLst[r] -= vLst[r]
            
            #%%
        # Evaluate progress using the test sample
        obsAcc = np.diag((C.E.Y - yHat).T*(C.E.Y - yHat))/C.E.Y.shape[0]
        rsq = [1 - x for x in obsAcc]
        objFunction = sum(np.diag(dEdyhat.T*dEdyhat))/dEdyhat.shape[0]
        
        batProgress += np.shape(B.E.X)[0]
        nProcessed += np.shape(B.E.X)[0]
        
        if batProgress >= n:
            np.random.shuffle(batchID)
            batProgress = 0
        label = (it+1)%nBatSamples
        B = getSample(C.R, batchID, label)
        xLst[0] = B.E.X
        
        string = '\r'+str(k) + '\t' +str(K)+ '\t'+str(it)
        string += '\t r-sqr = '+ ' '.join([str(round(r,3)) for r in rsq]) 
        string +='\t obj = '+ str(round(sum(obsAcc),5))
        print(string,end'')
    
