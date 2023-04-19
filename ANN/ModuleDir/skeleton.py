# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:04:55 2019

@author: brian
"""

import sys
print(sys.version_info)
import time
import random
import numpy as  np

from collections import namedtuple
dataSet = namedtuple('data','X Y labels meanY stdY')
dataTuple = namedtuple('variables','X Y mean stdDev')   

def rSqr(Y, errors):
    varY = np.var(Y, axis = 0)
    varE = np.var(errors, axis = 0)
    lst = np.matrix.tolist(1 - varE/varY)[0]

    return [round(float(r),4) for r in  lst]

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
    
def getCVsample(D, sampleID, k):
    dataTuple = namedtuple('variables','X Y mean stdDev')   
    
    n = len(sampleID)
    eIndex = [i for i in range(n) if sampleID[i] == k]
    rIndex = [i for i in range(n) if i not in eIndex]
    R = dataTuple(D.X[rIndex,:], D.Y[rIndex,:], 0, 1)
    E = dataTuple(D.X[eIndex,:], D.Y[eIndex,:], 0, 1)
    return R, E

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

      
def augment(X, value):
    n = X.shape[0]
    return np.hstack((value*np.ones(shape= (n,1)), X ))
    
path = '/Users/glenwoodworth/Documents/Spring 2019/M462/Data/parkinsons_updrs.csv'   

D = parkinsonsData(path)
n, p = D.X.shape
n, s = D.Y.shape
print(n,p, s)


q = 10
g = [p, q, s]

K = 10
sampleID = [random.choice(range(K)) for i in range(n)]
print(p, q, s)