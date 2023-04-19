# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 10:22:03 2018

@author: brian
"""
import numpy as np
from collections import namedtuple

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

def dEdyhatSqr(Y, yHat):
    return -2 * (Y - yHat)

def gradComputerOne(gLst, xLst, hLst, fns, dfns, dEdyhat):
    r = 0
    shape = gLst[r].shape
    A = xLst[r]             
    AH = A * hLst[r]
    zPrime = fns[0](AH)
    for k in range(shape[1]):
        for i in range(shape[0]):
            dyhatdhK = np.multiply( A[:,i], zPrime[:,k])
            gLst[r][i,k] = dyhatdhK.T*dEdyhat[:,k]
            
    return gLst
    
def initialize(g, X):
    
    u = .1
    ''' set up the main components of the NN '''
    n, p, s = g
    ''' xLst - contains input and output matrices '''
    xLst = [X, np.matrix(np.zeros((n,s))) ]
    hLst = [ np.matrix(np.random.uniform(-u, u, (p, s) ))   ]

    gLst  = [0*M.copy() for M in hLst]              
    ''' Not using this list ... yet '''
    stepSizeLst  = [0*M.copy() for M in hLst]          
    
    initialLst = [xLst, hLst, gLst, stepSizeLst]
    return initialLst    

def fProp(A, hLst, fns):
    ''' Forward propagation assuming one coefficient matrix '''
    
    AH = A * hLst[0]
    yHat = fns[0](AH)
    return yHat



def rSqr(Y, E):
    varY = np.var(Y, axis = 0)
    varE = np.var(E, axis = 0)
    lst = np.matrix.tolist(1 - varE/varY)[0]
    return [round(float(r),4) for r in  lst]
    
def augment(X, value):
    n, _ = np.shape(X)
    return np.hstack((value*np.ones(shape= (n,1)), X ))    

def getCVsample(D, sampleID, k):
    cvData = namedtuple('data','X Y')
    cvPartition = namedtuple('data', 'R E')
    n = len(sampleID)
    sIndex = [i for i in range(n) if sampleID[i] == k]
    rIndex = [i for i in range(n) if i not in sIndex]
    split = cvPartition(cvData(D.X[rIndex,:], D.Y[rIndex,:]), cvData(D.X[sIndex,:], D.Y[sIndex,:]) )
    return split    

''' *************************** Functions to read data files *********************************    '''
def parkinsonsData(path):
    
    dataSet = namedtuple('data','X Y meanY stdY labels')
    records =  open(path,'r').read().split('\n')
    variables = records[0].split(',')
    
    iX = [1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    iY = [4, 5]
    
    print('Predictor variables:')    
    for i in range(len(iX)) : 
        print(iX[i], variables[iX[i]])
    print('Target variables:')    
    for i in range(len(iY)) : 
        print(iY[i], variables[iY[i]])
        
    n = len(records)-1
    p = len(iX) + 1
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
        for k in range(p-1):
            X[i,k+1] = lst[iX[k]]    
    
    s = np.std(Y, axis=0)            
    m = np.mean(Y, axis = 0)    
    Y = (Y - m)/s
    
    X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0)) / np.std(X[:,1:], axis=0)            

    data = dataSet(X, Y, m, s, None)
    return data
    
def breastCancerData(path):
    dataSet = namedtuple('data','X Y meanY stdY labels')
    f = open(path,'r')
    data = f.read()
    f.close()
    records = data.split('\n')

    n = len(records)-1
    p = len(records[0].split(','))-1
    s = 2 # number of classes
    Y = np.matrix(np.zeros(shape = (n, s)))
    X = np.matrix(np.ones(shape = (n, p)))
    labels = [0]*n
    for i, line in enumerate(records):
        record = line.split(',')
        
        try:
            labels[i] = int(record[p+1]=='4')
            Y[i,labels[i]] = 1
            X[i,1:] =  [int(x)/10 for x  in record[1:p+1]]

        except(ValueError,IndexError):
            pass    
    s = np.std(Y, axis=0)            
    m = np.mean(Y, axis = 0)    
    data = dataSet(X, Y, m, s, [np.argmax(Y[i,:]) for i in range(n)]) 

    return data


def BostonHousing(path):
    #https://archive.ics.uci.edu/ml/datasets/housing
    dataSet = namedtuple('data','X Y meanY stdY labels')
    p = 13 + 1
    f = open(path,'r')
    D = f.read()
    records = D.split('\n')
    n = len(records) - 1
    Y = np.matrix(np.zeros(shape = (n,1)))
    X = np.matrix(np.ones(shape = (n,p)))
    
    endCol = [8,15,23,26,34,43,49,57,61,68,75,82,89,96]
    startCol = [0] + [col+1 for col in endCol[0:13]]
    
    for i, record in enumerate(records):
        
    
        try:
            for j, pair in enumerate(zip(startCol, endCol)):
                
                string = record[pair[0]:pair[1]]
                try:
                    X[i,j+1] = float(string)
                    #print(i,X[i,j+1] )
                except(IndexError):
                    Y[i] = float(string)
        except(ValueError):
            pass
            
    s = np.std(Y, axis=0)            
    m = np.mean(Y, axis = 0)    
    Y = (Y - m)/s
    
    X[:,1:] = (X[:,1:] - np.mean(X[:,1:], axis=0)) / np.std(X[:,1:], axis=0) 
     
    data =  dataSet(X, Y, m, s, None)
    return data
