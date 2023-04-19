# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:36:15 2019

@author: brian
"""

import numpy as np

def RMSProp(smoothGrad, grad, a):
    lamb = .9
    smoothGrad= lamb*smoothGrad + (1 - lamb)*np.power(grad,2)
    return smoothGrad, a/np.sqrt(smoothGrad + 1e-6)
    
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
        A = f1.augment(X, 1) if k > 0 else X
        
        X = fns[k](A * hLst[k])
        xLst.append(X)
        zpLst.append(dfns[k](A*hLst[k]))
        
        vLst = [H.copy() for H in hLst]
        sgLst = [abs(H.copy()) for H in hLst]
        iLst = [H.copy() for H in hLst]
    yHat = fns[m-1](A * hLst[k])
    return hLst, gLst, sgLst, vLst, iLst 
    
        
def gcDeep(gLst, hLst, X, fns, dfns, dEdyhat):    
    aLst = []        
    zPrimeLst = []
    m = len(hLst)
    s = dEdyhat.shape[1]
    
    for k in range(m):
        A = np.augment((X, 1)) if k > 0 else X
        
        aLst.append(A)
        X = fns[k](A * hLst[k])
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
                gLst[r][i, j] = sum([Z[:, c].T*dEdyhat[:, c] for c in range(s)])
    return gLst    

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


    ''' Code within the CV for loop '''       
    hLst, gLst, sgLst, vLst, iLst = initialize(g, X, fns)
     
    #  Begin iterations 
    it = 0    
    epoch = 0
    while it < 2000: 
        
        # Forward Propagation 
        yHat = fProp(F.R.X, hLst, fns)
        
        if catFlag:
            dEdyhat = dEdyhatCE(Y, yHat)
        else:
            dEdyhat = -2 * (Y - yHat) 
            
        alpha = .9 - .4*np.exp((1- it)*.0001)
        for r in range(len(hLst)):
            iLst[r] = hLst[r] - alpha*vLst[r]
        
        gLst = gcDeep(gLst, iLst, F.R.X, fns, dfns, dEdyhat)    

        a = .01*np.exp(- .001 * it)
        for r in range(len(hLst)):
    
            ''' RMS + Nesterov '''
            sgLst[r], stepSizes = RMSProp(sgLst[r], gLst[r], a)
            vLst[r] = .9*vLst[r] + np.multiply(stepSizes, gLst[r])
            hLst[r] -= vLst[r]
            
            
        # Evaluate progress using the test sample
        yHat = fProp(F.E.X, hLst, fns)
        obsAcc = np.diag((F.E.Y - yHat).T*(F.E.Y - yHat))/F.E.Y.shape[0]
        rsq = [1 - x for x in obsAcc]
        objFunction = sum(np.diag(dEdyhat.T*dEdyhat))/dEdyhat.shape[0]

        string = '\r'+str(k) + '/'+str(K)+' \t'+str(it)
        string += '\t r-sqr = '+ ' '.join([str(round(r,3)) for r in rsq]) 
        string +='\t obj = '+ str(round(sum(obsAcc),5))
        print(string,end="")
    