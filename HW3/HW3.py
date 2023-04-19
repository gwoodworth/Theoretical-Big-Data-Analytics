#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 15:00:48 2019

@author: glenwoodworth
"""
from collections import OrderedDict
import datetime
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import timedelta
import numpy as np

def plot(errorLst):
    fig, ax = plt.subplots(figsize=(10, 5))
    _,_,_,_= plt.axis()
    plt.grid(which='major', linestyle='--', linewidth='0.25', color='blue')
    plt.xlabel('Day', size = 8)
    plt.ylabel('Dollars')
    plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
    
    plt.scatter([i for i, _, _ in errorLst], [t for _, t, _ in errorLst], marker = '.', color = 'k',  s=20, label ='Targets')
    plt.plot([i for i, _, _ in errorLst], [f for _, _, f in errorLst],  color = 'r',   label ='Forecasts')
    
    plt.legend()
    plt.show()
    
def buildInteraction(dataDict, indexLst, p):
    ''' Interaction  - rebuild the data. Use the first p variables and to create 2 interactions ...'''
    ''' build X and y anew '''
    a,b,c = indexLst
    ''' a and b identify the predictors from which the interaction variable is constructed '''
    n = len(dataDict) - 1
    X = np.matrix(np.ones((n , p + 3)))
    y = np.matrix(np.zeros((n , 1)))
    for i in range(1, n):
        day = days[i]
        X[i,1:p+1] = dataDict[day][0][:p]
        X[i, p+1] = dataDict[day][0][a] * dataDict[day][0][b]
        X[i, p+2] = rlu(dataDict[day][0][b] * dataDict[day][0][c])
        X[i, p+1] = rlu(X[i, p+1])
        y[i, 0] = dataDict[day][1]
    return y, X

def cubicSplineDesignMatrix(retain, knots):
    ''' Cubic spline forecasting function :
    This function builds a desgin matrix consisting of cubic splines.
    The number of rows is determined by the variable retain.
    The number of splines (and columns of the design matrix) is determined
    by the length of the knot list (knots)
    '''
    p = len(knots)
    xSub = np.matrix(np.zeros((retain, p)))
    for j in range(p):
        for i in range(retain):
            
            xSub[i, j]  = (i > knots[j])*(i - knots[j])**3  
    xSub = np.hstack((np.ones((retain, 1)) , xSub))            
    return xSub
    
def rlu(x):
    return 0*int(x <= 0) + x*int(x > 0)

file = 'ibm.txt'
#path = 'C:\\M462\\Data\\'
path = '/Users/glenwoodworth/Documents/Spring 2019/M462/Data/'

with open(path+file, "r") as f: 
    data = f.read().split('\n')      
variables = data[0].split(',') 

print(' '.join(variables))
targetMean = float( data[1].split(',')[2] ) 
targetSD = float( data[2].split(',')[2] ) 
print(targetMean,targetSD)

days = []
for i, record in enumerate(data[3:]):
    
    lst = record.split(',')
    days.append(datetime.datetime.strptime(lst[0], '%Y-%m-%d').date())

dataDict = OrderedDict.fromkeys(days[1:])

for i, record in enumerate(data[3:]):
    lst = record.split(',')
    
    ''' Stagger the y's - these become the targets of forecasting '''
    ''' Accomplished by associating the day[i] X vector with next day (i+1) target in dataDict'''
    currentDay = days[i]
    x = [float(z) for i, z in enumerate(lst) if i > 2]
    y = float(lst[2])

    try:
        dataDict[currentDay].extend([y])
        ''' an exception is generated if the loop is on the first index; hence there is
        previous day target to associate with the current predictor vector (x). 
        If this is the case, then do not add the predictor to dataDict '''
    except(KeyError):
        pass
    try:
        nextDay = days[i+1]
        dataDict[nextDay] = [x]
        ''' an exception is generated if the loop has reached the last index; hence there is
        no days[i+1] in days. If this is the case, then we are done adding targets to dataDict '''
    except(IndexError):
        pass
    ''' keep the predictor vectors (x) associated with the current day'''
    print(currentDay, nextDay)
  
    
print('data for the last day = ', dataDict[days[-1]])    

''' translate the target values from the normalized data set values to the dollars '''

targets = [targetMean + targetSD*y for x, y in dataDict.values()]
#%%
print("1.a Exponential forecasting")
start = 20
p = 2
M = targets[start-p]
end = len(targets)
a = .99
m = targets[start-p]
M = 0

errorELst = []
for i in range(end):
    target = targets[i]
    M = a*targets[i-2] + (1-a)*m
    m = M
    errorELst.append([i,target,M])
rsq = (end*np.var([t for _, t, _ in errorELst]) - sum([(t-f)**2 for _, t, f in errorELst]))/(end*np.var([t for _, t, _ in errorELst]) )
print('α = ',a)
print("r**2 = ",rsq)

plot(errorELst)

#%%
print("1.b Holt-Winters")
a_h = a/2
a_b = 0.000001
m = []

b = []
B = 0
y = []
errormean = 0
errorexpon = 0
errorHWLst = []

for i in range(end):
    M = a_h*targets[i] + (1-a_h)*(M+B)
    m.append(M)
    B = a_b*(m[i]+m[i-1]) + (1-a_b)*(B)
    b.append(B)
    f = m[i] + b[i]
    y.append(f)
    errormean = errormean + (targets[i]-np.mean(targets))**2
    errorexpon = errorexpon + (targets[i]-y[i])**2
    target = targets[i]
    errorHWLst.append([i,target,f])
rsq = 1- (errorexpon/errormean)
print('α_h = ',a_h)
print('α_b = ',a_b)
print("r**2 = ",rsq)

plot(errorHWLst)
##
"1.c New Baseline Plot"
print('New Baseline: Holt-Winters')
print('Tuning Constants:','α_h = ',a_h,'α_b = ',a_b)
yh = y
###############################################
#%%
print('Problem 2')
def buildInteractionNone(dataDict, p):
    ''' Interaction  - rebuild the data. Use the first p variables and to create 2 interactions ...'''
    ''' build X and y anew '''

    ''' a and b identify the predictors from which the interaction variable is constructed '''
    n = len(dataDict) - 1
    X = np.matrix(np.ones((n , p+1)))
    y = np.matrix(np.zeros((n , 1)))
    for i in range(1, n):
        day = days[i]
        X[i,1:p+1] = dataDict[day][0][:p]
        y[i, 0] = dataDict[day][1]
    return y, X



p = 4
y, X = buildInteractionNone(dataDict, p)
 

betaHat = np.linalg.solve(X.T*X, X.T*y)    
print(betaHat)

retain = 20
errorLst = []
n = len(dataDict)
errorLst = []
p = X.shape[1] - 1    
for i in range(0, n-retain-1):
    xSub = X[i:retain+i,:]
    ySub = y[i:retain+i,:]
    target = float(yh[retain+i]) 
    
    betaHat = np.linalg.solve(xSub.T*xSub, xSub.T*ySub)    
    forecast = float(targetMean + targetSD*X[retain+i,:]*betaHat)
    errorLst.append([i, target , float(forecast)])

print('\nError summary')
r2 = (n*np.var([f for _, _, f in errorHWLst]) - sum([(t-f)**2 for _, t, f in errorLst]))/(n*np.var([t for _, t, _ in errorLst]) )
print(round(r2, 3),'\t',
      round(np.mean([t-f for _, t, f in errorLst]), 3),'\t',
      round(np.mean([abs(t-f) for _, t, f in errorLst]), 3),'\t',
      round(np.sqrt(np.mean([(t-f)**2 for _, t, f in errorLst])),3) )

plot(errorLst)
plt.figure()

#%%
def buildInteractionOneA(dataDict, indexLst, p):
    ''' Interaction  - rebuild the data. Use the first p variables and to create 2 interactions ...'''
    ''' build X and y anew '''
    a,b = indexLst
    ''' a and b identify the predictors from which the interaction variable is constructed '''
    n = len(dataDict) - 1
    X = np.matrix(np.ones((n , p + 2)))
    y = np.matrix(np.zeros((n , 1)))
    for i in range(1, n):
        day = days[i]
        X[i,1:p+1] = dataDict[day][0][:p]
        X[i, p+1] = dataDict[day][0][a] * dataDict[day][0][b]
        y[i, 0] = dataDict[day][1]
    return y, X
retain = 15
errorLst = []
n = len(dataDict)
  
indexLst = [1, 2]
p = 2    
y, X = buildInteractionOneA(dataDict, indexLst, p)  

betaHat = np.linalg.solve(X.T*X, X.T*y)    
print(betaHat)

retain = 20
errorLst = []
n = len(dataDict)
errorLst = []
p = X.shape[1] - 1    
for i in range(0, n-retain-1):
    xSub = X[i:retain+i,:]
    ySub = y[i:retain+i,:]
    target = float(targetMean + targetSD*y[retain+i]) 
    
    betaHat = np.linalg.solve(xSub.T*xSub, xSub.T*ySub)    
    forecast = float(targetMean + targetSD*X[retain+i,:]*betaHat)
    errorLst.append([i, target , float(forecast)])

print('\nError summary')
r21 = (n*np.var([t for _, t, _ in errorHWLst]) - sum([(t-f)**2 for _, t, f in errorLst]))/(n*np.var([t for _, t, _ in errorLst]) )
print(round(r21, 3),'\t',
      round(np.mean([t-f for _, t, f in errorLst]), 3),'\t',
      round(np.mean([abs(t-f) for _, t, f in errorLst]), 3),'\t',
      round(np.sqrt(np.mean([(t-f)**2 for _, t, f in errorLst])),3) )

plot(errorLst)
plt.figure()
#%%
def buildInteractionOneB(dataDict, indexLst, p):
    ''' Interaction  - rebuild the data. Use the first p variables and to create 2 interactions ...'''
    ''' build X and y anew '''
    b,c = indexLst
    ''' a and b identify the predictors from which the interaction variable is constructed '''
    n = len(dataDict) - 1
    X = np.matrix(np.ones((n , p + 2)))
    y = np.matrix(np.zeros((n , 1)))
    for i in range(1, n):
        day = days[i]
        X[i,1:p+1] = dataDict[day][0][:p]
        X[i, p+1] = dataDict[day][0][b] * dataDict[day][0][c]
        y[i, 0] = dataDict[day][1]
    return y, X
retain = 15
errorLst = []
n = len(dataDict)
  
indexLst = [2, 3]
p = 2    
y, X = buildInteractionOneB(dataDict, indexLst, p)  

betaHat = np.linalg.solve(X.T*X, X.T*y)    
print(betaHat)

retain = 20
errorLst = []
n = len(dataDict)
errorLst = []
p = X.shape[1] - 1    
for i in range(0, n-retain-1):
    xSub = X[i:retain+i,:]
    ySub = y[i:retain+i,:]
    target = float(targetMean + targetSD*y[retain+i]) 
    
    betaHat = np.linalg.solve(xSub.T*xSub, xSub.T*ySub)    
    forecast = float(targetMean + targetSD*X[retain+i,:]*betaHat)
    errorLst.append([i, target , float(forecast)])

print('\nError summary')
r22 = (n*np.var([t for _, t, _ in errorHWLst]) - sum([(t-f)**2 for _, t, f in errorLst]))/(n*np.var([t for _, t, _ in errorLst]) )
print(round(r22, 3),'\t',
      round(np.mean([t-f for _, t, f in errorLst]), 3),'\t',
      round(np.mean([abs(t-f) for _, t, f in errorLst]), 3),'\t',
      round(np.sqrt(np.mean([(t-f)**2 for _, t, f in errorLst])),3) )

plot(errorLst)
plt.figure()
#%%
def buildInteractionTwo(dataDict, indexLst, p):
    ''' Interaction  - rebuild the data. Use the first p variables and to create 2 interactions ...'''
    ''' build X and y anew '''
    a,b,c = indexLst
    ''' a and b identify the predictors from which the interaction variable is constructed '''
    n = len(dataDict) - 1
    X = np.matrix(np.ones((n , p + 3)))
    y = np.matrix(np.zeros((n , 1)))
    for i in range(1, n):
        day = days[i]
        X[i,1:p+1] = dataDict[day][0][:p]
        X[i, p+1] = dataDict[day][0][a] * dataDict[day][0][b]
        X[i, p+2] = dataDict[day][0][b] * dataDict[day][0][c]
        y[i, 0] = dataDict[day][1]
    return y, X
retain = 15
errorLst = []
n = len(dataDict)
  
indexLst = [1, 2, 3]
p = 3    
y, X = buildInteractionTwo(dataDict, indexLst, p)  

betaHat = np.linalg.solve(X.T*X, X.T*y)    
print(betaHat)

retain = 20
errorLst = []
n = len(dataDict)
errorLst = []
p = X.shape[1] - 1    
for i in range(0, n-retain-1):
    xSub = X[i:retain+i,:]
    ySub = y[i:retain+i,:]
    target = float(targetMean + targetSD*y[retain+i]) 
    
    betaHat = np.linalg.solve(xSub.T*xSub, xSub.T*ySub)    
    forecast = float(targetMean + targetSD*X[retain+i,:]*betaHat)
    errorLst.append([i, target , float(forecast)])

print('\nError summary')
r23 = (n*np.var([t for _, t, _ in errorHWLst]) - sum([(t-f)**2 for _, t, f in errorLst]))/(n*np.var([t for _, t, _ in errorLst]) )
print(round(r23, 3),'\t',
      round(np.mean([t-f for _, t, f in errorLst]), 3),'\t',
      round(np.mean([abs(t-f) for _, t, f in errorLst]), 3),'\t',
      round(np.sqrt(np.mean([(t-f)**2 for _, t, f in errorLst])),3) )

plot(errorLst)
plt.figure()
#%%
d = {'Interaction 1': [r2], 'Interaction 2': [r21], 'Interaction 3': [r22], 'Interaction 4': [r23]}
import pandas as pd
df = pd.DataFrame(data=d)
print(df.to_latex())
#%%
print('Problem 3')



L=0.27
retain = 15
errorLst = []
n = len(dataDict)
  
indexLst = [2, 3]
p = 2    
y, X = buildInteractionOneB(dataDict, indexLst, p)  

betaHat = np.linalg.solve(X.T*X, X.T*y)    
print(betaHat)

retain = 20
errorLst = []
n = len(dataDict)
errorLst = []
p = X.shape[1] - 1    
for i in range(0, n-retain-1):
    xSub = X[i:retain+i,:]
    ySub = y[i:retain+i,:]
    target = float(targetMean + targetSD*y[retain+i]) 
    
    betaHat = np.linalg.solve(xSub.T*xSub + L*np.eye(p+1), xSub.T*ySub)    
    forecast = float(targetMean + targetSD*X[retain+i,:]*betaHat)
    errorLst.append([i, target , float(forecast)])

print('\nError summary')
r22 = (n*np.var([t for _, t, _ in errorHWLst]) - sum([(t-f)**2 for _, t, f in errorLst]))/(n*np.var([t for _, t, _ in errorLst]) )
print(round(r22, 5),'\t',
      round(np.mean([t-f for _, t, f in errorLst]), 3),'\t',
      round(np.mean([abs(t-f) for _, t, f in errorLst]), 3),'\t',
      round(np.sqrt(np.mean([(t-f)**2 for _, t, f in errorLst])),3) )

plot(errorLst)
plt.figure()
#%%
print('Problem 4')

retain = 25

knots = [ int(retain*.20), int(retain*.80)]
print(knots)
x = cubicSplineDesignMatrix(retain, knots)
def buildInteractionNone(dataDict, p):
    ''' Interaction  - rebuild the data. Use the first p variables and to create 2 interactions ...'''
    ''' build X and y anew '''

    ''' a and b identify the predictors from which the interaction variable is constructed '''
    n = len(dataDict) - 1
    X = np.matrix(np.ones((n , p+1)))
    y = np.matrix(np.zeros((n , 1)))
    for i in range(1, n):
        day = days[i]
        X[i,1:p+1] = dataDict[day][0][:p]
        y[i, 0] = dataDict[day][1]
    return y, X

p = 2
y, X = buildInteractionNone(dataDict, p)
errorLst = []        
for i in range(0, n-retain-1):
    xSub = X[i:retain+i,:]+x
    ySub = y[i:retain+i,:]
    target = float(targetMean + targetSD*y[retain+i]) 
    betaHat1 = np.linalg.solve(xSub.T*xSub, xSub.T*ySub)        
    forecast = float(targetMean + targetSD*xSub[retain-1,:]*betaHat1)
    errorLst.append([i, target , float(forecast)])

print('\nError summary')
r2 = (n*np.var([t for _, t, _ in errorHWLst]) - sum([(t-f)**2 for _, t, f in errorLst]))/(n*np.var([t for _, t, _ in errorLst]) )
print(round(r2, 3),'\t',
      round(np.mean([t-f for _, t, f in errorLst]), 3),'\t',
      round(np.mean([abs(t-f) for _, t, f in errorLst]), 3),'\t',
      round(np.sqrt(np.mean([(t-f)**2 for _, t, f in errorLst])),3) )

plot(errorLst)
      