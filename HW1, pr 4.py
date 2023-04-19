import numpy as np
import matplotlib.pyplot as plt
####################################
N = 10000
p = 0
minv = []
maxv = []
fnorm = []
b = []
p_max = 2000
loop = int(p_max/20)
for k in range(0,loop):
    p = p+20
    A = np.matrix(np.random.uniform(0,1,(N,p))) #NxP matrix w/ random values [0,1]
    #print('a.',A)

    S= np.diag(A.T*A)
    for i in range(p):
        A[:,i] = A[:,i]/np.sqrt(S[i])

    D = A.T*A
    v=[]
    for ii in range(p-1):
        for j in range(ii+1,p):
            #print(i,j)
            v.append(D[ii,j])
        minv.append(min(v))
        maxv.append(max(v))
        fnorm.append(np.linalg.norm(D))#Frobenius Norm
        if p < p_max:
            b.append(p)
plt.figure()
plt.xlabel('p')
plt.ylabel('minv and maxv')
plt.plot(b,minv,'b')
plt.plot(b,maxv,'r')

plt.figure()
plt.xlabel('p')
plt.ylabel('Frobenius Norm')
plt.plot(b,fnorm)
###################################
