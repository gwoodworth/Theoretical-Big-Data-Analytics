#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:10:12 2019

@author: glenwoodworth
"""
import math
def sqrt(n):
    for l in range(math.floor((n))):
        s = l**2
        if s == n:
            return s
            break
        elif n < s:
            t = math.sqrt(s)
            return t
            break
        else:continue
def isPrime(n):
    if n == 2:
        return print(n,"is prime")
    elif n%2 ==0:
        return print(n,"is not prime")
    else:
        s = math.floor(.5*(math.floor(sqrt(n))-1))
        f = []
        q = []
        for k in range(1,s):
            i = 2*k+1
            p = n%i
            q.append(1)
            if p==0:
                f.append(0)
            else:f.append(1)
        if sum(f)==sum(q):
            return print(n,"is prime")
        else: return print(n,"is not prime")
def main():
    for i in range(100):
        num = eval(input("What is the number?: "))
        isPrime(num)
main()