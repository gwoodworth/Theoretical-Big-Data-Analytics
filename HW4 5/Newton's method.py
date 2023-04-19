#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 12:57:04 2019

@author: glenwoodworth
"""

def newton(y,x,stop):
    x = [x]
    for k in range(1,stop):
        f = (x[k-1]**2) - 2
        fp = 2*x[k-1]
        xk = x[k-1] + ((y - f)/fp)
        x.append(xk)
    print(x)
newton(0,1,8)