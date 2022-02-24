#!/usr/bin/env python
# coding: utf-8

import numpy as np


def f(t, y):
    return y


def RK45(y_n, t_n, dt, tol):
    
    A = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [1/5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [3/40, 9/40, 0.0, 0.0, 0.0, 0.0, 0.0],
    [44/45, -56/15, 32/9, 0.0, 0.0, 0.0, 0.0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0.0, 0.0, 0.0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0.0, 0.0],
    [35/384, 0.0, 500/1113, 125/192, -2187/6784, 11/84, 0.0]])
    
    b1 = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0.0])
    b2 = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40
])
    c = np.array([0.0, 1/5, 3/10, 4/5, 8/9, 1, 1])
    
    k1 = np.zeros(len(c)-1)
    k2 = np.zeros(len(c))
    
    for i in range(0,6):
        k2[i] = f(t_n +  dt*c[i], y_n + dt*np.sum(A[:,i]*k))
    k1 = np.delete(k2, -1)
    
    y1 = y_n + dt*np.sum(b1*k1)
    y2 = y_n + dt*np.sum(b2*k2)
    
    e = np.absolute(y2 - y1)
    
    if e < tol & e > tol/(2**5):
        y_n1 = y1
    if e < tol/(2**5):
        y_n1 = y1
        dt = 2*dt
    if e > tol:
        dt = dt/2
        
    return y_n1, dt


tol = 10**(-5);
dt = 0.5;
t = []
sol = []
y = []

i=0;
y0 = 1;
time = 0;

while time < 10:
    y[0] = y0;
    t[0] = 0;
    sol[0] = y[0];
    
    sol[i+1], dt = RK45(y[i], t[i], dt, tol)
    y[i+1] = sol[i+1];
    t[i+1] = t[i] + dt;
    time = t[i+1]
    i = i+1

print(sol)



