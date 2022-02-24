#!/usr/bin/env python
# coding: utf-8


import numpy as np
import matplotlib.pylab as plt


def upwind_scheme(u,a,dt,h):
    for n in range(len(u[:,-1])-1):
        for i in range(len(u[-1,:])):
            u[n+1][i] = u[n][i] - a*(dt/h)*(u[n][i]-u[n][i-1])
    return u
def lw_scheme(u,a,dt,h):
    for n in range(len(u[:,-1])-1):
        for i in range(len(u[-1,:])-1):
            u[n+1][i] = u[n][i] - (a/2)*(dt/h)*(u[n][i+1]-u[n][i-1]) + (a*a/2)*((dt/h)**2)*(u[n][i+1]-2*u[n][i]+u[n][i-1])
    return u


h = 0.05
dt = 0.04
a=1
c = 0.8
t = 17
shift = t
space_grid = np.arange(0,25,h)
time_grid = np.arange(0,17,dt)


def analytic_function(x):
    return np.exp(-20*((x-2)**2))+np.exp(-((x-5)**2))


v = np.zeros((len(time_grid),len(space_grid)))
v[0,:] = analytic_function(space_grid)
w = np.zeros((len(time_grid),len(space_grid)))
w[0,:] = analytic_function(space_grid)


sol_uw = upwind_scheme(v,a,dt,h)
sol_lw = lw_scheme(w,a,dt,h)
sol_uw = sol_uw[-1,:]
sol_lw = sol_lw[-1,:]


plt.plot(space_grid, sol_uw, 'r-.')
plt.plot(space_grid, sol_lw, 'b-.')
plt.plot(space_grid,analytic_function(space_grid-shift), 'k-.')
plt.show()




