#!/usr/bin/env python
# coding: utf-8

# %%
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

def amplification_factor(A, b, z):
    y0 = np.ones(len(b))
    bT = np.transpose(b)
    I = np.identity(len(b))
    a = np.matmul(z*bT,inv(I-z*A))
    amp = 1 + np.matmul(a,y0)
    return np.absolute(amp)

def tabulate_amplification_factor(A, b, z1d, nz):
    fac = np.zeros((nz,nz))
    for i in range(nz):
        for j in range(nz):
            zp = 0.5*(z1d[i-1]+z1d[i]) + 0.5*1j*(z1d[j-1]+z1d[j])
            fac[i,j] = amplification_factor(A,b,zp)
    return fac


def plot_stability_region(A, b, zmax=5.0, nz=512):
    # Create a grid
    z1d = np.linspace(-zmax, zmax, nz)
    z_re, z_im = np.meshgrid(z1d, z1d, indexing='ij')

    fac = tabulate_amplification_factor(A, b, z1d, nz)

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.95-0.15, 0.95-0.15])
    ax.set_xlabel(r"Re")
    ax.set_ylabel(r"Im")
    ax.set_aspect('equal')
    ax.pcolormesh(z_re, z_im, fac < 1.0, cmap='cividis')
    ax.grid(visible=True)

    return fig


# %%
# Explicit Euler
A = np.array([[0]])
b = np.array([[1]])
fig = plot_stability_region(A, b)
fig.gca().set_title("Forward Euler Stability Region")


# %%
# Implicit Euler
A = np.array([[1]])
b = np.array([[1]])
fig = plot_stability_region(A, b)
fig.gca().set_title("Implicit Euler Stability Region")


# %%
# Heun's method
A = np.array([[0, 0], [1, 0]])
b = np.array([0.5, 0.5])
fig = plot_stability_region(A, b)
fig.gca().set_title("RK2 Stability Region")


# %%
# Classical RK4
A = np.array([
    [0.0, 0.0, 0.0, 0.0],
    [0.5, 0.0, 0.0, 0.0],
    [0.0, 0.5, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]])
b = np.array([1/6, 1/3, 1/3, 1/6])
fig = plot_stability_region(A, b)
fig.gca().set_title("RK4 Stability Region")

# %%
# SSP-RK3
A = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1/4, 1/4, 0.0]])
b = np.array([1/6, 1/6, 2/3])
fig = plot_stability_region(A, b)
fig.gca().set_title("SSP-RK3 Stability Region")

# %%
# Gauss-Legendre RK2
s3 = np.sqrt(3.0)
A = np.array([
    [1/4, 1/4-s3/6],
    [1/4 + s3/6, 1/4]])
b = np.array([1/2, 1/2])
fig = plot_stability_region(A, b)
fig.gca().set_title("Gauss-Legendre-RK2 Stability Region")

# %%
plt.show()

