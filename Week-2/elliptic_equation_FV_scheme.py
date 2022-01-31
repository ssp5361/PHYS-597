#!/usr/bin/env python
# coding: utf-8


# -*- coding: utf-8 -*-
"""
This script solves the BVP

    - [k(x) u'(x)]' = eta(x) - alpha(x) u(x)
    u(0) = u(1) = 0

We use uniform grid and finite differencing to convert this to a linear
problem in the form

    A u = f

The problem is then solved using the conjugate gradient method
"""

import numpy as np
from numpy import pi, cos, sin, sqrt
from scipy.sparse.linalg import cg, LinearOperator
import matplotlib.pyplot as plt


class FiniteDiffOperator(LinearOperator):
    """
    This class represents the finite differencing discretization of

        - [k(x) u'(x)]' + alpha(x) u(x)

    It inherits from LinearOperator, so it can be used with the iterative
    solvers implemented in scipy.linalg
    """

    def __init__(self, h, kappa, alpha):
        """
        Initialize the FiniteDiffOperator

        Parameters
        ----------
        h : real
            Grid spacing.
        kappa : numpy array
            An array storing kappa at the grid point locations.
        alpha : numpy array
            An array storing alpha at the grid point locations.

        Returns
        -------
        None.
        """
        assert kappa.shape == alpha.shape

        # Initializes the base class
        super().__init__(kappa.dtype, (kappa.shape[0], kappa.shape[0]))

        self.h = h
        self.kappa = kappa
        self.alpha = alpha

    def _matvec(self, u):
        """
        Parameters
        ----------
        u : numpy array of shape (N,)
            input array.

        Returns
        -------
        v : numpy array
            A*u
        """
        # Output array
        v = u.copy()

        v[0] = u[0]
        v[-1] = u[-1]
        
        #Not looping as it's taking forever
        
        #F1 = np.zeros(len(u))
        #F2 = np.zeros(len(u))
        #for i in range(1,len(u)-2):
        #    F1[i] = ((self.kappa[i]+self.kappa[i+1])/2)*((u[i+1]-u[i])/self.h)
        #    F2[i] = ((self.kappa[i]+self.kappa[i-1])/2)*((u[i]-u[i-1])/self.h)
        #    v[i] = (F2[i]-F1[i])/self.h  + self.alpha[i]*u[i]
        #return v
        
        Du = (u[1:] - u[:-1]) / self.h
        k_avg = (self.kappa[1:] + self.kappa[:-1]) / 2
        F = Du * k_avg
        DF = (-F[1:] + F[:-1]) / self.h
        v[1:-1] = DF
        v = v + self.alpha * u
        return v
    
        


class HeatEquationSolver:
    def __init__(self, kappa, eta, alpha, N):
        """
        Initialize the heat equation solver

        Parameters
        ----------
        N : integer
            Number of grid points.
        kappa : real function
            Heat diffusion coefficient as a function of position.
        eta : real function
            Heating source as a function of position.
        alpha : real function
            Absorption opacity as a function of position.

        Returns
        -------
        None.
        """
        self.initialized = False
        self.kappa = kappa
        self.eta = eta
        self.alpha = alpha
        self.set_npoints(N)

    def set_npoints(self,N):
        """
        Set the number of points and initialize the linear operator

        Parameters
        ----------
        N : integer
            Number of grid points.

        Returns
        -------
        None.

        """
        self.N = N
        self.xp = np.linspace(0, 1, self.N + 2)
        self.h = self.xp[1] - self.xp[0]
        self.A = FiniteDiffOperator(self.h, self.kappa(self.xp),
                                    self.alpha(self.xp))
        self.b = self.eta(self.xp)
        
        # Apply boundary conditions to b
        self.b[0] = 0
        self.b[-1] = 0
        self.initialized = True

    def solve(self, opt={"tol": 1e-6}):
        """
        Solve the differential equation

        Parameters
        ----------
        opt : dictionary
              options for the linear solver

        Returns
        -------
        None
        """
        assert self.initialized
        self.u, ierr = cg(self.A, self.b, **opt)
        if ierr > 0:
            print("Warning: CG did not converge to desired accuracy!")
        if ierr < 0:
            raise Exception("Error: invalid input for CG")



N = [10**1,10**2,10**3,10**4]
L2 = np.zeros(len(N))
p = np.zeros(len(N))

def u_ex(x):
    return pi*sin(x)

for i in range(0,len(N)):
    xp = np.linspace(0, 1, N[i] + 2)
    h = xp[1] - xp[0]
    alpha = np.vectorize(lambda x: 1.0)
    kappa = np.vectorize(lambda x: 1.0 + x*(1.0 - x))
    eta = np.vectorize(lambda x: pi*(2*x-1)*cos(pi*x)+(pi**2)*(x*(1-x)+1)*sin(pi*x)+sin(pi*x))

    heat = HeatEquationSolver(kappa, eta, alpha, N[i])
    heat.solve()
    sol = heat.u

    # Run and plot errors
    L2[i] = np.sqrt(h*np.sum((sol-u_ex(xp))**2))
    p[i] = N[i]
       
plt.plot(p, L2,'r-o')
plt.xscale('log')
plt.ylabel(r'$||(\sum_{0}^{N}(E_{i}^{2}))^{0.5}||$')
plt.xlabel(r'$N$')
plt.show()



