#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Oppenheimer Snyder collapse

This script draws a spacetime diagram of the Oppenheimer-Snyder collapse
"""

from math import pi, sqrt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


def make_eta_grid(neta):
    """
    Generate a uniform grid in eta
    """
    return np.arange(0, pi, pi/neta)


def am(opt):
    """
    Returns a_m

    Parameters
    ----------
    opt : dictionary
        options.

    Returns
    -------
    a_m
        real.
    """
    return sqrt(opt["R0"]**3/(2*opt["M"]))

def eta_AH(opt):
    return 2*np.arccos(sqrt((2*opt["M"])/opt["R0"]))


def a_from_eta(eta, opt):
    """
    Returns a given eta

    Parameters
    ----------
    eta : numpy array
        eta values at which to evaluate tau.
    opt : dictionary
        options.

    Returns
    -------
    a : numpy array

    """
    return 0.5*am(opt)*(1 + np.cos(eta))


def tau_from_eta(eta, opt):
    """
    Returns tau given eta

    Parameters
    ----------
    eta : numpy array
        eta values at which to evaluate tau.
    opt : dictionary
        options.

    Returns
    -------
    tau : numpy array

    """
    return 0.5*am(opt)*(eta+np.sin(eta))


def chi0(opt):
    """
    Returns the value of \chi_0

    Parameters
    ----------
    opt : dictionary
        options.

    Returns
    -------
    float
        \chi_0.
    """
    return np.arcsin(sqrt((2*opt["M"])/opt["R0"]))


# In[2]:


def plot_space_time(opt):
    # Create figure
    
    eta = make_eta_grid(30)
    eta_eh = np.arange(eta_AH(options(5,1))-chi0(options(5,1)),eta_AH(options(5,1)),pi/30)
    a = a_from_eta(eta, opt)*np.sin(chi0(opt))
    tau = tau_from_eta(eta, opt)
    eh = a_from_eta(eta_AH(options(5,1)), options(5,1))*np.sin(chi0(options(5,1))+(eta_eh-eta_AH(options(5,1))))
    tau_eh = tau_from_eta(eta_eh, options(5,1))
    tau_AH = tau_from_eta(eta_AH(opt), opt)
    
    t = np.linspace(tau_from_eta(eta_AH(options(5,1))-chi0(options(5,1)), options(5,1)), 25, 100)
    triangle = 0.03*signal.sawtooth(2 * np.pi * 5 * t, 0.5)
    
    plt.plot(triangle, t, 'k')
    plt.plot(a,tau)
    plt.vlines(2,tau_AH,25, colors='k', linestyles='dotted')
    plt.plot(eh,tau_eh,'k--')
    plt.margins(x=0,y=0)
    plt.xlabel(r"$r_s/M$")
    plt.ylabel(r"$\tau/M$")


def options(a,b):
    return {
        "R0": a,
        "M": b,
    }


# In[3]:


# Run the main function if executed as a script
if __name__ == '__main__':
    fig5 = plot_space_time(options(5,1))
    fig4 = plot_space_time(options(5*(0.5**(1/3)),0.5))
    fig4 = plot_space_time(options(5*(0.25**(1/3)),0.25))
    fig2 = plot_space_time(options(5*(0.1**(1/3)),0.1))
    plt.savefig("/Users/surp/Desktop/Courses/Numrel/Python/PHYS-597/Week-1/OS-spacetime.jpg",dpi=500)
    plt.show()


# In[ ]:




