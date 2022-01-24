#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tolman-Oppenheimer-Volkoff

"""

from math import pi
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

class EOSPoly:
    """
    A class representing a polytropic equation of state
    """

    def __init__(self, n=1.0, K=1.0):
        self.n = n
        self.K = K

    def press_from_rho0(self, rho0):
        return self.K*rho0**(1.0 + 1.0/self.n)

    def rho_from_rho0(self, rho0):
        return self.n*self.K*rho0**(1.0/self.n)

    def rho_from_press(self, press):
        return self.n*press + self.rho0_from_press(press)

    def rho0_from_press(self, press):
        return (press/self.K)**(self.n/(self.n + 1))


class TOV:
    """
    A class representing a TOV
    """

    def __init__(self, rho0c, eos):
        self.rho0c = rho0c
        self.eos = eos

    def rhs(self, m, P, r):
        """
        TOV equations RHS
        """
        dPdr = (-(rho_from_press(self, P)*m)/(r**2))*(1+(P/rho_from_press(self, P)))*(1+(4*pi*P*r**3)/m)*((1-(2*m/r))**(-1))
        dmdr = 4*pi*(r**2)*rho_from_press(self, P)
        dm0dr = 4*pi*(r**2)*rho0c*((1-(2*m/r))**(-0.5))
        return [dPdr, dmdr, dm0dr]

    def solve(self, dr=0.001):
        """
        Solve the TOV equations and store the results
        """
        # Central values
        m = 0.0
        P = self.eos.press_from_rho0(self.rho0c)
        
        r = np.linspace(0,10)
        x0 = [P,m]
        x = odeint(rhs,x0,r)
        
        self.M    = x[1]  # Gravitational mass
        self.M0   = x[2]  # Baryonic mass
        self.rho0 = rho0c  # central density
        self.R    = 10  # areal radius


class TOVSequence:
    """
    A class representing a sequence of TOV
    """

    def __init__(self, eos):
        self.eos = eos

    def generate(self, rho0, dr=0.01):
        M = []
        M0 = []
        R = []
        for r in rho0:
            tov = TOV(r, self.eos)
            tov.solve(dr)
            M.append(tov.M)
            M0.append(tov.M0)
            R.append(tov.R)
        self.rho0 = np.array(rho0)
        self.M = np.array(M)
        self.M0 = np.array(M0)
        self.R = np.array(R)



def default_options():
    return {
        "K": 1.0,
        "n": 1.0,
        "dr": 0.001,
        "rho0": np.linspace(0.01, 1.5, 100)
    }



def __main__():
    opt = default_options()
    eos = EOSPoly(opt["n"], opt["K"])
    seq = TOVSequence(eos)
    seq.generate(opt["rho0"], opt["dr"])

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.95-0.15, 0.95-0.15])
    ax.plot(seq.rho0, seq.M, "k-", label=r"$M$")
    ax.plot(seq.rho0, seq.M0, "k--", label=r"$M_0$")
    ax.legend(loc="best")
    ax.set_xlabel(r"$\rho_0$ [P.U.]")
    ax.set_ylabel(r"$M$ [P.U.]")

    fig = plt.figure()
    ax = fig.add_axes([0.15, 0.15, 0.95-0.15, 0.95-0.15])
    ax.plot(seq.R, seq.M, "k-")
    ax.set_xlabel(r"$R$ [P.U.]")
    ax.set_ylabel(r"$M$ [P.U.]")

    plt.show()



# Run the main function if executed as a script
if __name__ == '__main__':
    fig = __main__()
    plt.show()

