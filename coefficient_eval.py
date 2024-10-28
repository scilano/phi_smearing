"""
This file contains the functions to evaluate the coefficients at the points of the grid.
"""

import numpy as np
import scipy as scp
import matplotlib.pyplot as plt
import scipy.signal as signal
import grid_1D
import configparser
import tqdm
import os
import mpmath as mp
from ast import literal_eval

path = os.path.dirname(os.path.abspath(__file__)) + "/"
config = configparser.ConfigParser()
config.read(path + "config.ini")
ion = config.get("General", "ion_name")


def formfactor(k, RA, aA, Z):
    if Z > 1:
        if k * np.pi * aA < 250:
            rho0 = -1 / (8 * np.pi * aA**3 * mp.polylog(3, -np.exp(RA / aA)))
            N_terms = config.getint("formfactor", "N_terms")

            Fch1 = (
                4
                * np.pi**2
                * rho0
                * aA**3
                / (k**2 * aA**2 * np.sinh(np.pi * k * aA) ** 2)
                * (np.pi * k * aA * np.cosh(np.pi * k * aA) * np.sin(k * RA) - k * RA * np.sinh(np.pi * k * aA) * np.cos(k * RA))
            )
            Fch2_term = [(-1 if i % 2 == 0 else 1) * i * np.exp(-i * RA / aA) / (i**2 + k**2 * aA**2) for i in range(1, N_terms + 1)]

            return float(Fch1 + 8 * np.pi * rho0 * aA**3 * np.sum(Fch2_term))
        else:
            return 0
    else:
        k2 = k**2
        if k2 > 3:
            return 0
        aa = 1.1867816581938533
        a0 = 0.999871
        a1 = -0.215829
        a2 = 0.509109
        a3 = -0.621597
        a4 = 0.246705
        b0 = 1.01809
        b1 = -0.778974e-1
        b2 = -0.184511e-1
        b3 = 0.289159e-2
        b4 = -0.121585e-3
        if k2 > 1.12:
            return (b0 + b1 * k2 + b2 * k2**2 + b3 * k2**3 + b4 * k2**4) / (1 + aa**2 * k2) ** 2
        else:
            return (a0 + a1 * k2 + a2 * k2**2 + a3 * k2**3 + a4 * k2**4) / (1 + aa**2 * k2) ** 2


def photon_flux(x, k2, RA, aA, Z):
    alphaem = 1 / 137.035999084
    mproton = 0.9382720813
    Q2 = k2 + x**2 * mproton**2
    return Z**2 * alphaem / (np.pi**2) * k2 * (formfactor(np.sqrt(Q2), RA, aA, Z) / Q2) ** 2


def I1_integrand(x1, x2, pt1, qt, th, RA, aA, Z):
    pt12 = pt1 * pt1
    qt2 = qt * qt
    pt22 = pt12 + qt2 - 2 * pt1 * qt * np.cos(th)
    return photon_flux(x1, pt12, RA, aA, Z) * photon_flux(x2, pt22, RA, aA, Z)


def I1_eval(X, RA, aA, Z, bornsup):
    x1, x2, qt = X

    def func(lnpt1, theta):
        pt1 = np.exp(lnpt1)
        return pt1**2 * I1_integrand(x1, x2, pt1, qt, theta, RA, aA, Z)

    result = scp.integrate.dblquad(func, 0, 2 * np.pi, -np.inf, bornsup)
    return result[0]


def I1_eval_old(X, RA, aA, Z):
    x1, x2, qt = X
    N_theta = 500
    N_pt = 1000
    TH1 = np.linspace(0, 2 * np.pi, N_theta)
    PT1_2 = np.logspace(-3, 1, N_pt - 200)
    PT1_1 = np.logspace(-5, -3, 200, endpoint=False)
    PT1 = np.append(PT1_1, PT1_2)
    grid_I1_th = grid_1D.grid1D(ion, N_theta)
    grid_I1_th.set_axis(TH1)
    for i, th1 in enumerate(TH1):
        grid_I1_pt = grid_1D.grid1D(ion, N_pt)
        grid_I1_pt.set_axis(PT1)
        for j, pt1 in enumerate(PT1):
            grid_I1_pt.values[j] = pt1 * I1_integrand(x1, x2, pt1, qt, th1, RA, aA, Z)
        grid_I1_th.values[i] = grid_peak_integration(grid_I1_pt)
    I = scp.integrate.quad(lambda th1: grid_I1_th.evaluate(th1), 0, 2 * np.pi, full_output=1)

    # if the integration didn't work (beacause too much oscilation probably), try other method
    if I[2]["last"] == 50:
        res = grid_peak_integration(grid_I1_th, plot=False)
    else:
        res = I[0]

    return res


def I2_integrand(x1, x2, pt1, qt, th, RA, aA, Z):
    pt12 = pt1 * pt1
    qt2 = qt * qt
    pt22 = pt12 + qt2 - 2 * pt1 * qt * np.cos(th)
    pt1pt2 = pt1 * qt * np.cos(th) - pt12
    N = photon_flux(x1, pt12, RA, aA, Z) * photon_flux(x2, pt22, RA, aA, Z)
    A = 0
    if pt12 * pt22 != 0:
        A = pt1pt2 * pt1pt2 / (pt12 * pt22)
    return (2 * A - 1) * N


def I2_eval(X, RA, aA, Z, bornsup):
    x1, x2, qt = X

    def func(lnpt1, theta):
        pt1 = np.exp(lnpt1)
        return pt1**2 * I2_integrand(x1, x2, pt1, qt, theta, RA, aA, Z)

    result = scp.integrate.dblquad(func, 0, 2 * np.pi, -np.inf, bornsup)
    return result[0]


def I3_integrand(x1, x2, pt1, qt, th, RA, aA, Z):
    qt2 = qt * qt
    pt12 = pt1 * pt1
    cth = np.cos(th)
    pt22 = pt12 + qt2 - 2 * pt1 * qt * cth
    N = photon_flux(x1, pt12, RA, aA, Z) * photon_flux(x2, pt22, RA, aA, Z)
    A = 0
    B = 0
    if qt2 * pt22 != 0:
        A = (qt2 - qt * pt1 * cth) ** 2 / (qt2 * pt22)
    if pt12 * qt2 != 0:
        B = (pt1 * qt * cth) ** 2 / (pt12 * qt2)
    return (2 * (A + B) - 2) * N


def I3_eval(X, RA, aA, Z, bornsup):
    x1, x2, qt = X

    def func(lnpt1, theta):
        pt1 = np.exp(lnpt1)
        return pt1**2 * I3_integrand(x1, x2, pt1, qt, theta, RA, aA, Z)

    result = scp.integrate.dblquad(func, 0, 2 * np.pi, -np.inf, bornsup)
    return result[0]


def I4_integrand(x1, x2, pt1, qt, th, RA, aA, Z):
    qt2 = qt * qt
    pt12 = pt1 * pt1
    cth = np.cos(th)
    pt22 = pt12 + qt2 - 2 * pt1 * qt * cth
    if pt22 < 0:
        pt22 = 0
    pt2 = np.sqrt(pt22)
    N = photon_flux(x1, pt12, RA, aA, Z) * photon_flux(x2, pt22, RA, aA, Z)
    A = 0
    B = 0
    if qt2 * pt1 * pt2 != 0:
        A = 2 * (qt2 - qt * pt1 * cth) * (pt1 * qt * cth) / (qt2 * pt1 * pt2)
    if pt1 * pt2 != 0:
        B = (pt1 * qt * cth - pt12) / (pt1 * pt2)
    return (2 * (A - B) ** 2 - 1) * N


def I4_eval(X, RA, aA, Z, bornsup):
    x1, x2, qt = X

    def func(lnpt1, theta):
        pt1 = np.exp(lnpt1)
        return pt1**2 * I4_integrand(x1, x2, pt1, qt, theta, RA, aA, Z)

    result = scp.integrate.dblquad(func, 0, 2 * np.pi, -np.inf, bornsup)
    return result[0]


def main():
    return 0


if __name__ == "__main__":
    main()
