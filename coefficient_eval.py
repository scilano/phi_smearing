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
import mpmath as mp

config = configparser.ConfigParser()
config.read('/afs/cern.ch/user/n/ncrepet/work/scripts/phi_smearing/config.ini')
ion = config.get("General","ion_name")


def formfactor(k, RA, aA, Z):
    if k*np.pi*aA < 250:
        rho0 = -1/(8*np.pi*aA**3*mp.polylog(3,-np.exp(RA/aA)))
        N_terms = config.getint('formfactor', 'N_terms')
        
        Fch1 = 4*np.pi**2*rho0*aA**3/(k**2*aA**2*np.sinh(np.pi*k*aA)**2)*(np.pi*k*aA*np.cosh(np.pi*k*aA)*np.sin(k*RA)-k*RA*np.sinh(np.pi*k*aA)*np.cos(k*RA))
        Fch2_term = [(-1 if i%2==0 else 1)*i*np.exp(-i*RA/aA)/(i**2+k**2*aA**2) for i in range(1,N_terms+1)]
        
        return float(Fch1 + 8*np.pi*rho0*aA**3*np.sum(Fch2_term))
    else : return 0

def photon_flux(x,k2,RA,aA,Z):
    alphaem = 1/137.035999084
    mproton = 0.9382720813
    Q2 = k2+x**2*mproton**2
    return Z**2*alphaem/(np.pi**2)*k2*(formfactor(np.sqrt(Q2),RA,aA,Z)/Q2)**2

def I1_integrand(x1,x2,pt1,qt,th,RA,aA,Z):
    pt12 = pt1*pt1
    qt2 = qt*qt
    pt22 = pt12 + qt2 - 2*pt1*qt*np.cos(th)
    return photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)

def I1_eval(X,RA,aA,Z):
    x1,x2,qt = X
    if qt < 0.05:
        N_theta = 10
    if 0.05 <= qt < 0.5:
        N_theta = 50
    if 0.5 <= qt:
        N_theta = 200
    
    N_pt = 3000
    TH1 = np.linspace(0,2*np.pi,N_theta)
    PT1_2 = np.logspace(-1,2,N_pt-200)
    PT1_1 = np.logspace(-5,-1,200,endpoint=False)
    PT1 = np.append(PT1_1,PT1_2)
    grid_I1_th = grid_1D.grid1D(ion,N_theta)
    grid_I1_th.set_axis(TH1)
    for i,th1 in enumerate(TH1):
        grid_I1_pt = grid_1D.grid1D(ion,N_pt)
        grid_I1_pt.set_axis(PT1)
        for j,pt1 in enumerate(PT1):
            grid_I1_pt.values[j] = pt1*I1_integrand(x1,x2,pt1,qt,th1,RA,aA,Z)
        grid_I1_th.values[i] = grid_peak_integration(grid_I1_pt)
    I = scp.integrate.quad(lambda th1: grid_I1_th.evaluate(th1),0,2*np.pi,full_output=1)

    #if the integration didn't work (beacause too much oscilation probably), try other method
    if I[2]["last"] == 50:
        res = grid_peak_integration(grid_I1_th,plot=False)
    else: res = I[0]
    
    return res

    
def I2_integrand(x1,x2,pt1,qt,th,RA,aA,Z):    
    pt12 = pt1*pt1
    qt2 = qt*qt
    pt22 = pt12 + qt2 - 2*pt1*qt*np.cos(th)
    pt1pt2 = pt1*qt*np.cos(th)- pt12 
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    A = 0
    if pt12*pt22 != 0:
        A = pt1pt2*pt1pt2/(pt12*pt22)
    return (2*A-1)*N


def I2_eval(X,RA,aA,Z):
    x1,x2,qt = X
    if qt < 0.05:
        N_theta = 10
    if 0.05 <= qt < 0.5:
        N_theta = 50
    if 0.5 <= qt:
        N_theta = 200
    N_pt = 3000
    
    TH1 = np.linspace(0,2*np.pi,N_theta)
    PT1_2 = np.logspace(-1,2,N_pt-200)
    PT1_1 = np.logspace(-5,-1,200,endpoint=False)
    PT1 = np.append(PT1_1,PT1_2)
    grid_I2_th = grid_1D.grid1D(ion,N_theta)
    grid_I2_th.set_axis(TH1)
    for i,th1 in enumerate(TH1):
        grid_I2_pt = grid_1D.grid1D(ion,N_pt)
        grid_I2_pt.set_axis(PT1)
        for j,pt1 in enumerate(PT1):
            grid_I2_pt.values[j] = pt1*I2_integrand(x1,x2,pt1,qt,th1,RA,aA,Z)
        grid_I2_th.values[i] = grid_peak_integration(grid_I2_pt)
    I = scp.integrate.quad(lambda th1: grid_I2_th.evaluate(th1),0,2*np.pi,full_output=1)
    
    #if the integration didn't work (beacause too much oscilation probably), try other method
    if I[2]["last"] == 50:
        res = grid_peak_integration(grid_I2_th)
    else: res = I[0]
    return res

    
def I3_integrand(x1,x2,pt1,qt,th,RA,aA,Z):
    qt2 = qt*qt
    pt12 = pt1*pt1
    cth = np.cos(th)
    pt22 = pt12 + qt2 - 2*pt1*qt*cth
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    A = 0
    B = 0
    if (qt2*pt22 != 0):
        A = (qt2-qt*pt1*cth)**2/(qt2*pt22)
    if (pt12*qt2 != 0):
        B = (pt1*qt*cth)**2/(pt12*qt2)
    return (2*(A + B)-2)*N
    
def I3_eval(X,RA,aA,Z):
    x1,x2,qt = X
    if qt < 0.05:
        N_theta = 10
    if 0.05 <= qt < 0.5:
        N_theta = 50
    if 0.5 <= qt:
        N_theta = 200
    N_pt = 3000
    
    TH1 = np.linspace(0,2*np.pi,N_theta)
    PT1_2 = np.logspace(-1,2,N_pt-200)
    PT1_1 = np.logspace(-5,-1,200,endpoint=False)
    PT1 = np.append(PT1_1,PT1_2)
    grid_I3_th = grid_1D.grid1D(ion,N_theta)
    grid_I3_th.set_axis(TH1)
    for i,th1 in enumerate(TH1):
        grid_I3_pt = grid_1D.grid1D(ion,N_pt)
        grid_I3_pt.set_axis(PT1)
        for j,pt1 in enumerate(PT1):
            grid_I3_pt.values[j] = pt1*I3_integrand(x1,x2,pt1,qt,th1,RA,aA,Z)
        grid_I3_th.values[i] = grid_peak_integration(grid_I3_pt)
    I = scp.integrate.quad(lambda th1: grid_I3_th.evaluate(th1),0,2*np.pi,full_output=1)
    
    #if the integration didn't work (beacause too much oscilation probably), try other method
    if I[2]["last"] == 50:
        res = grid_peak_integration(grid_I3_th)
    else: res = I[0]
    return res


    
def I4_integrand(x1,x2,pt1,qt,th,RA,aA,Z):
    qt2 = qt*qt
    pt12 = pt1*pt1
    cth = np.cos(th)
    pt22 = pt12 + qt2 - 2*pt1*qt*cth
    pt2 = np.sqrt(pt22)
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    A = 0
    B = 0
    if qt2*pt1*pt2 != 0:
        A = 2*(qt2-qt*pt1*cth)*(pt1*qt*cth)/(qt2*pt1*pt2)
    if pt1*pt2 != 0:
        B = (pt1*qt*cth - pt12)/(pt1*pt2)
    return (2*(A-B)**2-1)*N

def I4_eval(X,RA,aA,Z):
    x1,x2,qt = X
    if qt < 0.05:
        N_theta = 10
    if 0.05 <= qt < 0.5:
        N_theta = 50
    if 0.5 <= qt:
        N_theta = 200
    N_pt = 3000
    
    TH1 = np.linspace(0,2*np.pi,N_theta)
    PT1_2 = np.logspace(-1,2,N_pt-200)
    PT1_1 = np.logspace(-5,-1,200,endpoint=False)
    PT1 = np.append(PT1_1,PT1_2)
    grid_I4_th = grid_1D.grid1D(ion,N_theta)
    grid_I4_th.set_axis(TH1)
    for i,th1 in enumerate(TH1):
        grid_I4_pt = grid_1D.grid1D(ion,N_pt)
        grid_I4_pt.set_axis(PT1)
        for j,pt1 in enumerate(PT1):
            grid_I4_pt.values[j] = pt1*I4_integrand(x1,x2,pt1,qt,th1,RA,aA,Z)
        grid_I4_th.values[i] = grid_peak_integration(grid_I4_pt)
    I = scp.integrate.quad(lambda th1: grid_I4_th.evaluate(th1),0,2*np.pi,full_output=1)
    
    #if the integration didn't work (because too much oscilation probably), try other method
    if I[2]["last"] == 50:
        res = grid_peak_integration(grid_I4_th)
    else: res = I[0]
    return res


def grid_peak_integration(grid,plot=False):
    X = grid.axis
    Y = grid.values
    #Detect peaks
    peaks = signal.find_peaks(-Y,distance=1)[0]
    peaks = np.append([0],peaks)
    peaks = np.append(peaks,[len(X)-1])
    if plot:
        plt.plot(X,Y)
        plt.plot(X[peaks],Y[peaks],marker="x",linestyle="None")
        plt.semilogy()
        plt.show()
    I = 0
    e=0
    #Integrate between each peak
    for i in range(len(peaks)-1):
        p1 = peaks[i]
        p2 = peaks[i+1]
        f = lambda x: grid.evaluate(x)
        res = scp.integrate.quad(f,X[p1],X[p2])
        I += res[0]
    return I

def main():
    L =[0.0027701355532506783,0.002371241508120984,0.01427330393200141,0.22506577178010387,0.000511,0.5124664359554223,6.38,0.535,79]
    x1,x2,qt,Kt,ml,m_pair,RA,aA,Z =L
    qt = 1e-3
    qt2 = qt*qt
    PT1 = np.logspace(-6,1,1500)
    prod = np.zeros(len(PT1))
    L = np.zeros(len(PT1))
    H = np.zeros((len(PT1)))
    cth = np.cos(np.pi/8)
    
    for i,pt1 in enumerate(PT1):
        pt12 = pt1*pt1
        pt22 = pt12 + qt2 - 2*pt1*qt*cth
        L[i] = I4_integrand(x1,x2,pt1,qt,np.pi/5,RA,aA,Z)
        H[i] = I3_integrand(x1,x2,pt1,qt,np.pi/5,RA,aA,Z)
    plt.plot(PT1,np.abs(L),label="L")
    plt.plot(PT1,np.abs(H),label="H")
    plt.legend()
    plt.loglog()
    plt.show()

    
    return 0




if __name__ == "__main__":
    main()