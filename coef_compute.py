
"""
This program computes coefficients of the azimuthal distribution of a lepton pair in the case of a photon-photon collision.
The analytical expresion are give in arXiv:1307.3417v1 eq. (91) to (97)
Author: Nicolas
Date: December 2023
To do:
    - Implement grid calculation/interpolation
    - Implement other coefficients
    - Tune grid size to better performance/precision
    - link it with kt_smearing
"""



import numpy as np
import pickle
import scipy as scp
import mpmath as mp
import time
import matplotlib.pyplot as plt
import sys
import special_function as sf
import scipy.signal as signal
import os
import tqdm
from multiprocessing import Pool

class grid:
    """
    Represents a grid for interpolation.

    Attributes:
        ion (str): The ion associated with the grid.
        dim (int): The number of dimensions of the grid.
        length (int): Number of points the axis.
        name (str, optional): The name of the grid.
        function (function, optional): The function associated with the grid. under the form f(x,R,A,Z)
        axis (ndarray): The values of the grid axis.
        values (ndarray): The values of the grid.
        interpolated (bool): Indicates if the grid has been interpolated.
    """

    def __init__(self, ion: str, dim: int, lengths: int or list, name: str = None, function = None):
        """
        Initializes a new instance of the grid class.

        Args:
            ion (str): The ion associated with the grid.
            dim (int): The number of dimensions of the grid.
            lengths (tupple): The length of each dimension.
            name (str, optional): The name of the grid.
            function (function, optional): The function associated with the grid, under the form f(x,R,A,Z)
        """
        self.ion = ion
        self.RA = RA_dict[ion]/0.197
        self.aA = aA_dict[ion]/0.197
        self.Z = Z_dict[ion]
        self.dim = dim
        self.lengths = lengths
       
        self.name = name
        self.function = function
        if self.dim == 1 and type(lengths) == int:
            self.lengths = [self.lengths]

        if len(self.lengths) != self.dim:
            raise ValueError("lenghts does not match dimension of grid :",dim,"lengths =",self.lengths.shape)
        
        self.axis = [np.zeros((dim, self.lengths[i])) for i in range(dim)]
        self.values = np.zeros(self.lengths)
        self.interpolated = False

    def set_axis(self, i: int, values_axis: list):
        """
        Sets the values of the grid axis at the specified dimension.

        Args:
            i (int): The dimension index.
            values_axis (list): The values of the grid axis.
        
        Raises:
            ValueError: If the dimension index is out of range or the length of values does not match grid length.
        """
        if i > self.dim:
            raise ValueError("Axis index out of range")
        if len(values_axis) != self.lengths[i]:
            raise ValueError("Length of values does not match grid length")
        self.axis[i] = values_axis

    def enlarge_axis(self, i: int, new_axis_values: list):
        """
        Extand the grid axis at the specified dimension.

        Args:
            i (int): The dimension index.
            values_axis (list): The values of the grid axis.
        
        Raises:
            ValueError: If the dimension index is out of range or the length of values is not greater than grid length.
        """
        if i > self.dim:
            raise ValueError("Axis index out of range")
        
        self.interpolated = False
        
        #Create the new axis
        if new_axis_values[0] < self.axis[i][0]:
            self.axis[i] = np.append(new_axis_values,self.axis[i])
        elif new_axis_values[-1] > self.axis[i][-1]:
            self.axis[i] = np.append(self.axis[i],new_axis_values)
        else:
            raise ValueError("You are not supposed to be here")
        
         #Compute the new length of the axis
        self.lengths[i] = len(self.axis[i])

        #Compute the new values of the grid
        new_grid = np.zeros(self.lengths)
        print("INFO "+self.name+": Computing new grid values, may take a while")
        pbar = tqdm.tqdm(total=np.prod(self.lengths))
        for index in np.ndindex(tuple(self.lengths)):
            try :
                #If the index is in the old grid, copy the value
                new_grid[index] = self.values[index]
                pbar.update(1)
            except:
                x = np.array([self.axis[i][index[i]] for i in range(self.dim)])
                new_grid[index] = float(self.function(x,self.RA,self.aA,self.Z))
                pbar.update(1)
        self.values = new_grid
        if self.saved:
            self.write(self.path)
        print("INFO "+self.name+": New grid values computed")        

    
    def write(self, path: str):
        """
        Writes the grid object to a pickle file.

        Args:
            path (str): The path to the directory where the file will be saved.
        """
        if path[-1] != "/" and len(path) > 0:
            path = path + "/"
        self.path = path
        self.saved = True
        with open(f"{path}grid_{self.name}_{self.ion}.pkl", "wb") as file:
            pickle.dump(self, file)

    def interpolate(self):
        """
        Interpolates the grid values.
        User does not need to call this function, it is called automatically when the grid is evaluated.

        Raises:
            ValueError: If the grid dimension is not supported.
        """
        if self.dim == 1:
            self.interpolatefun = scp.interpolate.CubicSpline(self.axis[0], self.values, extrapolate=False)
        elif self.dim == 3:
            self.interpolatefun = scp.interpolate.RegularGridInterpolator(self.axis, self.values, bounds_error=False, fill_value=None)
        else:
            raise ValueError("Interpolation is only supported for 1-dimensional and 3-dimensional grids")

        self.interpolated = True

    def evaluate(self, x):
        """
        Evaluates the interpolated grid at the specified value.

        Args:
            x: The value at which to evaluate the grid.

        Returns:
            The interpolated value at the specified value.
        """
        if self.interpolated == False:
            self.interpolate()
        if self.dim == 1:
            if x > self.axis[0][-1] or x < self.axis[0][0]:
                if self.function != None:
                    #If the value is out of range, enlarge the grid, add 5 values per new decade
                    print("WARNING "+self.name+": Value out of range, augment grid size")
                    pointperdecade = int(self.lengths[0]/np.log10(self.axis[0][-1]/self.axis[0][0]))
                    if x > self.axis[0][-1]:
                        new_max = 10**np.ceil(np.log10(x))
                        numbernewdecade = int(np.log10(new_max/self.axis[0][-1]))
                        additonal_points = np.logspace(np.log10(self.axis[0][-1]),np.log10(new_max),numbernewdecade*pointperdecade+1)[1:]
                        print("INFO "+self.name+": Adding "+str(len(additonal_points))+" points, max updated to "+str(new_max))
                    elif x < self.axis[0][0]:
                        new_min = 10**np.floor(np.log10(x))
                        numbernewdecade = int(np.log10(self.axis[0][0]/new_min))
                        additonal_points = np.logspace(np.log10(new_min),np.log10(self.axis[0][0]),numbernewdecade*pointperdecade+1)[:-1]
                        print("INFO "+self.name+": Adding "+str(len(additonal_points))+" points, min updated to "+str(new_min))
                    else:
                        raise ValueError("You are not supposed to be here")
                    
                    self.enlarge_axis(0,additonal_points)
                    return self.evaluate(x)
                else:
                    #If the value is out of range and no function is associated to the gird, raise an error
                    raise ValueError("ERROR "+self.name+": Value out of range")
                
        return float(self.interpolatefun(x))
    
        



path = "/home/nicolas/Documents/ARPE_2023_2024/phi_smearing/"

ion = "Pb"
RA_dict = {"Pb":6.62,"Ca":4.43,"C":3.43,"Al":2.47}
aA_dict = {"Pb":0.546,"Ca":0.45,"C":0.35,"Al":0.3}
Z_dict = {"Pb":82,"Ca":20,"C":6,"Al":13}

RA_Pb = RA_dict["Pb"]/0.197
aA_Pb = aA_dict["Pb"]/0.197
Z_Pb = Z_dict["Pb"]
num_core = 4
rho0_Pb = -1/(8*np.pi*aA_Pb**3*mp.polylog(3,-np.exp(RA_Pb/aA_Pb)))

alphaem = 1/137.035999084
mproton = 0.9382720813

M1 = 511e-6
M2 = 511e-6

def rho(r,RA,aA):
    rho0 = rho0_Pb
    return rho0/(1+mp.exp((r-RA)/aA))

def formfactor(k, RA, aA, Z):
    """
    Computes the form factorn, using the charge form factor (from gammaUPC)

    Args:
        k (float): The momentum transfer.
        RA (float): The parameter RA.
        aA (float): The parameter aA.
        Z (int): The atomic number.

    Returns:
        float: The computed form factor.
    """
    global grid_formfactor

    # Check if the grid exists or is stored as a file
    if "grid_formfactor" in globals():
        return grid_formfactor.evaluate(k)
    elif os.path.exists(f"{path}grid_FORMFACTOR_{ion}.pkl"):
        print("INFO FORMFACTOR: Grid found, loading grid")
        with open(f"{path}grid_FORMFACTOR_{ion}.pkl", "rb") as file:
            grid_formfactor = pickle.load(file)
            print("INFO FORMFACTOR: Grid loaded")
            return grid_formfactor.evaluate(k)
    else:
        # If no grid is found, compute it and store it
        print("INFO FORMFACTOR: No grid found, computing grid")
        Npoints = 10000
        logkmin = -3
        logkmax = 2
        grid_formfactor = grid(ion,1,Npoints,"FORMFACTOR",formfactor_eval)
        K = np.logspace(logkmin,logkmax,Npoints)
        grid_formfactor.set_axis(0,K)
        pbar = tqdm.tqdm(total=Npoints)
        for i,k in enumerate(K):
            grid_formfactor.values[i] = formfactor_eval([k],RA,aA,Z)
            pbar.update(1)
        grid_formfactor.write(path)
        return grid_formfactor.evaluate(k)
        
def formfactor_eval(k,RA,aA,Z):
    k = k[0]
    A = sf.lerch_hankel(-mp.exp(RA/aA),2,1-1j*k*aA)
    B = sf.lerch_hankel(-mp.exp(RA/aA),2,1+1j*k*aA)
    val = -2j*aA**2*mp.exp(RA/aA)*rho0_Pb*(A-B)*np.pi/k
    if mp.im(val) != 0:
        raise ValueError("Value should be real")
    return mp.re(val)


def photon_flux(x,k2,RA,aA,Z,usegrid = True):
    Q2 = k2+x**2*mproton**2
    if usegrid:
        return Z**2*alphaem/(x*np.pi**2)*k2*formfactor(np.sqrt(Q2),RA,aA,Z)**2/Q2**2
    else:
        return Z**2*alphaem/(x*np.pi**2)*k2*formfactor_eval(np.sqrt(Q2),RA,aA,Z)**2/Q2**2


def Fgammagamma_integrand(x1,x2,pt12,pt22,RA,aA,Z):
    return photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)

def Fgammagamma_eval(X,RA,aA,Z):
    x1,x2,qt = X
    if qt < 0.05:
        N_theta = 10
    if 0.05 <= qt < 0.5:
        N_theta = 50
    if 0.5 <= qt <= 1:
        N_theta = 200
    N_pt = 2000
    qt2 = qt*qt
    TH1 = np.linspace(0,2*np.pi,N_theta)
    PT1_2 = np.logspace(-1,np.log10(500),N_pt-50)
    PT1_1 = np.logspace(-5,-1,50,endpoint=False)
    PT1 = np.append(PT1_1,PT1_2)
    grid_Fgammagamma_th = grid(ion,1,N_theta)
    grid_Fgammagamma_th.set_axis(0,TH1)
    for i,th1 in enumerate(TH1):
        grid_Fgammagamma_pt = grid(ion,1,N_pt)
        grid_Fgammagamma_pt.set_axis(0,PT1)
        for j,pt1 in enumerate(PT1):
            grid_Fgammagamma_pt.values[j] = pt1*Fgammagamma_integrand(x1,x2,pt1*pt1,qt2+pt1*pt1 -2*qt*pt1*np.cos(th1),RA,aA,Z)
        grid_Fgammagamma_th.values[i] = grid_peak_integration(grid_Fgammagamma_pt)
    I = scp.integrate.quad(lambda th1: grid_Fgammagamma_th.evaluate(th1),0,2*np.pi,full_output=1)

    #if the integration didn't work (beacuse too mucj oscilation probably), try other method
    if I[2]["last"] == 50:
        res = grid_peak_integration(grid_Fgammagamma_th,plot=True)
    else: res = I[0]
    
    print(res,I[0])
    return res

def Fgammagamma(x1,x2,qt,RA,aA,Z):
    global grid_Fgammagamma
    if "grid_Fgammagamma" in globals():
        return grid_Fgammagamma.evaluate([x1,x2,qt])
    elif os.path.exists(f"{path}grid_Fgammagamma_{ion}.pkl"):
        print("INFO Fgammagamma: Grid found, loading grid")
        with open(f"{path}grid_Fgammagamma_{ion}.pkl", "rb") as file:
            grid_Fgammagamma = pickle.load(file)
            print("INFO Fgammagamma: Grid loaded")
            return grid_Fgammagamma.evaluate([x1,x2,qt])
    else:
        print("INFO Fgammagamma: No grid found, computing grid")
        Npointsx1 = 10
        Npointsx2 = 10
        Npointsqt = 50
        logx1min = -3
        logx1max = 0
        logx2min = -3
        logx2max = 0
        logqtmin = -3
        logqtmax = 0
        grid_Fgammagamma = grid(ion,3,[Npointsx1,Npointsx2,Npointsqt],"Fgammagamma",Fgammagamma_eval)

        #Fonction for multiprocessing
        global Fgammagamma_multiprocess
        
        def Fgammagamma_multiprocess(X):
            i,j,k = X
            x1 = grid_Fgammagamma.axis[0][i]
            x2 = grid_Fgammagamma.axis[1][j]
            qt = grid_Fgammagamma.axis[2][k]
            res = Fgammagamma_eval([x1,x2,qt],RA,aA,Z)
            return i,j,k,res
        
        X1 = np.logspace(logx1min,logx1max,Npointsx1)
        X2 = np.logspace(logx2min,logx2max,Npointsx2)
        QT = np.logspace(logqtmin,logqtmax,Npointsqt)
        index = [(i,j,k) for i in range(Npointsx1) for j in range(Npointsx2) for k in range(Npointsqt)]
        grid_Fgammagamma.set_axis(0,X1)
        grid_Fgammagamma.set_axis(1,X2)
        grid_Fgammagamma.set_axis(2,QT)
        
        #Multiprocessing
        pool = Pool(num_core)
        res = list(tqdm.tqdm(pool.imap(Fgammagamma_multiprocess, index), total=Npointsx1*Npointsx2*Npointsqt, desc="Computing", position=1, leave=True))
        pool.close()
        pool.join()
        for i,j,k,r in res:
            grid_Fgammagamma.values[i,j,k] = r
        grid_Fgammagamma.write(path)
        print("INFO Fgammagamma: Grid computed")
        
        return grid_Fgammagamma.evaluate([x1,x2,qt])




#WIP: I have to implement the other coefficients
""" def qt4Ngammagamma_integrand(x1,x2,pt1,pt2,RA,aA,Z):
    #(2(pt1@pt2)**2 - (pt1**2*pt2**2)*N)/(x1*x2)
    pt12 = pt1[0]*pt1[0]+pt1[1]*pt1[1]
    pt22 = pt2[0]*pt2[0]+pt2[1]*pt2[1]
    pt1pt2 = pt1[0]*pt2[0]+pt1[1]*pt2[1]
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    return (2*pt1pt2*pt1pt2 - pt12*pt22) * N



def qt2Hgammagamma_integrand(x1,x2,pt1,pt2,RA,aA,Z):
    #N*((2(h@pt1)**2 + 2(h@pt2)**2 -pt1**2 - pt2**2)))
    qt = pt1+pt2
    qt2 = qt[0]*qt[0]+qt[1]*qt[1]
    pt12 = pt1[0]*pt1[0]+pt1[1]*pt1[1]
    pt22 = pt2[0]*pt2[0]+pt2[1]*pt2[1]
    pt1pt2 = pt1[0]*pt2[0]+pt1[1]*pt2[1]
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    return (2/qt2*(pt12*pt12 + pt22*pt22 + 2*pt1pt2*pt1pt2 +2*pt1pt2*(pt12+pt22))-pt12-pt22) * N


def qt4Lgammagamma_integrand(x1,x2,pt1,pt2,RA,aA,Z):
    #N*(pt1**2*pt2**2)
    pt12 = pt1[0]*pt1[0]+pt1[1]*pt1[1]
    pt22 = pt2[0]*pt2[0]+pt2[1]*pt2[1]
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    return pt12*pt22*N


def qt4Igammagamma_integrand(x1,x2,pt1,pt2,RA,aA,Z):
    #N*(2(h@pt1)(h@pt2) - (pt1@pt2))**2)
    qt = pt1+pt2
    qt2 = qt[0]*qt[0]+qt[1]*qt[1]
    pt12 = pt1[0]*pt1[0]+pt1[1]*pt1[1]
    pt22 = pt2[0]*pt2[0]+pt2[1]*pt2[1]
    pt1pt2 = pt1[0]*pt2[0]+pt1[1]*pt2[1]
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    return (2/qt2*(pt12*pt22 + pt1pt2*(pt12+pt22)+pt1pt2*pt1pt2)-pt1pt2)**2 * N """





""" def integ_peak(f,X,print_progress = False):
    Y = np.zeros_like(X)
    for i,x in enumerate(X):
        Y[i] = f(x)
    peaks = signal.find_peaks(-Y,distance=1)[0]
    peaks = np.append([0],peaks)
    peaks = np.append(peaks,[len(X)-1])
    plt.plot(X,Y)
    plt.plot(X[peaks],Y[peaks],marker="x",linestyle="None")
    I = 0
    for i in range(len(peaks)-1):
        p1 = peaks[i]
        p2 = peaks[i+1]
        res = mp.quad(f,[X[p1],X[p2]])
        I += res
        if print_progress:
            progress((i+1)/len(peaks)*100)
        plt.plot()
    return I """

def grid_peak_integration(grid,plot=False):
    if grid.dim != 1:
        raise ValueError("Grid dimension must be 1")
    X = grid.axis[0]
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
x1 = 0.5
x2 = 0.5
QT = np.logspace(-5,0,30)
res = np.zeros_like(QT)
Fgammagamma(x1,x2,0.1,RA_Pb,aA_Pb,Z_Pb)
print(grid_Fgammagamma.values)
for i,qt in enumerate(QT):
    res[i] = Fgammagamma(x1,x2,qt,RA_Pb,aA_Pb,Z_Pb)
    print(res[i])
    
plt.plot(QT,res)
plt.show()
    
