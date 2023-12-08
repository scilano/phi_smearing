
"""
This program computes coefficients of the azimuthal distribution of a lepton pair in the case of a photon-photon collision.
The analytical expresion are give in arXiv:1307.3417v1 eq. (91) to (97)
Author: Nicolas
Date: December 2023
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


class grid:
    """
    Represents a grid for interpolation.

    Attributes:
        ion (str): The ion associated with the grid.
        dim (int): The number of dimensions of the grid.
        length (int): Number of points the axis.
        name (str, optional): The name of the grid.
        function (function, optional): The function associated with the grid.
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
        Enlarges the values of the grid axis at the specified dimension.

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
            self.values[i] = np.append(new_axis_values,self.values)
        elif new_axis_values[-1] > self.axis[i][-1]:
            self.axis[i] = np.append(self.axis[i],new_axis_values)
        else:
            raise ValueError("You are not supposed to be here")
        
         #Compute the new length of the axis
        self.lengths[i] = len(self.axis[i])

        #Compute the new values of the grid
        new_grid = np.zeros(self.lengths)
        print("INFO "+self.name+": Computing new grid values, may take a while")

        for index in np.ndindex(tuple(self.lengths)):
            try :
                #If the index is in the old grid, copy the value
                new_grid[index] = self.values[index]
            except:
                x = np.array([self.axis[i][index[i]] for i in range(self.dim)])
                new_grid[index] = float(self.function(x,self.RA,self.aA,self.Z))
            progress(index[0]/self.lengths[0]*100)
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
        else:
            raise ValueError("Interpolation is only supported for 1-dimensional grids")

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
RA_dict = {"Pb":6.62,"Ca":4.43,"C":3.43,"Al":2.47,"H":0.0}
aA_dict = {"Pb":0.546,"Ca":0.45,"C":0.35,"Al":0.3,"H":0.0}
Z_dict = {"Pb":82,"Ca":20,"C":6,"Al":13,"H":1}

RA_Pb = RA_dict["Pb"]/0.197
aA_Pb = aA_dict["Pb"]/0.197
Z_Pb = 82

rho0_Pb = -1/(8*np.pi*aA_Pb**3*mp.polylog(3,-np.exp(RA_Pb/aA_Pb)))

alphaem = 1/137.035999084
mproton = 0.9382720813

M1 = 511e-6
M2 = 511e-6

def progress(percent):
    bar_length = 30
    sys.stdout.write('\r')
    sys.stdout.write("Completed: [{:{}}] {:>3}%"
                     .format('='*int(percent/(100.0/bar_length)),
                             bar_length, int(percent)))
    sys.stdout.flush()
    if percent > 99.9:
        print("\n")

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
        Npoints = 1000
        logkmin = -3
        logkmax = 2
        grid_formfactor = grid(ion,1,Npoints,"FORMFACTOR",formfactor_eval)
        K = np.logspace(logkmin,logkmax,Npoints)
        grid_formfactor.set_axis(0,K)
        for i,k in enumerate(K):
            grid_formfactor.values[i] = formfactor_eval([k],RA,aA,Z)
            progress((i+1)/len(K)*100)
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

def Fgammagamma_eval(x1,x2,qt,RA,aA,Z):
    qt2 = qt*qt
    TH1 = np.linspace(0,2*np.pi,1000)
    PT1 = np.linspace(0,20,1000)
    grid_Fgammagamma_th = grid(ion,1,1000)
    grid_Fgammagamma_th.set_axis(0,TH1)
    for i,th1 in enumerate(TH1):
        grid_Fgammagamma_pt = grid(ion,1,1000)
        grid_Fgammagamma_pt.set_axis(0,PT1)
        for j,pt1 in enumerate(PT1):
            grid_Fgammagamma_pt.values[j] = pt1*Fgammagamma_integrand(x1,x2,pt1*pt1,qt2+pt1*pt1 -2*qt*pt1*np.cos(th1),RA,aA,Z)
        if i == 1:
            grid_Fgammagamma_th.values[i] = grid_peak_integration(grid_Fgammagamma_pt,plot=True)
        else:
            grid_Fgammagamma_th.values[i] = grid_peak_integration(grid_Fgammagamma_pt)
        progress((i+1)/len(TH1)*100)
    plt.plot(TH1,grid_Fgammagamma_th.values)
    plt.show()
    return scp.integrate.quad(lambda th1: grid_Fgammagamma_th.evaluate(th1),0,2*np.pi)[0]


def qt4Ngammagamma_integrand(x1,x2,pt1,pt2,RA,aA,Z):
    #(2(pt1@pt2)**2 - (pt1**2*pt2**2)*N)/(x1*x2)
    pt12 = pt1[0]*pt1[0]+pt1[1]*pt1[1]
    pt22 = pt2[0]*pt2[0]+pt2[1]*pt2[1]
    pt1pt2 = pt1[0]*pt2[0]+pt1[1]*pt2[1]
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    return (2*pt1pt2*pt1pt2 - pt12*pt22) * N

def qt4Ngammagamma_eval(x1,x2,qt,RA,aA,Z):
    integrale = mp.quad(lambda pt1x,pt1y: qt4Ngammagamma_integrand(x1,x2,[pt1x,pt1y],[qt[0]-pt1x,qt[1]-pt1y],RA,aA,Z),mp.linspace(-1,1,5),[-1,1],verbose=False,error=False)
    return integrale/(M1*M1*M2*M2)


def qt2Hgammagamma_integrand(x1,x2,pt1,pt2,RA,aA,Z):
    #N*((2(h@pt1)**2 + 2(h@pt2)**2 -pt1**2 - pt2**2)))
    qt = pt1+pt2
    qt2 = qt[0]*qt[0]+qt[1]*qt[1]
    pt12 = pt1[0]*pt1[0]+pt1[1]*pt1[1]
    pt22 = pt2[0]*pt2[0]+pt2[1]*pt2[1]
    pt1pt2 = pt1[0]*pt2[0]+pt1[1]*pt2[1]
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    return (2/qt2*(pt12*pt12 + pt22*pt22 + 2*pt1pt2*pt1pt2 +2*pt1pt2*(pt12+pt22))-pt12-pt22) * N

def qt2Hgammagamma_eval(x1,x2,qt,RA,aA,Z):
    integrale = mp.quad(lambda pt1x,pt1y: qt2Hgammagamma_integrand(x1,x2,[pt1x,pt1y],[qt[0]-pt1x,qt[1]-pt1y],RA,aA,Z),mp.linspace(-1,1,5),[-1,1],verbose=False,error=False)
    return integrale/(M1*M2)

def qt4Lgammagamma_integrand(x1,x2,pt1,pt2,RA,aA,Z):
    #N*(pt1**2*pt2**2)
    pt12 = pt1[0]*pt1[0]+pt1[1]*pt1[1]
    pt22 = pt2[0]*pt2[0]+pt2[1]*pt2[1]
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    return pt12*pt22*N

def qt4Lgammagamma_eval(x1,x2,qt,RA,aA,Z):
    integrale = mp.quad(lambda pt1x,pt1y: qt4Lgammagamma_integrand(x1,x2,[pt1x,pt1y],[qt[0]-pt1x,qt[1]-pt1y],RA,aA,Z),mp.linspace(-1,1,5),[-1,1],verbose=False,error=False)
    return integrale/(M1*M1*M2*M2)

def qt4Igammagamma_integrand(x1,x2,pt1,pt2,RA,aA,Z):
    #N*(2(h@pt1)(h@pt2) - (pt1@pt2))**2)
    qt = pt1+pt2
    qt2 = qt[0]*qt[0]+qt[1]*qt[1]
    pt12 = pt1[0]*pt1[0]+pt1[1]*pt1[1]
    pt22 = pt2[0]*pt2[0]+pt2[1]*pt2[1]
    pt1pt2 = pt1[0]*pt2[0]+pt1[1]*pt2[1]
    N = photon_flux(x1,pt12,RA,aA,Z)*photon_flux(x2,pt22,RA,aA,Z)
    return (2/qt2*(pt12*pt22 + pt1pt2*(pt12+pt22)+pt1pt2*pt1pt2)-pt1pt2)**2 * N

def qt4Igammagamma_eval(x1,x2,qt,RA,aA,Z):
    integrale = mp.quad(lambda pt1x,pt1y: qt4Igammagamma_integrand(x1,x2,[pt1x,pt1y],[qt[0]-pt1x,qt[1]-pt1y],RA,aA,Z),mp.linspace(-1,1,5),[-1,1],verbose=False,error=False)
    return integrale/(M1*M1*M2*M2)



def A1(z,Mratio):
    Mratio2 = Mratio*Mratio
    return 2*(z*z+(1-z)*(1-z)+4*z*(1-z)*(1-Mratio2)*Mratio2)

def Agammagamma(x1,x2,qt,RA,aA,Z,Mratio,z):
    return A1(z,Mratio)*Fgammagamma_eval(x1,x2,qt,RA,aA,Z) - Mratio**4*z*(1-z)*qt4Ngammagamma_eval(x1,x2,qt,RA,aA,Z)

def Bgammagamma(x1,x2,qt,RA,aA,Z,Mratio,z):
    return 4*z*(1-z)*(1-Mratio*Mratio)*qt2Hgammagamma_eval(x1,x2,qt,RA,aA,Z)/(qt[0]*qt[0]+qt[1]*qt[1])

def Cgammagamma(x1,x2,qt,RA,aA,Z,Mratio,z):
    qt2 = qt[0]*qt[0]+qt[1]*qt[1]
    return -z(1-z)*(1-Mratio*Mratio)*(2*qt4Igammagamma_eval(x1,x2,qt,RA,aA,Z)-qt4Lgammagamma_eval(x1,x2,qt,RA,aA,Z))/(qt2*qt2)

def integ_peak(f,X,print_progress = False):
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
    return I

def grid_peak_integration(grid,print_progress = False,plot=False):
    if grid.dim != 1:
        raise ValueError("Grid dimension must be 1")
    X = grid.axis[0]
    Y = grid.values
    peaks = signal.find_peaks(-Y,distance=1)[0]
    peaks = np.append([0],peaks)
    peaks = np.append(peaks,[len(X)-1])
    if plot:
        plt.plot(X,Y)
        plt.plot(X[peaks],Y[peaks],marker="x",linestyle="None")
        plt.semilogy()
        plt.show()
    I = 0
    for i in range(len(peaks)-1):
        p1 = peaks[i]
        p2 = peaks[i+1]
        f = lambda x: grid.evaluate(x)
        res = scp.integrate.quad(f,X[p1],X[p2])[0]
        I += res
        if print_progress:
            progress((i+1)/len(peaks)*100)
    return I


x1,x2 = np.random.rand(2)
th1 = np.random.rand()*2*np.pi
QT =[0.01,0.1,1]
PT = np.linspace(0,100,10000)
res = np.zeros((len(QT),len(PT)))
for i,qt in enumerate(QT):
    for j,pt1 in enumerate(PT):
        res[i,j] = pt1*Fgammagamma_integrand(x1,x2,pt1*pt1,qt*qt+pt1*pt1 -2*qt*pt1*np.cos(th1),RA_Pb,aA_Pb,Z_Pb)
    plt.plot(PT,res[i],label=f"qt = {qt}")
plt.title("x1,x2,th1 = "+str(x1)+","+str(x2)+","+str(th1))
plt.semilogy()
plt.legend()
plt.show()

