
import numpy as np
import pickle
import os
import tqdm
from multiprocessing import Pool
import grid_3D
import coefficient_eval as c_eval
import configparser
import matplotlib.pyplot as plt


config = configparser.ConfigParser()
config.read('/afs/cern.ch/user/n/ncrepet/work/scripts/phi_smearing_0.3/config.ini')
path = config.get('General', 'path')
num_core = config.getint('General', 'num_core')
ion = config.get("General","ion_name")

alphaem = 1/137.035999084
mproton = 0.9382720813





def I1(x1,x2,qt,RA,aA,Z, force_computation=False):
    global grid_I1
    if "grid_I1" in globals() and not force_computation:
        return grid_I1.evaluate([x1,x2,qt])
    
    elif os.path.exists(f"{path}{ion}/grid_I1_{ion}.pkl") and not force_computation:
        with open(f"{path}{ion}/grid_I1_{ion}.pkl", "rb") as file:
            grid_I1 = pickle.load(file)
            return grid_I1.evaluate([x1,x2,qt])
    elif force_computation:
        return c_eval.I1_eval([x1,x2,qt],RA,aA,Z)
	
    else:
        tqdm.tqdm.write("INFO I1: No grid found, computing grid")
        tqdm.tqdm.write("INFO I1: This may take a while, maybe you want to use compute_grid_HTCONDOR.py")
        Npointsx1 = config.getint('grid', 'Npointsx1')
        Npointsx2 = config.getint('grid', 'Npointsx2')
        Npointsqt = config.getint('grid', 'Npointsqt')
        logx1min = config.getfloat('grid', 'logx1min')
        logx1max = config.getfloat('grid', 'logx1max')
        logx2min = config.getfloat('grid', 'logx2min')
        logx2max = config.getfloat('grid', 'logx2max')
        logqtmin = config.getfloat('grid', 'logqtmin')
        logqtmax = config.getfloat('grid', 'logqtmax')
        grid_I1 = grid_3D.grid3D(ion,[Npointsx1,Npointsx2,Npointsqt],"I1",c_eval.I1_eval)

        #Fonction for multiprocessing
        global I1_multiprocess
        
        def I1_multiprocess(X):
            i,j,k = X
            x1 = grid_I1.axis[0][i]
            x2 = grid_I1.axis[1][j]
            qt = grid_I1.axis[2][k]
            res = c_eval.I1_eval([x1,x2,qt],RA,aA,Z)
            return i,j,k,res
        
        X1 = np.logspace(logx1min,logx1max,Npointsx1)
        X2 = np.logspace(logx2min,logx2max,Npointsx2)
        QT = np.logspace(logqtmin,logqtmax,Npointsqt)
        index = [(i,j,k) for i in range(Npointsx1) for j in range(Npointsx2) for k in range(Npointsqt)]
        grid_I1.set_axis(0,X1)
        grid_I1.set_axis(1,X2)
        grid_I1.set_axis(2,QT)
        
        #Multiprocessing
        with Pool(num_core) as pool:
            res = list(tqdm.tqdm(pool.imap(I1_multiprocess, index), total=Npointsx1*Npointsx2*Npointsqt, desc="Computing", position=1, leave=True))
        #Construct the grid with the results
        for i,j,k,r in res:
            grid_I1.values[i,j,k] = r
        grid_I1.write(path)
        tqdm.tqdm.write("INFO I1: Grid computed")
        
        return grid_I1.evaluate([x1,x2,qt])
    
def I2(x1,x2,qt,RA,aA,Z, force_computation=False):
    global grid_I2
    if "grid_I2" in globals() and not force_computation:
        return grid_I2.evaluate([x1,x2,qt])
    
    elif os.path.exists(f"{path}{ion}/grid_I2_{ion}.pkl") and not force_computation:
            with open(f"{path}{ion}/grid_I2_{ion}.pkl", "rb") as file:
                grid_I2 = pickle.load(file)
                return grid_I2.evaluate([x1,x2,qt])
            
    elif force_computation:
        return c_eval.I2_eval([x1,x2,qt],RA,aA,Z)
        
    else:
        tqdm.tqdm.write("INFO I2: No grid found, computing grid")
        tqdm.tqdm.write("INFO I2: This may take a while, maybe you want to use compute_grid_HTCONDOR.py")
        Npointsx1 = config.getint('grid', 'Npointsx1')
        Npointsx2 = config.getint('grid', 'Npointsx2')
        Npointsqt = config.getint('grid', 'Npointsqt')
        logx1min = config.getfloat('grid', 'logx1min')
        logx1max = config.getfloat('grid', 'logx1max')
        logx2min = config.getfloat('grid', 'logx2min')
        logx2max = config.getfloat('grid', 'logx2max')
        logqtmin = config.getfloat('grid', 'logqtmin')
        logqtmax = config.getfloat('grid', 'logqtmax')
        grid_I2 = grid_3D.grid3D(ion,[Npointsx1,Npointsx2,Npointsqt],"I2",c_eval.I2_eval)

        #Fonction for multiprocessing
        global I2_multiprocess
        
        def I2_multiprocess(X):
            i,j,k = X
            x1 = grid_I2.axis[0][i]
            x2 = grid_I2.axis[1][j]
            qt = grid_I2.axis[2][k]
            res = c_eval.I2_eval([x1,x2,qt],RA,aA,Z)
            return i,j,k,res

        X1 = np.logspace(logx1min,logx1max,Npointsx1)
        X2 = np.logspace(logx2min,logx2max,Npointsx2)
        QT = np.logspace(logqtmin,logqtmax,Npointsqt)
        index = [(i,j,k) for i in range(Npointsx1) for j in range(Npointsx2) for k in range(Npointsqt)]
        grid_I2.set_axis(0,X1)
        grid_I2.set_axis(1,X2)
        grid_I2.set_axis(2,QT)
        
        #Multiprocessing
        with Pool(num_core) as pool:
            res = list(tqdm.tqdm(pool.imap(I2_multiprocess, index), total=Npointsx1*Npointsx2*Npointsqt, desc="Computing", position=1, leave=True))
        #Construct the grid with the results
        for i,j,k,r in res:
            grid_I2.values[i,j,k] = r
        grid_I2.write(path)
        tqdm.tqdm.write("INFO I2: Grid computed")
        
        return grid_I2.evaluate([x1,x2,qt])
    
def I3(x1,x2,qt,RA,aA,Z, force_computation=False):
    global grid_I3
    if "grid_I3" in globals() and not force_computation:
        return grid_I3.evaluate([x1,x2,qt])
    
    elif os.path.exists(f"{path}{ion}/grid_I3_{ion}.pkl") and not force_computation:
        with open(f"{path}{ion}/grid_I3_{ion}.pkl", "rb") as file:
            grid_I3 = pickle.load(file)
            return grid_I3.evaluate([x1,x2,qt])
    
    elif force_computation:
        return c_eval.I3_eval([x1,x2,qt],RA,aA,Z)
    
    else:
        tqdm.tqdm.write("INFO I3: No grid found, computing grid")
        tqdm.tqdm.write("INFO I3: This may take a while, maybe you want to use compute_grid_HTCONDOR.py")
        Npointsx1 = config.getint('grid', 'Npointsx1')
        Npointsx2 = config.getint('grid', 'Npointsx2')
        Npointsqt = config.getint('grid', 'Npointsqt')
        logx1min = config.getfloat('grid', 'logx1min')
        logx1max = config.getfloat('grid', 'logx1max')
        logx2min = config.getfloat('grid', 'logx2min')
        logx2max = config.getfloat('grid', 'logx2max')
        logqtmin = config.getfloat('grid', 'logqtmin')
        logqtmax = config.getfloat('grid', 'logqtmax')
        grid_I3 = grid_3D.grid3D(ion,[Npointsx1,Npointsx2,Npointsqt],"I3",c_eval.I3_eval)

        #Fonction for multiprocessing
        global I3_multiprocess
        
        def I3_multiprocess(X):
            i,j,k = X
            x1 = grid_I3.axis[0][i]
            x2 = grid_I3.axis[1][j]
            qt = grid_I3.axis[2][k]
            res = c_eval.I3_eval([x1,x2,qt],RA,aA,Z)
            return i,j,k,res
        
        X1 = np.logspace(logx1min,logx1max,Npointsx1)
        X2 = np.logspace(logx2min,logx2max,Npointsx2)
        QT = np.logspace(logqtmin,logqtmax,Npointsqt)
        index = [(i,j,k) for i in range(Npointsx1) for j in range(Npointsx2) for k in range(Npointsqt)]
        grid_I3.set_axis(0,X1)
        grid_I3.set_axis(1,X2)
        grid_I3.set_axis(2,QT)
        
        #Multiprocessing
        with Pool(num_core) as pool:
            res = list(tqdm.tqdm(pool.imap(I3_multiprocess, index), total=Npointsx1*Npointsx2*Npointsqt, desc="Computing", position=1, leave=True))
        #Construct the grid with the results
        for i,j,k,r in res:
            grid_I3.values[i,j,k] = r
        grid_I3.write(path)
        tqdm.tqdm.write("INFO I3: Grid computed")
        
        return grid_I3.evaluate([x1,x2,qt])
        
def I4(x1,x2,qt,RA,aA,Z, force_computation=False):
    global grid_I4
    if "grid_I4" in globals() and not force_computation:
        return grid_I4.evaluate([x1,x2,qt])
    
    elif os.path.exists(f"{path}{ion}/grid_I4_{ion}.pkl") and not force_computation:
        with open(f"{path}{ion}/grid_I4_{ion}.pkl", "rb") as file:
            grid_I4 = pickle.load(file)
            return grid_I4.evaluate([x1,x2,qt])
        
    elif force_computation:
        return c_eval.I4_eval([x1,x2,qt],RA,aA,Z)
    
    else:
        tqdm.tqdm.write("INFO I4: No grid found, computing grid")
        tqdm.tqdm.write("INFO I4: This may take a while, maybe you want to use compute_grid_HTCONDOR.py")
        Npointsx1 = config.getint('grid', 'Npointsx1')
        Npointsx2 = config.getint('grid', 'Npointsx2')
        Npointsqt = config.getint('grid', 'Npointsqt')
        logx1min = config.getfloat('grid', 'logx1min')
        logx1max = config.getfloat('grid', 'logx1max')
        logx2min = config.getfloat('grid', 'logx2min')
        logx2max = config.getfloat('grid', 'logx2max')
        logqtmin = config.getfloat('grid', 'logqtmin')
        logqtmax = config.getfloat('grid', 'logqtmax')
        grid_I4 = grid_3D.grid3D(ion,[Npointsx1,Npointsx2,Npointsqt],"I4",c_eval.I4_eval)

        #Fonction for multiprocessing
        global I4_multiprocess
        
        def I4_multiprocess(X):
            i,j,k = X
            x1 = grid_I4.axis[0][i]
            x2 = grid_I4.axis[1][j]
            qt = grid_I4.axis[2][k]
            res = c_eval.I4_eval([x1,x2,qt],RA,aA,Z)
            return i,j,k,res

        X1 = np.logspace(logx1min,logx1max,Npointsx1)
        X2 = np.logspace(logx2min,logx2max,Npointsx2)
        QT = np.logspace(logqtmin,logqtmax,Npointsqt)
        index = [(i,j,k) for i in range(Npointsx1) for j in range(Npointsx2) for k in range(Npointsqt)]
        grid_I4.set_axis(0,X1)
        grid_I4.set_axis(1,X2)
        grid_I4.set_axis(2,QT)
        
        #Multiprocessing
        with Pool(num_core) as pool:
            res = list(tqdm.tqdm(pool.imap(I4_multiprocess, index), total=Npointsx1*Npointsx2*Npointsqt, desc="Computing", position=1, leave=True))
        #Construct the grid with the results
        for i,j,k,r in res:
            grid_I4.values[i,j,k] = r
        grid_I4.write(path)
        tqdm.tqdm.write("INFO I4: Grid computed")
        
        return grid_I4.evaluate([x1,x2,qt])
    
    
def A_gammagamma(x1,x2,qt,Kt,ml,m_pair,RA,aA,Z, force_computation=False):
    return ((m_pair**2-2*ml**2)*ml**2 + (m_pair**2-2*Kt**2)*Kt**2)/(ml**2+Kt**2)**2*I1(x1,x2,qt,RA,aA,Z, force_computation) + ml**4/(ml**2+Kt**2)**2*I2(x1,x2,qt,RA,aA,Z, force_computation)

def B_gammagamma(x1,x2,qt,Kt,ml,m_pair,RA,aA,Z, force_computation=False):
    return 4*ml**2*Kt**2/(ml**2+Kt**2)**2*I3(x1,x2,qt,RA,aA,Z, force_computation)

def C_gammagamma(x1,x2,qt,Kt,ml,m_pair,RA,aA,Z, force_computation=False):
    return -2*Kt**4/(ml**2+Kt**2)**2*I4(x1,x2,qt,RA,aA,Z, force_computation)

def main():
    return 0

if __name__ == "__main__":
    main()
