import coefficient as cc
import coefficient_eval as c_eval
import grid_3D
import numpy as np
import htmap
import tqdm
import configparser
import compute_grid_HTCONDOR
from ast import literal_eval
import pickle
config = configparser.ConfigParser()
config.read('config.ini')



def construct_grid(grid_name,missing = False):
    path = config.get('General', 'path')
    ion = config.get('General', 'ion_name')
    WoodsSaxon = literal_eval(config.get("Ion", "WoodsSaxon"))
    R = WoodsSaxon[ion][0]/0.197
    a = WoodsSaxon[ion][1]/0.197
    Z = WoodsSaxon[ion][3]
    Npointsx1 = config.getint(f'grid', 'Npointsx1')
    Npointsx2 = config.getint(f'grid', 'Npointsx2')
    Npointsqt = config.getint(f'grid', 'Npointsqt')
    logx1min = config.getfloat(f'grid', 'logx1min')
    logx1max = config.getfloat(f'grid', 'logx1max')
    logx2min = config.getfloat(f'grid', 'logx2min')
    logx2max = config.getfloat(f'grid', 'logx2max')
    logqtmin = config.getfloat(f'grid', 'logqtmin')
    logqtmax = config.getfloat(f'grid', 'logqtmax')
    X1 = np.logspace(logx1min,logx1max,Npointsx1)
    X2 = np.logspace(logx2min,logx2max,Npointsx2)
    QT = np.logspace(logqtmin,logqtmax,Npointsqt)
    Fgrid = htmap.load(f"{grid_name}")
    
    result = Fgrid.components_by_status()[htmap.ComponentStatus.COMPLETED]
    print(f"Map {grid_name} loaded.")
    
    
    

    if missing:
        with open(f"{ion}/grid_{grid_name}_temp_{ion}.pkl", "rb") as f:
            tempGrid = pickle.load(f)
            tempGrid.name = grid_name
            for X in result:
                i,j,res = Fgrid.get(X)
                tempGrid.values[i,j,:] = np.array(res)
            
            if tempGrid.values[np.isnan(tempGrid.values)].size > 0:
                print("WARNING: Some values are NaN. Something may have gone wrong.")
                return 1
            
            if tempGrid.values[tempGrid.values == 0].size > 0:
                print("WARNING: Some values are zero. Something may have gone wrong.")
                return 1
            
            tempGrid.write(path)
            
            return 0
            
            
    else:      
        f = getattr(c_eval, f'{grid_name}_eval')
        grid = grid_3D.grid3D(ion,[Npointsx1,Npointsx2,Npointsqt],grid_name,f)
        grid.set_axis(0,X1)
        grid.set_axis(1,X2)
        grid.set_axis(2,QT)

        contruct_progress = tqdm.tqdm(total=len(result),desc="Constructing grid")

        values3D = np.zeros((Npointsx1,Npointsx2,Npointsqt))
        for X in result:
            i,j,res = Fgrid.get(X)
            values3D[i,j,:] = np.array(res)
            contruct_progress.update(1)
        contruct_progress.close()
        grid.values = values3D

        if len(result) != Npointsx1*Npointsx2:
            print("WARNING: Some components are missing. Rerunning missing components on a longer queue.")
            print(len(result))
            grid.name = grid_name+"_temp"
            removed = Fgrid.components_by_status()[htmap.ComponentStatus.REMOVED]
            old_index = [(i,j) for i in range(Npointsx1) for j in range(Npointsx2)]
            new_index = [old_index[i] for i in removed]
            compute_grid_HTCONDOR.compute_grid(grid_name,new_index,jobFlavour='"nextweek"')

        
        if values3D[np.isnan(values3D)].size > 0:
            print("WARNING: Some values are NaN. Something may have gone wrong.")
            return 1
   
        grid.write(path)
        print(f"Grid {grid_name} saved.")
    
    
grid_names = ["I2","I3"]

for grid in grid_names:
    construct_grid(grid)
