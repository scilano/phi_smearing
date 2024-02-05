import coefficient as cc
import coefficient_eval as c_eval
import grid_3D
import numpy as np
import htmap
import tqdm
import configparser
from ast import literal_eval
config = configparser.ConfigParser()
config.read('config.ini')



def construct_grid(grid_name):
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


    index = [(i,j) for i in range(Npointsx1) for j in range(Npointsx2)]
    X1 = np.logspace(logx1min,logx1max,Npointsx1)
    X2 = np.logspace(logx2min,logx2max,Npointsx2)
    QT = np.logspace(logqtmin,logqtmax,Npointsqt)

    Fgrid = htmap.load(f"{grid_name}")



    result = Fgrid.iter(timeout = 1)

    print(f"Map {grid_name} loaded.")

    f = getattr(c_eval, f'{grid_name}_eval')
    grid = grid_3D.grid3D(ion,[Npointsx1,Npointsx2,Npointsqt],grid_name,f)
    grid.set_axis(0,X1)
    grid.set_axis(1,X2)
    grid.set_axis(2,QT)

    contruct_progress = tqdm.tqdm(total=len(Fgrid),desc="Constructing grid")

    values3D = np.zeros((Npointsx1,Npointsx2,Npointsqt))


    for X in result:
        i,j,res = X
        values3D[j,:,i] = np.array(res)
        contruct_progress.update(1)
    contruct_progress.close()
    
    if values3D[values3D == 0].size > 0:
        print("WARNING: Some values are still zero. Something may have gone wrong.")
        return 0
    
    if values3D[np.isnan(values3D)].size > 0:
        print("WARNING: Some values are NaN. Something may have gone wrong.")
        return 0
    grid.values = values3D

    grid.write(path)
    print(f"Grid {grid_name} saved.")
    
    
grid_names = ["I4"]

for grid in grid_names:
    construct_grid(grid)