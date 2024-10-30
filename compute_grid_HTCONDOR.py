import coefficient_eval as c_eval
import os
import numpy as np
import htmap
import htcondor
from ast import literal_eval
import configparser
from time import perf_counter



def compute_grid(grid_name,index,jobFlavour='"testmatch"'):
    config = configparser.ConfigParser()
    config.read('config.ini')
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


    credd = htcondor.Credd()
    print("[CREDD] Adding user credentials to credd daemon")
    credd.add_user_cred(htcondor.CredTypes.Kerberos, None)
    htmap.settings["DELIVERY_METHOD"] = "shared"
    htmap.settings["MAP_OPTIONS.getenv"] = "True"
    

    input_files_py = [file for file in os.listdir(path) if (".py" in file) or (".ini" in file)]
    X1 = np.logspace(logx1min,logx1max,Npointsx1)
    X2 = np.logspace(logx2min,logx2max,Npointsx2)
    QT = np.logspace(logqtmin,logqtmax,Npointsqt)
    
    global grid_func

    def grid_func(X, RA, aA, Z, grid_name):
        i, j = X
        print(f"Computing {grid_name} at ({i},{j})...")
        f = getattr(c_eval, f'{grid_name}_eval')
        x1 = X1[i]
        x2 = X2[j]
        res = np.zeros_like(QT)
        N=len(QT)
        t1 = perf_counter()
        for k, qt in enumerate(QT):
            print(k/N*100)
            res[k] = f([x1, x2, qt], RA, aA, Z)
        t2 = perf_counter()
        print(t2-t1)
        return i, j, res


    if grid_name in htmap.get_tags():
        print(f"Previous {grid_name} found, deleting...")
        htmap.remove(grid_name,True)
        print(f"Previous {grid_name} deleted.")

    print(f"Building {grid_name}...")
    opt = {}
    opt["JobFlavour"] = jobFlavour
    opt["MY.SendCredential"] = "true"
    with htmap.build_map(grid_func, tag=grid_name, map_options=htmap.MapOptions(fixed_input_files=input_files_py, custom_options=opt)) as grid_builder:
        for X in index:
            grid_builder(X, R, a, Z, grid_name)
    print(f"{grid_name} built.")

    grid = grid_builder.map

    print(f"Computation launched on HTCondor for {grid_name}.")


def main():
    grid_names =["I2","I3","I4"]

    for grid in grid_names:
        config = configparser.ConfigParser()
        config.read('config.ini')
        Npointsx1 = config.getint(f'grid', 'Npointsx1')
        Npointsx2 = config.getint(f'grid', 'Npointsx2')
        index = [(i,j) for i in range(Npointsx1) for j in range(Npointsx2)]
        
        compute_grid(grid,index)
    return 0


if __name__ == "__main__":
    main()
        
