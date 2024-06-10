import numpy as np
import pickle
import grid_3D

path = "Au197/grid_I1_Au197"

with open(f"{path}.pkl", "rb") as f:
    grid = pickle.load(f)
    
axex1 = grid.axis[0]
axex2 = grid.axis[1]
axeqt = grid.axis[2]
values = grid.values


with open(f"{path}.grid", "w") as f:
    f.write(f"{len(axex1)} {len(axex2)} {len(axeqt)}\n")
    for val in axex1:
        f.write(f"{val} ")
    f.write("\n")
    for val in axex2:
        f.write(f"{val} ")
    f.write("\n")
    for val in axeqt:
        f.write(f"{val} ")
    f.write("\n")
    for index,val in np.ndenumerate(values):
        f.write(f"{index[0]} {index[1]} {index[2]} {val}\n")