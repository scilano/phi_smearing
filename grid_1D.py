"""
File containing the grid class in 3D.
"""
import numpy as np
import pickle
import scipy as scp
import tqdm
import configparser
from multiprocessing import Pool


class grid1D:
    """
    Represents a grid for interpolation in 1D.
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
    
    def __init__(self, ion: str, lengths: int or list, name: str = None, function = None):
        """
        Initializes a new instance of the grid class.

        Args:
            ion (str): The ion associated with the grid.
            dim (int): The number of dimensions of the grid.
            lengths (tupple): The length of each dimension.
            name (str, optional): The name of the grid.
            function (function, optional): The function associated with the grid, under the form f(x,R,A,Z)
        """
        config = configparser.ConfigParser()
        config.read('/afs/cern.ch/user/n/ncrepet/work/scripts/phi_smearing_0.3/config.ini')
        WoodsSaxon = eval(config.get("Ion", "WoodsSaxon"))
        self.ion = ion
        self.RA = WoodsSaxon[ion][0]/0.197
        self.aA = WoodsSaxon[ion][1]/0.197
        self.Z = WoodsSaxon[ion][3]
        self.lengths = lengths
       
        self.name = name
        self.function = function

        
        self.axis = np.zeros((self.lengths))
        self.values = np.zeros(self.lengths)
        self.interpolated = False

    def set_axis(self, values_axis: list):
        """
        Sets the values of the grid axis at the specified dimension.

        Args:
            i (int): The dimension index.
            values_axis (list): The values of the grid axis.
        
        Raises:
            ValueError: If the dimension index is out of range or the length of values does not match grid length.
        """
        if len(values_axis) != self.lengths:
            raise ValueError("Length of values does not match grid length")
        self.axis = values_axis

    def enlarge_axis(self, new_axis_values: list):
        """
        Extand the grid axis at the specified dimension.

        Args:
            values_axis (list): The values of the grid axis.
        
        Raises:
            ValueError: If the dimension index is out of range or the length of values is not greater than grid length.
        """
        self.interpolated = False
        
        #Create the new axis
        if new_axis_values[0] < self.axis[0]:
            self.axis = np.append(new_axis_values,self.axis)
        elif new_axis_values[-1] > self.axis[-1]:
            self.axis = np.append(self.axis,new_axis_values)
        else:
            raise ValueError("You are not supposed to be here")
        
         #Compute the new length of the axis
        self.lengths = len(self.axis)

        #Compute the new values of the grid
        new_grid = np.zeros(self.lengths)
        tqdm.tqdm.write(f"INFO {self.name}: Computing new grid values, may take a while")
        pbar = tqdm.tqdm(total=np.prod(self.lengths))
        for index in range(self.lengths):
            try :
                #If the index is in the old grid, copy the value
                new_grid[index] = self.values[index]
                pbar.update(1)
            except:
                x = self.axis[index]
                new_grid[index] = float(self.function(x,self.RA,self.aA,self.Z))
                pbar.update(1)
        self.values = new_grid
        if self.saved:
            self.write(self.path)
        tqdm.tqdm.write(f"INFO {self.name}: New grid values computed")        

    
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

        self.interpolatefun = scp.interpolate.CubicSpline(self.axis, self.values, extrapolate=False)

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
        
        if x > self.axis[-1] or x < self.axis[0]:
            if self.function != None:
                #If the value is out of range, enlarge the grid, add 5 values per new decade
                tqdm.tqdm.write(f"ERROR {self.name}: Value out of range, augment grid size")
                pointperdecade = int(self.lengths/np.log10(self.axis[-1]/self.axis[0]))
                
                if x > self.axis[-1]:
                    new_max = 10**np.ceil(np.log10(x))
                    numbernewdecade = int(np.log10(new_max/self.axis[-1]))
                    additonal_points = np.logspace(np.log10(self.axis[-1]),np.log10(new_max),numbernewdecade*pointperdecade+1)[1:]
                    tqdm.tqdm.write(f"INFO {self.name}: Adding "+str(len(additonal_points))+" points, max updated to "+str(new_max))
                elif x < self.axis[0]:
                    new_min = 10**np.floor(np.log10(x))
                    numbernewdecade = int(np.log10(self.axis[0]/new_min))
                    additonal_points = np.logspace(np.log10(new_min),np.log10(self.axis[0]),numbernewdecade*pointperdecade+1)[:-1]
                    tqdm.tqdm.write(f"INFO {self.name}: Adding "+str(len(additonal_points))+" points, min updated to "+str(new_min))
                else:
                    raise ValueError("You are not supposed to be here")
                
                self.enlarge_axis(additonal_points)
                return self.evaluate(x)
            else:
                #If the value is out of range and no function is associated to the gird, raise an error
                raise ValueError(f"ERROR {self.name}: Value out of range")
                
        return float(self.interpolatefun(x))