"""
File containing the grid class in 3D.
"""
import numpy as np
import pickle
import scipy as scp
import tqdm
import configparser
from coefficient_eval import *
from ast import literal_eval
import os


class grid3D:
    """
    Represents a grid for interpolation in 3D.
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
        path = os.path.dirname(os.path.abspath(__file__))+'/'
        config = configparser.ConfigParser()
        config.read(path+'config.ini')
        WoodsSaxon = literal_eval(config.get("Ion", "WoodsSaxon"))
        self.ion = ion
        self.RA = WoodsSaxon[ion][0]/0.197
        self.aA = WoodsSaxon[ion][1]/0.197
        self.Z = WoodsSaxon[ion][3]
        self.lengths = lengths
       
        self.name = name
        self.function = function
        if len(self.lengths) != 3:
            raise ValueError(f"lenghts does not match dimension of grid : 3 {lengths} = {self.lengths.shape}")
        
        self.axis = [np.zeros((3, self.lengths[i])) for i in range(3)]
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
        if i > 3:
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
        if i > 3:
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
        tqdm.tqdm.write(f"INFO {self.name}: Computing new grid values, may take a while")
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
        with open(f"{path}{self.ion}/grid_{self.name}_{self.ion}.pkl", "wb") as file:
            pickle.dump(self, file)

    def interpolate(self):
        """
        Interpolates the grid values.
        User does not need to call this function, it is called automatically when the grid is evaluated.

        Raises:
            ValueError: If the grid dimension is not supported.
        """
        self.interpolatefun = scp.interpolate.RegularGridInterpolator(self.axis, self.values, bounds_error=False, fill_value=None)
        self.interpolated = True

    def evaluate(self, x):
        """
        Evaluates the interpolated grid at the specified value.

        Args:
            x: The value at which to evaluate the grid.

        Returns:
            The interpolated value at the specified value.
        """
        inBound1 = x[0] >= self.axis[0][0] and x[0] <= self.axis[0][-1]
        inBound2 = x[1] >= self.axis[1][0] and x[1] <= self.axis[1][-1]
        inBound3 = x[2] >= self.axis[2][0] and x[2] <= self.axis[2][-1]
        inBound = inBound1 and inBound2 and inBound3
        if self.interpolated == False:
            self.interpolate()
        if inBound:
            val = float(self.interpolatefun(x))
            return float(self.interpolatefun(x))
        else:
            raise ValueError(f"Value {x} out of grid bounds:({self.axis[0][0]}-{self.axis[0][-1]}) ({self.axis[1][0]}-{self.axis[1][-1]}) ({self.axis[2][0]}-{self.axis[2][-1]})")
            
        