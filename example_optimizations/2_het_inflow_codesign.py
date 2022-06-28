# Copyright 2022 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

# See https://floris.readthedocs.io for documentation


from dataclasses import dataclass
import os
from turtle import speed
import numpy as np

from floris.tools import FlorisInterface
from codesign import CoDesignOptimizationPyOptSparse
from floris.tools.floris_interface import generate_heterogeneous_wind_map
from geometric_yaw import geometric_yaw
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import time
import yaml


import sys
sys.path.insert(0, './inputs')
from wind_roses import alturasRose



"""
This example shows a simple layout optimization using the python module pyOptSparse.

A 26 turbine array is optimized such that the layout of the turbine produces the
highest annual energy production (AEP) based on the given wind resource. The turbines
are constrained to a square boundary and a heterogenous wind resource is supplied.
"""

def rotate_wind_map(data_x, data_y, wind_function, new_dir):
    """
    rotates speed ups map for a new wind direction, assuming the baseline
    data is for wind from 270 degrees
    """
    # func = interpolate.interp2d(rotated_x, rotated_y, data_speed_ups, kind='linear')
    # func = interpolate.RectBivariateSpline(grid_x, grid_y, data_speed_ups)
    rotation_angle = np.deg2rad((180-new_dir))

    rotated_x = np.cos(rotation_angle)*data_x + np.sin(rotation_angle)*data_y
    rotated_y = -np.sin(rotation_angle)*data_x + np.cos(rotation_angle)*data_y
    
    new_speed_up = np.zeros_like(rotated_x)
    for i in range(len(rotated_x)):
        new_speed_up[i] = wind_function(rotated_x[i], rotated_y[i])

    return new_speed_up


def create_wind_map(grid_x, grid_y):
    sx = 600
    sy = 600
    x0h = 0
    y0h = 0
    x0s = 400
    y0s = 0
    I, J = np.shape(grid_x)
    speed_ups = np.zeros_like(grid_x)
    for i in range(I):
        for j in range(J):
            speed_ups[i,j] = 1.0 + 0.4 * np.exp(-((grid_x[i,j]-x0h)**2/(2*sx**2) + (grid_y[i,j]-y0h)**2/(2*sy**2))) \
                        - 0.2 * np.exp(-((grid_x[i,j]-x0s)**2/(2*sx**2) + (grid_y[i,j]-y0s)**2/(2*sy**2)))
    
    return speed_ups


if __name__ == "__main__":
    minx = -2000
    maxx = 2000
    miny = -2000
    maxy = 2000

    grid_x = np.linspace(minx,maxx,50)
    grid_y = np.linspace(minx,maxx,50)
    grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
    x_locs = np.ndarray.flatten(grid_xx)
    y_locs = np.ndarray.flatten(grid_yy)

    speed_ups = create_wind_map(grid_xx, grid_yy)
    wind_function_small = interpolate.RectBivariateSpline(grid_x, grid_y, speed_ups)

    ndirs = 72
    nspeeds = 1 # directionally averaged if nspeeds = 1
    wind_directions, freq, wind_speeds = alturasRose(ndirs, nSpeeds=nspeeds)
    freq = freq/np.sum(freq)

    speed_ups_array = np.zeros((ndirs, len(x_locs)))
    for i in range(ndirs):
        speed_ups_array[i,:] = rotate_wind_map(x_locs, y_locs, wind_function_small, wind_directions[i])

    # Generate the linear interpolation to be used for the heterogeneous inflow.
    het_map_2d = generate_heterogeneous_wind_map(speed_ups_array, x_locs, y_locs)

    # Initialize the FLORIS interface fi
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fi = FlorisInterface('inputs/gch.yaml', het_map=het_map_2d)

    N = 1000
    for j in range(N):
        k = j + 51
        minx_bound = -1000
        maxx_bound = 1000
        miny_bound = -1000
        maxy_bound = 1000

        nturbs = 16
        np.random.seed(k)
        layout_x = np.ndarray.flatten((np.random.rand(nturbs) - 0.5) * 2000)
        layout_y = np.ndarray.flatten((np.random.rand(nturbs) - 0.5) * 2000)
        fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds, layout=(layout_x, layout_y), time_series=True)
        fi.calculate_wake()
        
        # The boundaries for the turbines, specified as vertices
        boundaries = [(minx_bound, miny_bound), (minx_bound, maxy_bound), (maxx_bound, maxy_bound), 
                    (maxx_bound, miny_bound), (minx_bound, miny_bound)]

        # Setup the optimization problem
        start_time = time.time()
        layout_opt = CoDesignOptimizationPyOptSparse(fi, boundaries, wind_directions, wind_speeds,
                                                     solver='SLSQP', storeHistory='hist_codesign_%s.hist'%k,
                                                     freq=freq, opt_yaw=True)
        print(layout_opt.rotor_diameter)
        # Run the optimization
        sol = layout_opt.optimize()
        run_time = time.time() - start_time
        ncalls = layout_opt.ncalls

        # Print and plot the results
        locsx = sol.getDVs()["x"]
        locsy = sol.getDVs()["y"]
        fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds,
                             layout=(locsx, locsy), time_series=True)

        yaw = np.zeros((ndirs, 1, nturbs))
        rd = 126.0
        for i in range(ndirs):
                temp_yaw = geometric_yaw(locsx, locsy, wind_directions[i], rd)
                yaw[i,:] = temp_yaw[:]
        fi.calculate_wake(yaw_angles=yaw)
        aep = np.sum(fi.get_farm_power() * freq * 8760)

        results_dict = {}
        results_dict["aep"] = float(aep)
        results_dict["run_time"] = float(run_time)
        results_dict["function_calls"] = int(ncalls)
        results_dict["opt_turbine_x"] = locsx.tolist()
        results_dict["opt_turbine_y"] = locsy.tolist()
        results_dict["start_turbine_x"] = layout_x.tolist()
        results_dict["start_turbine_y"] = layout_y.tolist()
        results_dict["yaw"] = np.ndarray.flatten(yaw).tolist()
        results_filename = '2_results_codesign/results_codesign_%s.yml'%k
        with open(results_filename, 'w') as outfile:
            yaml.dump(results_dict, outfile)
