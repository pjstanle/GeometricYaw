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
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import time
import yaml
import pyoptsparse
from floris.tools.optimization.yaw_optimization.yaw_optimizer_sr import (
    YawOptimizationSR,
)

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


def obj_func(varDict):
    global ncalls
    global fi
    global freq

    ndirs = 72
    nturbs = 16
    yaw_angles = varDict["yaw"]
    yaw = np.zeros((ndirs,1,nturbs))
    for i in range(ndirs):
        yaw[i,:] = yaw_angles[nturbs*i:nturbs*(i+1)]
    ncalls += 1
    fi.calculate_wake(yaw_angles=yaw)

    # Compute the objective function
    funcs = {}
    funcs["obj"] = (
            - np.sum(fi.get_farm_power() * freq * 8760)
        )/1E11

    fail = False
    return funcs, fail


if __name__ == "__main__":
    global ncalls
    global fi
    global freq

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
    
    # layout only
    # index = 31
    # filename = "2_results/results_%s.yml"%index

    # coupled layout and yaw
    index = 40
    filename = "2_results_codesign/results_codesign_%s.yml"%index
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    turbine_x = data_loaded["opt_turbine_x"]
    turbine_y = data_loaded["opt_turbine_y"]
    fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds, layout=(turbine_x, turbine_y), time_series=True)

    nturbs = len(turbine_x)

    ncalls = 0

    initial_yaw = np.zeros(nturbs*ndirs)
    x = {}
    x["yaw"] = initial_yaw

    funcs,fail = obj_func(x)
    initial_aep = -funcs["obj"]

    # OPTIMIZATION
    optProb = pyoptsparse.Optimization("optimize yaw",obj_func)

    optProb.addVarGroup("yaw", nturbs*ndirs, type="c", value=initial_yaw, upper=30, lower=-30)
    optProb.addObj("obj")
    optimize = pyoptsparse.SLSQP()
    # optimize.setOption("MAXIT",value=50)
    # optimize.setOption("ACC",value=1E-4)

    ncalls = 0
    start_time = time.time()
    solution = optimize(optProb,sens="FD")
    opt_time = time.time() - start_time

    # END RESULTS
    opt_DVs = solution.getDVs()
    opt_yaw = opt_DVs["yaw"]

    funcs, fail = obj_func(opt_DVs)
    opt_aep = -funcs["obj"]

    print("opt_aep: ", opt_aep)
    print("ncalls: ", ncalls)
    print("time: ", opt_time)

    # start_power_array[i] = initial_power
    # print("percent improvement: ", (opt_power_array[i] - initial_power)/initial_power*100.0)

    results_dict = {}
    results_dict["opt_time"] = float(opt_time)
    results_dict["ncalls"] = int(ncalls)
    results_dict["opt_aep"] = float(opt_aep)
    results_dict["yaw_angles"] = opt_yaw.tolist()
    results_filename = '2_results_codesign/sequential_codesign_yaw_40.yml'
    with open(results_filename, 'w') as outfile:
        yaml.dump(results_dict, outfile)
        



    # # # filename = "2_results_codesign/results_codesign_40.yml"
    # # # with open(filename, 'r') as stream:
    # # #     data_loaded = yaml.safe_load(stream)
    # # # turbine_xc = data_loaded["opt_turbine_x"]
    # # # turbine_yc = data_loaded["opt_turbine_y"]

    # # # from plotting_functions import plot_turbines
    # # # plt.figure(1)
    # # # plot_turbines(turbine_x,turbine_y,126/2,ax=plt.gca())
    # # # plt.axis("square")
    
    # # # plt.figure(2)
    # # # plot_turbines(turbine_xc,turbine_yc,126/2,ax=plt.gca())
    # # # plt.axis("square")

    # # # plt.show()