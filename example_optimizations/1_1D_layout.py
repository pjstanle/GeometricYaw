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


import os
import numpy as np

from floris.tools import FlorisInterface
import pandas as pd
import matplotlib.pyplot as plt
import time
import yaml
import pyoptsparse

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometric_yaw import geometric_yaw


def obj_func(varDict):
    global ncalls
    global fi
    global wind_directions
    global wind_speeds

    opt_yaw = True

    ncalls += 1
    nturbs = 16
    spacing_array = varDict["spacing_array"]
    turbine_x = np.zeros(nturbs)
    turbine_y = np.zeros(nturbs)
    for i in range(nturbs-1):
        turbine_x[i+1] = turbine_x[i] + spacing_array[i]

    fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds,
                    layout=(turbine_x, turbine_y))

    if opt_yaw:
        yaw = np.zeros((1,1,nturbs))
        rd = 126.0
        yaw[0,:] = geometric_yaw(turbine_x, turbine_y, wind_directions[0], rd)
        fi.calculate_wake(yaw_angles=yaw)
    else:
        fi.calculate_wake()

    # Compute the objective function
    funcs = {}
    funcs["obj"] = (
            - np.sum(fi.get_farm_power())
        ) / np.sum(spacing_array)

    fail = False
    return funcs, fail


if __name__ == "__main__":
    global ncalls
    global fi
    global wind_directions
    global wind_speeds

    ncalls = 0

    wind_directions = [270.0]
    wind_speeds = [10.0]
    freq = [1.0]

    # Initialize the FLORIS interface fi
    file_dir = os.path.dirname(os.path.abspath(__file__))
    fi = FlorisInterface('inputs/gch.yaml')

    nturbs = 16
    rotor_diameter = 126.0
    initial_spacing_array = np.zeros(nturbs-1) + 5*rotor_diameter
    
    x = {}
    x["spacing_array"] = initial_spacing_array

    funcs,fail = obj_func(x)
    initial_density = -funcs["obj"]
    print(initial_density)

    # OPTIMIZATION
    optProb = pyoptsparse.Optimization("optimize 1D layout",obj_func)

    optProb.addVarGroup("spacing_array", nturbs-1, type="c", value=initial_spacing_array, lower=0.0, upper=1E6)
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
    opt_spacing = opt_DVs["spacing_array"]

    funcs, fail = obj_func(opt_DVs)
    opt_density = -funcs["obj"]

    print("opt_density: ", opt_density)
    print("ncalls: ", ncalls)
    print("time: ", opt_time)
    print("opt_spacing: ", repr(opt_spacing))

    results_dict = {}
    results_dict["opt_time"] = float(opt_time)
    results_dict["ncalls"] = int(ncalls)
    results_dict["opt_density"] = float(opt_density)
    results_dict["opt_spacing"] = opt_spacing.tolist()
    results_filename = '1_results/codesign.yml'
    with open(results_filename, 'w') as outfile:
        yaml.dump(results_dict, outfile)


