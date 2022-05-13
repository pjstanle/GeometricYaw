from tracemalloc import start
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface
import numpy as np
import pyoptsparse
from plotting_functions import plot_turbines
import time
from scipy.special import eval_legendre, eval_hermite


def process_layout(turbine_x,turbine_y,rotor_diameter,spread=0.1):

    nturbs = len(turbine_x)
    dx = np.zeros(nturbs) + 1E10
    dy = np.zeros(nturbs)
    for waking_index in range(nturbs):
        for waked_index in range(nturbs):
            if turbine_x[waked_index] > turbine_x[waking_index]:
                r = spread*(turbine_x[waked_index]-turbine_x[waking_index]) + rotor_diameter/2.0
                if abs(turbine_y[waked_index]-turbine_y[waking_index]) < (r+rotor_diameter/2.0):
                    if (turbine_x[waked_index] - turbine_x[waking_index]) < dx[waking_index]:
                        dx[waking_index] = turbine_x[waked_index] - turbine_x[waking_index]
                        dy[waking_index] = turbine_y[waked_index]- turbine_y[waking_index]
        if dx[waking_index] == 1E10:
            dx[waking_index] = 0.0

    return dx, dy


def get_yaw_angles(x,y,max_yaw,edge_yaw,min_x,max_x):
    if x <= 0:
        return 0.0
    else:
        edge_y = (1-(x-min_x)/(max_x-min_x))
        if abs(y) > edge_y:
            return 0.0
        else:
            yaw = (max_yaw-edge_yaw)*(1-abs(y/edge_y)) + edge_yaw
            if y < 0:
                return -yaw*(1-(x-min_x)/(max_x-min_x))
            else:
                return yaw*(1-(x-min_x)/(max_x-min_x))


def objective_function(x):

    global turbs_array
    global spacing_array
    global nruns
    global ws_array
    global spread
    global rotor_diameter
    global obj_scale
    global best_solution

    max_yaw = x["max_yaw"]
    edge_yaw = x["edge_yaw"]
    min_x = x["min_x"]
    max_x = x["max_x"]

    sum_power = 0.0

    for h in range(len(ws_array)):
        ws = ws_array[h]
        wind_speeds = [ws]
        fi.reinitialize(wind_directions=wind_directions,wind_speeds=wind_speeds)
        for i in range(len(turbs_array)):
            for j in range(len(spacing_array)):
                for k in range(nruns):
                    nturbs = turbs_array[i]
                    spacing = spacing_array[j]
                    if ws == 10:
                        file = np.load("yaw_data/ws%s/random_%s_%s_%s.npz"%(ws, nturbs, spacing, k))
                    else:
                        file = np.load("yaw_data/ws%s/ws%srandom_%s_%s_%s.npz"%(ws, ws, nturbs, spacing, k))

                    turbine_x = file['turbine_x']
                    turbine_y = file['turbine_y']
                    yaw_angles = np.zeros((1,1,len(turbine_x)))

                    fi.reinitialize(layout=(turbine_x,turbine_y))

                    dx, dy = process_layout(turbine_x,turbine_y, rotor_diameter, spread=spread)
                    for m in range(len(dx)):
                        yaw_angles[0,0,m] = get_yaw_angles(dx[m]/rotor_diameter, dy[m]/rotor_diameter,
                                                           max_yaw, edge_yaw, min_x, max_x)
                    
                    fi.calculate_wake(yaw_angles=yaw_angles)
                    sum_power += fi.get_farm_power()
    
    if -sum_power/obj_scale < best_solution:
        best_solution = -sum_power/obj_scale
        print("best_solution: ", best_solution)
        print("max_yaw: ", max_yaw)
        print("edge_yaw: ", edge_yaw)
        print("min_x: ", min_x)
        print("max_x: ", max_x)

    fail = False
    funcs = {}
    funcs["obj"] = -sum_power/obj_scale

    return funcs, fail


if __name__=="__main__":

    global turbs_array
    global spacing_array
    global nruns
    global ws_array
    global spread
    global rotor_diameter
    global obj_scale
    global best_solution

    fi = FlorisInterface("cc.yaml")
    wind_directions = [270.0]
    fi.reinitialize(wind_directions=wind_directions)
    
    turbs_array = [10,20,30]
    spacing_array = [10.0,9.0,8.0,7.0]
    nruns = 10
    ws_array = [6,8,10]
    spread = 0.15
    rotor_diameter = fi.floris.farm.rotor_diameters[0][0][0]

    start_max_yaw = 25
    start_edge_yaw = 5
    start_min_x = 5
    start_max_x = 30

    # x = {}
    # x["max_yaw"] = start_max_yaw
    # x["edge_yaw"] = start_edge_yaw
    # x["min_x"] = start_min_x
    # x["max_x"] = start_max_x
    # obj_scale = 1.0
    # print(-objective_function(x))
    obj_scale = 1.17481446e+9
    best_solution = 0.0

    optProb = pyoptsparse.Optimization("optimize rules", objective_function)
    optProb.addVar("max_yaw", type="c", value=start_max_yaw, upper=30.0, lower=0.0)
    optProb.addVar("edge_yaw", type="c", value=start_edge_yaw, upper=30.0, lower=0.0)
    optProb.addVar("min_x", type="c", value=start_min_x, upper=500.0, lower=-20.0)
    optProb.addVar("max_x", type="c", value=start_max_x, upper=500.0, lower=-20.0)
    optProb.addObj("obj")
    optimize = pyoptsparse.SNOPT()
    # optimize = pyoptsparse.SLSQP()
    solution = optimize(optProb,sens="FD")

    opt_DVs = solution.getDVs()
    print(opt_DVs)
    funcs, fail = objective_function(opt_DVs)
    print(funcs)
    