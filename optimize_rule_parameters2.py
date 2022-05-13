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


def get_yaw_angles(x, y, left_x, top_left_y, right_x, top_right_y, top_left_yaw,
                   top_right_yaw, bottom_left_yaw, bottom_right_yaw):

    if x <= 0:
        return 0.0
    else:
        dx = (x-left_x)/(right_x-left_x)
        edge_y = top_left_y + (top_right_y-top_left_y)*dx
        if abs(y) > edge_y:
            return 0.0
        else:
            top_yaw = top_left_yaw + (top_right_yaw-top_left_yaw)*dx
            bottom_yaw = bottom_left_yaw + (bottom_right_yaw-bottom_left_yaw)*dx
            yaw = bottom_yaw + (top_yaw-bottom_yaw)*abs(y)/edge_y
            if y < 0:
                return -yaw
            else:
                return yaw


def objective_function(x):

    global turbs_array
    global spacing_array
    global nruns
    global ws_array
    global spread
    global rotor_diameter
    global obj_scale
    global best_solution

    left_x = x["left_x"]
    top_left_y = x["top_left_y"]
    right_x = x["right_x"]
    top_right_y = x["top_right_y"]
    top_left_yaw = x["top_left_yaw"]
    top_right_yaw = x["top_right_yaw"]
    bottom_left_yaw = x["bottom_left_yaw"]
    bottom_right_yaw = x["bottom_right_yaw"]

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
                                                           left_x, top_left_y, right_x, top_right_y, top_left_yaw,
                                                           top_right_yaw, bottom_left_yaw, bottom_right_yaw)
                    
                    fi.calculate_wake(yaw_angles=yaw_angles)
                    sum_power += fi.get_farm_power()
    
    if -sum_power/obj_scale < best_solution:
        best_solution = -sum_power/obj_scale
        print("best_solution: ", best_solution)
        print("left_x: ", left_x)
        print("top_left_y: ", top_left_y)
        print("right_x: ", right_x)
        print("top_right_y: ", top_right_y)
        print("top_left_yaw: ", top_left_yaw)
        print("top_right_yaw: ", top_right_yaw)
        print("bottom_left_yaw: ", bottom_left_yaw)
        print("bottom_right_yaw: ", bottom_right_yaw)

    fail = False
    funcs = {}
    funcs["obj"] = -sum_power[0][0]/obj_scale

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
    
    # turbs_array = [10,20,30]
    turbs_array = [30]
    spacing_array = [10.0,9.0,8.0,7.0]
    nruns = 10
    # ws_array = [6,8,10]
    ws_array = [10]
    spread = 0.15
    rotor_diameter = fi.floris.farm.rotor_diameters[0]

    start_left_x = 5.0
    start_top_left_y = 1.0
    start_right_x = 15.0
    start_top_right_y = 1.0
    start_top_left_yaw = 30.0
    start_top_right_yaw = 0.0
    start_bottom_left_yaw = 30.0
    start_bottom_right_yaw = 0.0

    x = {}
    x["left_x"] = start_left_x
    x["top_left_y"] = start_top_left_y
    x["right_x"] = start_right_x
    x["top_right_y"] = start_top_right_y
    x["top_left_yaw"] = start_top_left_yaw
    x["top_right_yaw"] = start_top_right_yaw
    x["bottom_left_yaw"] = start_bottom_left_yaw
    x["bottom_right_yaw"] = start_bottom_right_yaw


    # obj_scale = 1.0
    best_solution = 0.0
    # print(-objective_function(x))
    # obj_scale = 1.16524042e+10
    obj_scale = 3.33059712e+09

    optProb = pyoptsparse.Optimization("optimize rules", objective_function)
    optProb.addVar("left_x", type="c", value=start_left_x, upper=None, lower=5.0)
    optProb.addVar("top_left_y", type="c", value=start_top_left_y, upper=None, lower=0.0)
    optProb.addVar("right_x", type="c", value=start_right_x, upper=None, lower=0.0)
    optProb.addVar("top_right_y", type="c", value=start_top_right_y, upper=None, lower=0.0)
    optProb.addVar("top_left_yaw", type="c", value=start_top_left_yaw, upper=30.0, lower=0.0)
    optProb.addVar("top_right_yaw", type="c", value=start_top_right_yaw, upper=30.0, lower=0.0)
    optProb.addVar("bottom_left_yaw", type="c", value=start_bottom_left_yaw, upper=30.0, lower=0.0)
    optProb.addVar("bottom_right_yaw", type="c", value=start_bottom_right_yaw, upper=30.0, lower=0.0)
    optProb.addObj("obj")
    optimize = pyoptsparse.SNOPT()
    # optimize = pyoptsparse.SLSQP()
    solution = optimize(optProb,sens="FD")

    opt_DVs = solution.getDVs()
    print(opt_DVs)
    funcs, fail = objective_function(opt_DVs)
    print(funcs)
    