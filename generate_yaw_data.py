from tracemalloc import start
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface
import numpy as np
import pyoptsparse
from plotting_functions import plot_turbines
import time
from scipy.special import eval_legendre, eval_hermite


def place_turbines(nturbs,side,min_spacing):
    iterate = True
    while iterate:
        turbine_x = np.array([])
        turbine_y = np.array([])
        for i in range(nturbs):
            placed = False
            counter = 0
            while placed == False:
                counter += 1
                temp_x = np.random.rand()*side
                temp_y = np.random.rand()*side
                good_point = True
                for j in range(len(turbine_x)):
                    dist = np.sqrt((temp_y - turbine_y[j])**2 + (temp_x - turbine_x[j])**2)
                    if dist < min_spacing:
                        good_point = False
                if good_point == True:
                    turbine_x = np.append(turbine_x,temp_x)
                    turbine_y = np.append(turbine_y,temp_y)
                    placed = True
                # while loop
                if counter == 1000:
                    break

            # for loop
            if counter == 1000:
                    break

        if counter != 1000:
            return turbine_x, turbine_y


def objective_function_baseline(x):

    global fi
    global scale_power

    yaw = x["yaw"]
    nturbs = len(fi.floris.farm.layout_x)
    yaw_angles = np.zeros((1,1,nturbs))
    yaw_angles[0,0,:] = yaw
    fi.calculate_wake(yaw_angles=yaw_angles)
    farm_power = fi.get_farm_power()

    funcs = {}
    fail = False

    funcs["obj"] = -farm_power/scale_power
    return funcs, fail


if __name__=="__main__":

    global fi
    global scale_power
    global dx
    global dy
    global order_x

    fi = FlorisInterface("cc.yaml")
    wind_directions = [270.0]
    wind_speeds = [10.0]
    fi.reinitialize(wind_directions=wind_directions,wind_speeds=wind_speeds)

    D = fi.floris.farm.rotor_diameters[0][0][0]
    
    turbs_array = [20,30]
    spacing_array = [9.0,8.0,7.0]
    nruns = 10
    ws_array = [12.0,14.0]
    for h in range(len(ws_array)):
        wind_speeds = [ws_array[h]]
        fi.reinitialize(wind_speeds=wind_speeds)
        for i in range(len(turbs_array)):
            for j in range(len(spacing_array)):
                for k in range(nruns):
                    nturbs = turbs_array[i]
                    spacing = spacing_array[j]
                    s = spacing*(np.sqrt(nturbs)-1)*D
                    layout_x, layout_y = place_turbines(nturbs,s,5*D)

                    fi.reinitialize(layout=(layout_x,layout_y))
                    fi.calculate_wake()
                    scale_power = fi.get_farm_power()*10.0

                    optProb = pyoptsparse.Optimization("optimize yaw", objective_function_baseline)
                    optProb.addVarGroup("yaw", nturbs, type="c", value=0.0, upper=30.0, lower=-30.0)
                    optProb.addObj("obj")
                    optimize = pyoptsparse.SNOPT()
                    start_time = time.time()
                    solution = optimize(optProb,sens="FD")

                    opt_DVs = solution.getDVs()
                    yaws = opt_DVs["yaw"]
                    funcs, fail = objective_function_baseline(opt_DVs)

                    np.savez("yaw_data/ws%s"%(int(ws_array[h])) + "random_%s_%s_%s"%(nturbs, spacing, k), turbine_x=layout_x,
                                    turbine_y=layout_y, yaw_angles=yaws)


        
        