from tracemalloc import start
import matplotlib.pyplot as plt
from floris.tools import FlorisInterface
import numpy as np
import pyoptsparse
from plotting_functions import plot_turbines
import time


def create_initial_grid(x_spacing,y_spacing,nrows,ncols):

    xlocs = np.arange(ncols)*x_spacing
    ylocs = np.arange(nrows)*y_spacing

    # make initial grid
    layout_x = np.array([x for x in xlocs for y in ylocs])
    layout_y = np.array([y for x in xlocs for y in ylocs])

    return layout_x, layout_y


def shear_grid_locs(shear,x,y):
    shear_y = np.copy(y)
    shear_x = np.zeros_like(shear_y)
    dy = y[1]-y[0]
    nturbs = len(x)
    for i in range(nturbs):
        row_num = (y[i]-np.min(y))/dy
        shear_x[i] = x[i] + (row_num-1)*dy*np.tan(shear)
    return shear_x, shear_y


def rotate_grid_locs(rotation,x,y):
    # rotate
    rotate_x = np.cos(rotation)*x - np.sin(rotation)*y
    rotate_y = np.sin(rotation)*x + np.cos(rotation)*y

    return rotate_x, rotate_y


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

    fi = FlorisInterface("cc.yaml")
    wind_directions = [270.0]
    wind_speeds = [10.0]
    fi.reinitialize(wind_directions=wind_directions,wind_speeds=wind_speeds)

    D = fi.floris.farm.rotor_diameters[0]
    

    nrows_arr = [3,4,5,6]
    ncols_arr = [3,4,5,6]
    x_spacing_arr = np.array([10.0,9.0,8.0,7.0,6.0,5.0])*D
    y_spacing_arr = np.array([10.0,9.0,8.0,7.0,6.0,5.0])*D
    # shear = np.linspace(-np.pi/4,np.pi/4,9)
    shear_arr = [0.0]
    rotation_arr = np.linspace(-np.pi/2,np.pi/2,91)
    ws_array = [10.0,12.0]
    nruns = 1
    for h in range(len(ws_array)):
        wind_speeds = [ws_array[h]]
        fi.reinitialize(wind_speeds=wind_speeds)
        for i in range(len(nrows_arr)):
            for j in range(len(x_spacing_arr)):
                for m in range(len(rotation_arr)):
                    for k in range(nruns):

                        nrows = nrows_arr[i]
                        ncols = nrows_arr[i]
                        x_spacing = x_spacing_arr[j]
                        y_spacing = x_spacing_arr[j]
                        shear = shear_arr[0]
                        rotation = rotation_arr[m]

                        base_x,base_y = create_initial_grid(x_spacing,y_spacing,nrows,ncols)
                        shear_x,shear_y = shear_grid_locs(shear,base_x,base_y)
                        turbine_x,turbine_y = rotate_grid_locs(rotation,shear_x,shear_y)

                        fi.reinitialize(layout=(turbine_x,turbine_y))                   
                        fi.calculate_wake()
                        scale_power = fi.get_farm_power()*10.0

                        optProb = pyoptsparse.Optimization("optimize yaw", objective_function_baseline)
                        optProb.addVarGroup("yaw", nrows*ncols, type="c", value=0.0, upper=30.0, lower=-30.0)
                        optProb.addObj("obj")
                        optimize = pyoptsparse.SNOPT()
                        start_time = time.time()
                        solution = optimize(optProb,sens="FD")

                        opt_DVs = solution.getDVs()
                        yaws = opt_DVs["yaw"]
                        funcs, fail = objective_function_baseline(opt_DVs)

                        np.savez("yaw_data_grid/ws%s"%(int(ws_array[h])) + "grid_%s_%s_%s"%(nrows, int(x_spacing/D), round(np.rad2deg(rotation),2)),
                                  turbine_x=turbine_x, turbine_y=turbine_y, yaw_angles=yaws)


        
        