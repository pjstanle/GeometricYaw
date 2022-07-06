from turtle import color
from floris.tools.visualization import visualize_cut_plane, plot_turbines
from floris.tools import FlorisInterface
from floris.tools.floris_interface import generate_heterogeneous_wind_map
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import sys
sys.path.insert(0, '../inputs')
from wind_roses import alturasRose

import matplotlib.cm as cm
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
import os


def hex_to_rgb(value, normalized=False):
    '''
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values'''
    value = value.strip("#") # removes hash symbol if present
    lv = len(value)
    if normalized:
        RGB = tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
        return tuple(i/255 for i in RGB)
    else:
        return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    '''
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values'''
    return [v/256 for v in value]
    

def get_continuous_cmap(hex_list, float_list=None):
    ''' creates and returns a color map that can be used in heat map figures.
        If float_list is not provided, colour map graduates linearly between each color in hex_list.
        If float_list is provided, each color in hex_list is mapped to the respective location in float_list. 
        
        Parameters
        ----------
        hex_list: list of hex code strings
        float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.
        
        Returns
        ----------
        colour map'''
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = mcolors.LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp


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


def plot_layout_opt_results_with_flow(fi, wd, ws, yaw_angles, ax):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        global colormap
        fi.calculate_wake(yaw_angles=yaw_angles)
        b = 1300
        horizontal_plane_2d = fi.calculate_horizontal_plane(x_resolution=500,
                                y_resolution=100, height=90.0, wd=wd, ws=ws,
                                x_bounds=(-b, b), y_bounds=(-b, b),
                                yaw_angles=yaw_angles)

        im = visualize_cut_plane(horizontal_plane_2d, color_bar=False, ax=ax, cmap=colormap,minSpeed=1.0,maxSpeed=13.0)

        return im


global colormap

white = "FFFFFF"
black = "000000"
purple = "293462"
lav = "ACACC6"
pink = "FFEEEC"
# blue = "6CA6CD"
blue = "000053"
dark_blue = "000120"
orange = "EC9B3B"
yellow = "F7D716"

fontsize=8
fig = plt.figure(figsize=(6,5.5))

# cbar_ax2 = fig.add_axes([0.89,0.55,0.02,0.4])

ax1 = fig.add_axes([0.05,0.52,0.3,0.4],polar=True)
plt.tick_params(which="both", labelsize=fontsize)
ax3 = fig.add_axes([0.1,0.05,0.31,0.4])
plt.tick_params(which="both", labelsize=fontsize)
ax4 = fig.add_axes([0.57,0.05,0.31,0.4])
plt.tick_params(which="both", labelsize=fontsize)
ax2 = fig.add_axes([0.57,0.52,0.31,0.4])
plt.tick_params(which="both", labelsize=fontsize)

ax2.axis("square")
ax2.set_xlim(-1300,1300)
ax2.set_ylim(-1300,1300)

ax3.axis("square")
ax3.set_xlim(-1300,1300)
ax3.set_ylim(-1300,1300)

ax4.axis("square")
ax4.set_xlim(-1300,1300)
ax4.set_ylim(-1300,1300)


"""
ax1 windrose
"""
colormap = get_continuous_cmap([blue,lav])
ndirs = 72
nspeeds = 1 # directionally averaged if nspeeds = 1
wd, freq, ws = alturasRose(ndirs, nSpeeds=nspeeds)
freq = freq/np.sum(freq)

dirs = np.zeros(ndirs)
for i in range(ndirs):
    dirs[i] = np.deg2rad(90.0 - wd[i])
width = 1.0 * 2 * np.pi / ndirs
for i in range(ndirs):
    bottom = 0.0
    frac = (ws[i]-min(ws))/(max(ws)-min(ws))
    ax1.bar(dirs[i], freq[i],
            bottom=bottom, width=width,
            color=colormap(frac)
            )

pi = np.pi
ax1.set_xticks((0, pi / 4, pi / 2, 3 * pi / 4, pi, 5 * pi / 4,
                3 * pi / 2, 7 * pi / 4))
ax1.set_xticklabels(("E", "NE", "N", "NW", "W", "SW", "S", "SE"),
                    fontsize=8)
ax1.xaxis.labelpad = -2000.0
plt.yticks(fontsize=8)
ax1.set_yticks((0.01,0.02,0.03))
ax1.set_yticklabels(("1%","2%","3%"))

cbar_ax1 = fig.add_axes([0.4,0.5505,0.02,0.3388])
norm = mcolors.Normalize(vmin=min(ws), vmax=max(ws))
cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=colormap),
             cax=cbar_ax1)
cb.set_label(label='wind speed (m/s)',fontsize=8)
cb.ax.tick_params(labelsize=8)


"""
ax2 het inflow map
"""
# colormap = get_continuous_cmap([dark_blue,blue,orange],float_list=[0.0,0.09853168047904871,1.0])
colormap = get_continuous_cmap([blue,white])
minx = -2000
maxx = 2000
miny = -2000
maxy = 2000

grid_x = np.linspace(minx,maxx,500)
grid_y = np.linspace(minx,maxx,500)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
x_locs = np.ndarray.flatten(grid_xx)
y_locs = np.ndarray.flatten(grid_yy)

speed_ups = create_wind_map(grid_xx, grid_yy)
wind_function_small = interpolate.RectBivariateSpline(grid_x, grid_y, speed_ups)
im = ax2.pcolor(grid_xx, grid_yy, speed_ups, cmap=colormap)

minx_bound = -1000
maxx_bound = 1000
miny_bound = -1000
maxy_bound = 1000
bx = np.array([minx_bound,minx_bound,maxx_bound,maxx_bound,minx_bound])
by = np.array([miny_bound,maxy_bound,maxy_bound,miny_bound,miny_bound])
ax2.plot(bx,by,color=hex_to_rgb(orange,normalized=True))

cbar_ax2 = fig.add_axes([0.89,0.5505,0.02,0.3388])
cbar = fig.colorbar(im,cax=cbar_ax2)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("local speed multiplier", fontsize=8)

"""
ax3 layout only opt
"""
colormap = get_continuous_cmap([blue,white])

grid_x = np.linspace(minx,maxx,50)
grid_y = np.linspace(minx,maxx,50)
grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
x_locs = np.ndarray.flatten(grid_xx)
y_locs = np.ndarray.flatten(grid_yy)

speed_ups = create_wind_map(grid_xx, grid_yy)
wind_function_small = interpolate.RectBivariateSpline(grid_x, grid_y, speed_ups)

index = np.argmax(freq)
index = np.where(wd==270.0)[0]
# print(index)
wind_directions = [wd[index]]
wind_speeds = [ws[index]]

speed_ups_array = np.zeros((1, len(x_locs)))
speed_ups_array[0,:] = rotate_wind_map(x_locs, y_locs, wind_function_small, wind_directions[0])

# Generate the linear interpolation to be used for the heterogeneous inflow.
het_map_2d = generate_heterogeneous_wind_map(speed_ups_array, x_locs, y_locs)

# Initialize the FLORIS interface fi
file_dir = os.path.dirname(os.path.abspath(__file__))
fi = FlorisInterface('/Users/astanley/Projects/active_projects/GeometricYaw/example_optimizations/inputs/gch.yaml', het_map=het_map_2d)

rotation_angle = -np.deg2rad((270-wind_directions[0]))
ct = np.cos((rotation_angle))
st = np.sin((rotation_angle))
rx = bx*ct - by*st
ry = bx*st + by*ct

filename = "/Users/astanley/Projects/active_projects/GeometricYaw/example_optimizations/2_results/results_31.yml"
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
turbine_x = data_loaded["opt_turbine_x"]
turbine_y = data_loaded["opt_turbine_y"]
nturbs = len(turbine_x)
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds,
                    layout=(turbine_x, turbine_y))

rot_tx = np.array(turbine_x)*ct - np.array(turbine_y)*st
rot_ty = np.array(turbine_x)*st + np.array(turbine_y)*ct

filename = "/Users/astanley/Projects/active_projects/GeometricYaw/example_optimizations/2_results/sequential_yaw_31.yml"
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
opt_yaw = np.array(data_loaded["yaw_angles"])
opt_yaw = opt_yaw.reshape(72,16)
yaw_angles = np.zeros((1,1,16))
yaw_angles[0,0,:] = opt_yaw[index,:]
print(yaw_angles)
im = plot_layout_opt_results_with_flow(fi, wind_directions, wind_speeds, yaw_angles, ax3)
ax3.plot(rx,ry,color=hex_to_rgb(orange,normalized=True))
plot_turbines(ax3, rot_tx, rot_ty, opt_yaw[index], np.zeros(nturbs)+126, color=None,
              wind_direction=wind_directions[0],linewidth=0.75)


"""
ax4 geometric yaw opt
"""

filename = "/Users/astanley/Projects/active_projects/GeometricYaw/example_optimizations/2_results_codesign/results_codesign_40.yml"# with open(filename, 'r') as stream:
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
turbine_x = data_loaded["opt_turbine_x"]
turbine_y = data_loaded["opt_turbine_y"]
nturbs = len(turbine_x)
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds,
                    layout=(turbine_x, turbine_y))

rot_tx = np.array(turbine_x)*ct - np.array(turbine_y)*st
rot_ty = np.array(turbine_x)*st + np.array(turbine_y)*ct

filename = "/Users/astanley/Projects/active_projects/GeometricYaw/example_optimizations/2_results_codesign/sequential_codesign_yaw_40.yml"
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
opt_yaw = np.array(data_loaded["yaw_angles"])
opt_yaw = opt_yaw.reshape(72,16)

yaw_angles = np.zeros((1,1,16))
yaw_angles[0,0,:] = opt_yaw[index,:]
print(yaw_angles)
im = plot_layout_opt_results_with_flow(fi, wind_directions, wind_speeds, yaw_angles, ax4)
ax4.plot(rx,ry,color=hex_to_rgb(orange,normalized=True))
plot_turbines(ax4, rot_tx, rot_ty, opt_yaw[index,:], np.zeros(nturbs)+126, color=None,
              wind_direction=wind_directions[0],linewidth=0.75)

cbar_ax4 = fig.add_axes([0.89,0.08,0.02,0.3388])
cbar = fig.colorbar(im,cax=cbar_ax4)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("wind speed (m/s)", fontsize=8)

fontsize = 8
ax2.set_xlabel("x (m)", fontsize=fontsize)
ax2.set_ylabel("y (m)", fontsize=fontsize)
ax3.set_xlabel("x (m)", fontsize=fontsize)
ax3.set_ylabel("y (m)", fontsize=fontsize)
ax4.set_xlabel("x (m)", fontsize=fontsize)
ax4.set_ylabel("y (m)", fontsize=fontsize)

ax2.yaxis.labelpad = -15
ax3.yaxis.labelpad = -15
ax4.yaxis.labelpad = -15
ax2.xaxis.labelpad = -0.5
ax3.xaxis.labelpad = -0.5
ax4.xaxis.labelpad = -0.5

ax1.set_title("wind rose", fontsize=8, y=1.15)
ax2.set_title("gaussian hill inflow", fontsize=8, y=1.04)
ax3.set_title("layout optimized\nwithout yaw", fontsize=8, y=1.01)
ax4.set_title("layout optimized\nwith geometric yaw", fontsize=8, y=1.01)


ax2.text(-5500,1700,"A",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="top")
ax2.text(-1600,1700,"B",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="top")
ax3.text(-1600,1700,"C",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="top")
ax4.text(-1600,1700,"D",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="top")

# plt.savefig("het_layouts_update.png",transparent=True)
plt.show()