from floris.tools.visualization import visualize_cut_plane, plot_turbines
from floris.tools import FlorisInterface
import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

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


def plot_layout_opt_results_with_flow(fi, wd, ws, yaw_angles, ax):
        """
        Method to plot the old and new locations of the layout opitimization.
        """
        global colormap
        fi.calculate_wake(yaw_angles=yaw_angles)
        horizontal_plane_2d = fi.calculate_horizontal_plane(x_resolution=500,
                                y_resolution=100, height=90.0, wd=wd, ws=ws,
                                x_bounds=(-200.0, 14000.0), y_bounds=(-500.0, 500.0),
                                yaw_angles=yaw_angles)

        # plt.figure(figsize=(9, 6))
        im = visualize_cut_plane(horizontal_plane_2d, color_bar=False, ax=ax, maxSpeed=10, minSpeed=4, cmap=colormap)

        fontsize = 8
        # plt.plot(locsx, locsy, "or")
        # plt.title('Layout Optimization Results', fontsize=fontsize)
        plt.xlabel("x (m)", fontsize=fontsize)
        plt.ylabel("y (m)", fontsize=fontsize)
        plt.axis("equal")
        # plt.grid()
        plt.tick_params(which="both", labelsize=fontsize)

        return im

global colormap

white = "FFFFFF"
black = "000000"
purple = "293462"
lavendar = "DCD0FF"
pink = "FFEEEC"
# blue = "6CA6CD"
blue = "000053"
orange = "EC9B3B"
yellow = "F7D716"
colormap = get_continuous_cmap([blue,white])

fi = FlorisInterface('/glb/hou/pt.sgs/data/offshorewind/uspsty/Projects/GeometricYaw/example_optimizations/inputs/gch.yaml')
nturbs = 16
wind_directions = [270.0]
wind_speeds = [10.0]
turbine_x = np.zeros(nturbs)
turbine_y = np.zeros(nturbs)



filename = "/glb/hou/pt.sgs/data/offshorewind/uspsty/Projects/GeometricYaw/example_optimizations/1_results/layout_continuous_25.yml"
# filename = "/Users/astanley/Projects/active_projects/GeometricYaw/example_optimizations/1_results/codesign.yml"
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
spacing_array = data_loaded["opt_spacing"]
for i in range(nturbs-1):
    # turbine_x[i+1] = turbine_x[i] + spacing_array[i]
    turbine_x[i+1] = turbine_x[i] + spacing_array
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds,
                    layout=(turbine_x, turbine_y))

fig = plt.figure(figsize=(6,3))
ax1 = plt.subplot(411)
im = plot_layout_opt_results_with_flow(fi, wind_directions, wind_speeds, np.zeros((1,1,16)), ax1)
plot_turbines(ax1, turbine_x, turbine_y, np.zeros(16), np.zeros(nturbs)+126, color=None,
              wind_direction=270.0)

filename = "/glb/hou/pt.sgs/data/offshorewind/uspsty/Projects/GeometricYaw/example_optimizations/1_results/layout_continuous_yaw.yml"
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
opt_yaw = data_loaded["opt_yaw"]
ax2 = plt.subplot(412)
yaw_angles = np.zeros((1,1,16))
yaw_angles[0,0,:] = opt_yaw
print(yaw_angles)
im = plot_layout_opt_results_with_flow(fi, wind_directions, wind_speeds, yaw_angles, ax2)
plot_turbines(ax2, turbine_x, turbine_y, opt_yaw, np.zeros(nturbs)+126, color=None,
              wind_direction=270.0)

# filename = "/Users/astanley/Projects/active_projects/GeometricYaw/example_optimizations/1_results/layout.yml"
filename = "/glb/hou/pt.sgs/data/offshorewind/uspsty/Projects/GeometricYaw/example_optimizations/1_results/codesign_continuous_25.yml"
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
spacing_array = data_loaded["opt_spacing"]
for i in range(nturbs-1):
    # turbine_x[i+1] = turbine_x[i] + spacing_array[i]
    turbine_x[i+1] = turbine_x[i] + spacing_array
fi.reinitialize(wind_directions=wind_directions, wind_speeds=wind_speeds,
                    layout=(turbine_x, turbine_y))

ax3 = plt.subplot(413)
im = plot_layout_opt_results_with_flow(fi, wind_directions, wind_speeds, np.zeros((1,1,16)), ax3)
plot_turbines(ax3, turbine_x, turbine_y, np.zeros(16), np.zeros(nturbs)+126, color=None,
              wind_direction=270.0)


filename = "/glb/hou/pt.sgs/data/offshorewind/uspsty/Projects/GeometricYaw/example_optimizations/1_results/codesign_continuous_yaw.yml"
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
opt_yaw = data_loaded["opt_yaw"]
ax4 = plt.subplot(414)
yaw_angles = np.zeros((1,1,16))
yaw_angles[0,0,:] = opt_yaw
im = plot_layout_opt_results_with_flow(fi, wind_directions, wind_speeds, yaw_angles, ax4)
plot_turbines(ax4, turbine_x, turbine_y, opt_yaw, np.zeros(nturbs)+126, color=None,
              wind_direction=270.0)


ax1.axis("off")
ax2.axis("off")
ax3.axis("off")
ax4.axis("off")

print(repr(turbine_x))

# ax1.set_title("no yaw control: 1785 W/m",fontsize=8,y=0.7)
# ax2.set_title("optimized yaw control: 1991 W/m",fontsize=8,y=0.7)
# ax3.set_title("no yaw control: 1740 W/m",fontsize=8,y=0.7)
# ax4.set_title("optimized yaw control: 2135 W/m",fontsize=8,y=0.7)
ax1.set_title("no yaw control: 1784 W/m",fontsize=8,y=0.7)
ax2.set_title("optimized yaw control: 1996 W/m",fontsize=8,y=0.7)
ax3.set_title("no yaw control: 1718 W/m",fontsize=8,y=0.7)
ax4.set_title("optimized yaw control: 2125 W/m",fontsize=8,y=0.7)

plt.subplots_adjust(left=0.06,right=0.91,bottom=0.01,top=0.97,hspace=0.0)

ax1.text(-600,-2000,"layout optimized\nwithout yaw",fontsize=8,horizontalalignment="center",rotation=90)
ax3.text(-600,-2000,"layout optimized\nwith geometric yaw",fontsize=8,horizontalalignment="center",rotation=90)

cbar_ax = fig.add_axes([0.92,0.05,0.02,0.9])
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("wind speed", fontsize=8, labelpad=-0.1)

ax1.text(0,400,"A.1",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="bottom")
ax2.text(0,400,"A.2",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="bottom")
ax3.text(0,400,"B.1",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="bottom")
ax4.text(0,400,"B.2",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="bottom")
plt.savefig("line_layouts_revision.png", transparent=True)
plt.show()

