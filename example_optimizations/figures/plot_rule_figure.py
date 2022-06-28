import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
import sys
import os
sys.path.insert(0, '../../')
from geometric_yaw import process_layout, get_yaw_angles


def place_turbines(nturbs,side_x,side_y,min_spacing):
    iterate = True
    while iterate:
        turbine_x = np.array([])
        turbine_y = np.array([])
        for i in range(nturbs):
            placed = False
            counter = 0
            while placed == False:
                counter += 1
                temp_x = np.random.rand()*side_x
                temp_y = np.random.rand()*side_y
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


yaw_angles = np.array([])
x_array = np.array([])
y_array = np.array([])

rotor_diameter = 126.0

for path, currentDirectory, files in os.walk("/Users/astanley/Projects/active_projects/GeometricYaw/yaw_data_grid"):
    for file in files:
        filename = os.path.join(path, file)
        data = np.load(filename)
        yaw_angles = np.append(yaw_angles, data["yaw_angles"])
        x, y = process_layout(data["turbine_x"], data["turbine_y"], rotor_diameter)
        x_array = np.append(x_array, x)
        y_array = np.append(y_array, y)

for path, currentDirectory, files in os.walk("/Users/astanley/Projects/active_projects/GeometricYaw/yaw_data"):
    for file in files:
        filename = os.path.join(path, file)
        data = np.load(filename)
        yaw_angles = np.append(yaw_angles, data["yaw_angles"])
        x, y = process_layout(data["turbine_x"], data["turbine_y"], rotor_diameter)
        x_array = np.append(x_array, x)
        y_array = np.append(y_array, y)


# orange = "F15412"
# blue = "34B3F1"
white = "FFFFFF"
black = "000000"
purple = "293462"
pink = "F24C4C"
yellow = "F7D716"
blue = "000053"
orange = "EC9B3B"
lav = "ACACC6"
mycolormap = get_continuous_cmap([orange,black,lav])
fig = plt.figure(figsize=(6,5))

np.random.seed(1)
x, y = place_turbines(5,10,10,5)
x[2] = 3.1
y[2] = 2.4
x[3] = 7.5
y[3] = 0.0
print(x,y)
wake_x = np.array([0,10])
top_y = np.array([0.5, 0.5+10*0.1])
bot_y = np.array([-0.5, -(0.5+10*0.1)])

ax1 = plt.subplot(221)
wind_direction = 0.0
index = [0]
for i in range(len(x)):
    if i in index:
        circ = plt.Circle((x[i],y[i]),0.5,color=hex_to_rgb(orange, normalized=True))
        ax1.add_patch(circ)
    elif i in [4]:
        circ = plt.Circle((x[i],y[i]),0.5,color=hex_to_rgb(lav, normalized=True))
        ax1.add_patch(circ)
    else:
        circ = plt.Circle((x[i],y[i]),0.5,color=hex_to_rgb(black, normalized=True))
        ax1.add_patch(circ)

rot_wake_x_top = wake_x*np.cos(wind_direction) - top_y*np.sin(wind_direction)
rot_wake_x_bot = wake_x*np.cos(wind_direction) - bot_y*np.sin(wind_direction)
rot_wake_y_top = wake_x*np.sin(wind_direction) + top_y*np.cos(wind_direction)
rot_wake_y_bot = wake_x*np.sin(wind_direction) + bot_y*np.cos(wind_direction)

for i in index:
    ax1.plot(rot_wake_x_top+x[i],rot_wake_y_top+y[i],"--k",linewidth=0.5)
    ax1.plot(rot_wake_x_bot+x[i],rot_wake_y_bot+y[i],"--k",linewidth=0.5)
ax1.axis("square")
ax1.set_xlim(-2,12)
ax1.set_xticks((0,5,10))
ax1.set_yticks((0,5,10))
ax1.set_ylim(-2,12)


ax2 = plt.subplot(222)
wind_direction = -0.9
index = [0]
for i in range(len(x)):
    if i in index:
        circ = plt.Circle((x[i],y[i]),0.5,color=hex_to_rgb(orange, normalized=True))
        ax2.add_patch(circ)
    elif i in [2]:
        circ = plt.Circle((x[i],y[i]),0.5,color=hex_to_rgb(lav, normalized=True))
        ax2.add_patch(circ)
    else:
        circ = plt.Circle((x[i],y[i]),0.5,color=hex_to_rgb(black, normalized=True))
        ax2.add_patch(circ)

rot_wake_x_top = wake_x*np.cos(wind_direction) - top_y*np.sin(wind_direction)
rot_wake_x_bot = wake_x*np.cos(wind_direction) - bot_y*np.sin(wind_direction)
rot_wake_y_top = wake_x*np.sin(wind_direction) + top_y*np.cos(wind_direction)
rot_wake_y_bot = wake_x*np.sin(wind_direction) + bot_y*np.cos(wind_direction)

for i in index:
    ax2.plot(rot_wake_x_top+x[i],rot_wake_y_top+y[i],"--k",linewidth=0.5)
    ax2.plot(rot_wake_x_bot+x[i],rot_wake_y_bot+y[i],"--k",linewidth=0.5)
ax2.axis("square")
ax2.set_xlim(-2,12)
ax2.set_xticks((0,5,10))
ax2.set_yticks((0,5,10))
ax2.set_ylim(-2,12)





ax3 = plt.subplot(223)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
ax3.scatter(x_array,y_array,s=0.2,c=yaw_angles,cmap=mycolormap,vmin=-30.0,vmax=30.0)
ax3.set_xlim(4,35)
ax3.set_xlabel("dx/D",fontsize=8)
ax3.set_ylabel("dy/D",fontsize=8)
ax3.set_title("continuously optimized yaw angles",fontsize=8)
ax3.text(5, 3.5, "C", verticalalignment="top", fontweight="bold",fontsize=12)

ax4 = plt.subplot(224,sharex=ax3,sharey=ax3)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)

top_left_y=1.0
right_x=25.0
top_right_y=1.0
top_left_yaw=30.0
bottom_left_yaw=30.0
# top_right_yaw=0.4695
# bottom_right_yaw=0.7895
top_right_yaw=0.0
bottom_right_yaw=0.0

boundary_x = np.array([4.75,4.75,50,50,4.75,4.75,right_x,right_x,4.75])
boundary_y = np.array([1.0,1.5,6.35,-6.35,-1.5,-1.0,-1.0,1.0,1.0])
ax4.fill(boundary_x,boundary_y,color="black")


ax4.set_xlabel("dx/D",fontsize=8)
ax4.set_title("geometric yaw relationship",fontsize=8)
ax4.text(5, 3.5, "D", verticalalignment="top", fontweight="bold",fontsize=12)

x = np.linspace(4.75,25,500)
y = np.linspace(-1,1,500)
z = np.zeros((len(y)-1,len(x)-1))
for i in range(len(x)-1):
    for j in range(len(y)-1):
        X = (x[i]+x[i+1])/2.0
        Y = (y[j]+y[j+1])/2.0
        z[j,i] = get_yaw_angles(X, Y, 0.0, top_left_y, right_x, top_right_y, top_left_yaw,
                   top_right_yaw, bottom_left_yaw, bottom_right_yaw)


ax4.pcolormesh(x, y, z,cmap=mycolormap)

# fig.subplots_adjust(right=0.85,left=0.1,bottom=0.15)
# cbar_ax = fig.add_axes([0.88, 0.1495, 0.03, 0.73])
# cbar = fig.colorbar(im, cax=cbar_ax)
# cbar.set_label("yaw angle",fontsize=8)
# cbar.ax.tick_params(labelsize=8)

# plt.savefig("figures/yaw_rule.png",transparent=True)
plt.show()