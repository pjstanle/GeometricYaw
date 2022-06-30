import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors as mcolors
from matplotlib.gridspec import GridSpec
import sys
import os
sys.path.insert(0, '../../')
from geometric_yaw import process_layout, get_yaw_angles
from floris.utilities import rotate_coordinates_rel_west


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

wind_direction = 0.0
rot_wake_x_top = wake_x*np.cos(wind_direction) - top_y*np.sin(wind_direction)
rot_wake_x_bot = wake_x*np.cos(wind_direction) - bot_y*np.sin(wind_direction)
rot_wake_y_top = wake_x*np.sin(wind_direction) + top_y*np.cos(wind_direction)
rot_wake_y_bot = wake_x*np.sin(wind_direction) + bot_y*np.cos(wind_direction)

index = [0]
ax1 = plt.subplot(221)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
arrow = plt.arrow(0,8.5,2,0,head_width=0.5,head_length=0.5,color="black")
ax1.text(1,9,"wind\ndirection",fontsize=8,verticalalignment="bottom",horizontalalignment="center")
for i in index:
    ax1.plot(rot_wake_x_top+x[i],rot_wake_y_top+y[i],"--k",linewidth=0.5)
    ax1.plot(rot_wake_x_bot+x[i],rot_wake_y_bot+y[i],"--k",linewidth=0.5)
ax1.fill_between(rot_wake_x_top+x[i],rot_wake_y_top+y[i],rot_wake_y_bot+y[i],color=hex_to_rgb(orange, normalized=True), alpha=0.2)


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



ax1.plot([x[0],x[4]],[y[0],y[0]],color=hex_to_rgb(black, normalized=True))
ax1.plot([x[4],x[4]],[y[0],y[4]],color=hex_to_rgb(black, normalized=True))
ax1.text((x[4]-x[0])/2+x[0], y[0], "dx", fontsize=8, horizontalalignment="center", verticalalignment="bottom")
ax1.text(x[4], (y[0]-y[4])/2+y[4], "dy", fontsize=8, horizontalalignment="left", verticalalignment="center")

ax2 = plt.subplot(222)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
wind_direction = -0.9
arrow = plt.arrow(-0.75,10,2*np.cos(wind_direction),2*np.sin(wind_direction),head_width=0.5,head_length=0.5,color=black)
ax2.text(2.0,8.75,"wind\ndirection",fontsize=8,verticalalignment="bottom",horizontalalignment="center")
index = [0]

rot_wake_x_top = wake_x*np.cos(wind_direction) - top_y*np.sin(wind_direction)
rot_wake_x_bot = wake_x*np.cos(wind_direction) - bot_y*np.sin(wind_direction)
rot_wake_y_top = wake_x*np.sin(wind_direction) + top_y*np.cos(wind_direction)
rot_wake_y_bot = wake_x*np.sin(wind_direction) + bot_y*np.cos(wind_direction)

for i in index:
    ax2.plot(rot_wake_x_top+x[i],rot_wake_y_top+y[i],"--k",linewidth=0.5)
    ax2.plot(rot_wake_x_bot+x[i],rot_wake_y_bot+y[i],"--k",linewidth=0.5)
ax2.fill(np.array([rot_wake_x_top[0],rot_wake_x_top[-1],rot_wake_x_bot[-1],rot_wake_x_bot[0],rot_wake_x_top[0]])+x[0],
         np.array([rot_wake_y_top[0],rot_wake_y_top[-1],rot_wake_y_bot[-1],rot_wake_y_bot[0],rot_wake_y_top[0]])+y[0],color=hex_to_rgb(orange, normalized=True), alpha=0.2)


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

nturbs = 5
turbine_coordinates_array = np.zeros((nturbs,3))
turbine_coordinates_array[:,0] = x[:]
turbine_coordinates_array[:,1] = y[:]
rtx, rty, _ = rotate_coordinates_rel_west(np.array([-90-np.rad2deg(wind_direction)]), turbine_coordinates_array)
dx = rtx[0,0,2] - rtx[0,0,0]
dy = rty[0,0,0] - rty[0,0,2]

dx_line_x = [x[0],dx*np.cos(wind_direction)+x[0]]
dx_line_y = [y[0],dx*np.sin(wind_direction)+y[0]]
plt.plot(dx_line_x,dx_line_y,color="black")
dy_line_x = [dx*np.cos(wind_direction)+x[0],dx*np.cos(wind_direction)+dy*np.sin(wind_direction)+x[0]]
dy_line_y = [dx*np.sin(wind_direction)+y[0],dx*np.sin(wind_direction)-dy*np.cos(wind_direction)+y[0]]
plt.plot(dy_line_x,dy_line_y,color="black")

ax2.text(dx*np.cos(wind_direction)/2+x[0],(dx_line_y[1]-dx_line_y[0])/2+y[0],"dx",fontsize=8,horizontalalignment="left",verticalalignment="bottom")
ax2.text((dy_line_x[0]-dy_line_x[1])/2+x[2],(dy_line_y[1]-dy_line_y[0])/2+y[2],"dy",fontsize=8,horizontalalignment="left",verticalalignment="top")




ax1.axis("equal")
ax1.set_xticks((0,5,10))
ax1.set_yticks((0,5,10))
ax1.set_xbound(lower=-2,upper=12)
ax1.set_ybound(lower=-1,upper=11)
ax1.text(-0.5,-0.25,"A",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="bottom")


ax2.axis("equal")
ax2.set_xticks((0,5,10))
ax2.set_yticks((0,5,10))
ax2.set_xbound(lower=-2,upper=12)
ax2.set_ybound(lower=-1,upper=11)
ax2.text(-1.0,-0.6,"B",fontsize=12,weight="bold",horizontalalignment="left",verticalalignment="bottom")

ax1.set_xlabel("x (D)",fontsize=8)
ax1.set_ylabel("y (D)",fontsize=8)

ax2.set_xlabel("x (D)",fontsize=8)
ax2.set_ylabel("y (D)",fontsize=8)



ax3 = plt.subplot(223)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
ax3.scatter(x_array,y_array,s=0.2,c=yaw_angles,cmap=mycolormap,vmin=-30.0,vmax=30.0)
ax3.set_xlim(4,35)
ax3.set_xlabel("dx/D",fontsize=8)
ax3.set_ylabel("dy/D",fontsize=8)
ax3.set_title("continuously optimized yaw angles",fontsize=8)


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


ax4.set_ylabel("dy/D",fontsize=8)
ax4.set_xlabel("dx/D",fontsize=8)
ax4.set_title("geometric yaw relationship",fontsize=8)


x = np.linspace(4.75,25,500)
y = np.linspace(-1,1,500)
z = np.zeros((len(y)-1,len(x)-1))
for i in range(len(x)-1):
    for j in range(len(y)-1):
        X = (x[i]+x[i+1])/2.0
        Y = (y[j]+y[j+1])/2.0
        z[j,i] = get_yaw_angles(X, Y, 0.0, top_left_y, right_x, top_right_y, top_left_yaw,
                   top_right_yaw, bottom_left_yaw, bottom_right_yaw)


im = ax4.pcolormesh(x, y, z,cmap=mycolormap,vmax=30,vmin=-30)

ax3.text(6, -6, "C", verticalalignment="bottom", fontweight="bold",fontsize=12)
ax4.text(6, -6, "D", verticalalignment="bottom", fontweight="bold",fontsize=12)


plt.suptitle("dx and dy definition", fontsize=8, y=0.99)

plt.subplots_adjust(left=0.1,right=0.85,bottom=0.08,top=0.95,wspace=0.3,hspace=0.4)
cbar_ax = fig.add_axes([0.88,0.08,0.025,0.362])
cbar = fig.colorbar(im,cax=cbar_ax)
cbar.ax.tick_params(labelsize=8)
cbar.set_label("wind speed", fontsize=8)

plt.savefig("yaw_rule.png",transparent=True)

plt.show()

