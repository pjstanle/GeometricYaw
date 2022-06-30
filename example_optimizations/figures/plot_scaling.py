import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


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


def eval_polynom(f, x):
    res = 0
    for i in range(len(f)):
        res += f[i] * x**(len(f)-i-1)

    return res  


white = "FFFFFF"
black = "000000"
purple = "293462"
pink = "FFEEEC"
blue = "000053"
orange = "EC9B3B"
yellow = "F7D716"
lav = "ACACC6"

# plt.figure(figsize=(6,2.5))
# ax1 = plt.subplot(121)
# plt.xticks(fontsize=8)
# plt.yticks(fontsize=8)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.yaxis.set_ticks_position('left')
# ax1.xaxis.set_ticks_position('bottom')

# dimvec2 = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
# snopt_grad = np.array([31, 35, 36, 45, 64, 68, 157, 183, 195, 244])
# slsqp_grad = np.array([33, 30, 43, 76, 135, 271, 535, 271, 445, 577])
# snopt_fd = np.array([102, 248, 375, 1017, 3054, 6315, 60142, 151453, 212337, 776704])
# slsqp_fd = np.array([84, 118, 227, 621, 1646, 5324, 16877, 36639, 46939, 152124])
# dimvec = [2, 4, 8, 16, 32, 64]
# alpso = np.array([1150, 32780, 108040, 488240, 2649760, 12301760])


# ax1.set_xscale('log')
# ax1.set_yscale('log')

# ax1.plot(dimvec2,snopt_grad,'o',color=hex_to_rgb(blue,normalized=True))
# ax1.plot(dimvec2,snopt_fd,'o',color=hex_to_rgb(lav,normalized=True))
# ax1.plot(dimvec,alpso,'o',color=hex_to_rgb(orange,normalized=True))

# ax1.set_xticks((1E1,1E2,1E3))
# ax1.set_yticks((1E1,1E3,1E5,1E7))
# ax1.set_yticklabels((r'10$^1$',r'10$^3$',r'10$^5$',r'10$^7$'))

# ax1.text(45, 20, 'analytic gradients',weight="bold",fontsize=8,color=hex_to_rgb(blue,normalized=True))
# ax1.text(45, 1*5e2, 'finite difference\ngradients',weight="bold",fontsize=8,color=hex_to_rgb(lav,normalized=True))
# ax1.text(30, 7e5, 'gradient-free',weight="bold",fontsize=8,color=hex_to_rgb(orange,normalized=True))

# ax1.set_ylabel('number of function\ncalls to optimize',fontsize=8)
# ax1.set_xlabel('number of design variables',fontsize=8)

# plt.minorticks_off()

plt.figure(figsize=(3,2))
ax2 = plt.subplot(111)
plt.xticks(fontsize=8)
plt.yticks(fontsize=8)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.yaxis.set_ticks_position('left')
ax2.xaxis.set_ticks_position('bottom')

plt.minorticks_off()

nturbs = np.array([4,9,4,9,16])
ndirs = np.array([36,36,24,24,24])
time = np.array([945.8585067,87542.75563,527.7842207,37656.76095,279360.4023])/3600
ncalls = np.array([168,3190,126,2074,4960])
nvars = np.zeros(len(nturbs))
for i in range(len(nvars)):
    nvars[i] = nturbs[i]*2 + nturbs[i]*ndirs[i]

ax2.plot(nturbs[2:],time[2:],"o",color=hex_to_rgb(blue,normalized=True))
# ax2.plot(nturbs[:2],ncalls[:2],"o",color=hex_to_rgb(orange,normalized=True))

f = np.polyfit(nturbs[2:],time[2:],2)
print(f)
x = np.linspace(4,20,1000)
ax2.plot(x,eval_polynom(f,x),color=hex_to_rgb(blue,normalized=True))
plt.subplots_adjust(bottom=0.2,left=0.2,top=0.85)

ax2.set_xlabel("number of turbines", fontsize=8)
ax2.set_ylabel("time to optimize (hr)", fontsize=8)
ax2.set_title("fully coupled layout and yaw optimization\nwith 24 wind directions",fontsize=8)

# plt.text(nturbs[2]+1,time[2]+10,"%s variables"%int(nvars[2]),fontsize=8,horizontalalignment="center",verticalalignment="bottom")
# plt.text(nturbs[3],time[3]+10,"%s"%int(nvars[3]),fontsize=8,horizontalalignment="center",verticalalignment="bottom")
# plt.text(nturbs[4]-1,time[4]+10,"%s"%int(nvars[4]),fontsize=8,horizontalalignment="center",verticalalignment="bottom")
plt.savefig('scaling.png',transparent=True)
plt.show()
