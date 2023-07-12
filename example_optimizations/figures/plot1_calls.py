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

white = "FFFFFF"
black = "000000"
purple = "293462"
lavendar = "DCD0FF"
pink = "FFEEEC"
# blue = "6CA6CD"
blue = "000053"
orange = "EC9B3B"
yellow = "F7D716"





filename = "/glb/hou/pt.sgs/data/offshorewind/uspsty/Projects/GeometricYaw/example_optimizations/1_results/layout_continuous_25.yml"

nruns = 50
layout_calls = np.zeros(nruns)
layout_time = np.zeros(nruns)
codesign_calls = np.zeros(nruns)
codesign_time = np.zeros(nruns)
for i in range(nruns):
    filename = "/glb/hou/pt.sgs/data/offshorewind/uspsty/Projects/GeometricYaw/example_optimizations/1_results/layout_continuous_%s.yml"%i
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    layout_calls[i] = data_loaded["ncalls"]
    layout_time[i] = data_loaded["opt_time"]

    filename = "/glb/hou/pt.sgs/data/offshorewind/uspsty/Projects/GeometricYaw/example_optimizations/1_results/codesign_continuous_%s.yml"%i
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    codesign_calls[i] = data_loaded["ncalls"]
    codesign_time[i] = data_loaded["opt_time"]

print(np.sum(layout_calls))
print(np.sum(codesign_calls))
print(np.sum(layout_time))
print(np.sum(codesign_time))

plt.figure(figsize=(6,2))
bins = np.linspace(10,26,17)
ax1 = plt.subplot(121)
ax1.hist(layout_calls, color=hex_to_rgb(blue,normalized=True), alpha=0.5, bins=bins, label="layout only")
ax1.hist(codesign_calls, color=hex_to_rgb(orange,normalized=True), alpha=0.5, bins=bins, label="with geometric\nyaw")
ax1.legend(fontsize=8, loc=2)
# ax1.set_xticks((10,12,14,16,18,20,))
plt.tick_params(which="both", labelsize=8)

bins = np.linspace(2,5,17)
ax2 = plt.subplot(122)
ax2.hist(layout_time, color=hex_to_rgb(blue,normalized=True), alpha=0.5, bins=bins)
ax2.hist(codesign_time, color=hex_to_rgb(orange,normalized=True), alpha=0.5, bins=bins)
plt.tick_params(which="both", labelsize=8)

ax1.set_ylim(0,18)
ax2.set_ylim(0,18)

ax1.set_xlabel("function calls", fontsize=8)
ax2.set_xlabel("wall time", fontsize=8)
ax1.set_ylabel("count", fontsize=8)



plt.subplots_adjust(top=0.99, bottom=0.25, left=0.1, right=0.99)

plt.savefig("line_expense.png")
plt.show()
