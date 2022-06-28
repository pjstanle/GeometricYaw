import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.insert(0, './inputs')
from wind_roses import alturasRose

indices = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,51,52]

nfiles = len(indices)
aep_layout = np.zeros(nfiles)
aep_codesign = np.zeros(nfiles)
for i in range(nfiles):
    index = indices[i]
    filename = "2_results_codesign/results_codesign_%s.yml"%index
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    aep_codesign[i] = data_loaded["aep"]/1E13

    filename = "2_results/results_%s.yml"%index
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    aep_layout[i] = data_loaded["aep"]/1E13


filename = "2_results/sequential_yaw_31.yml"
with open(filename, 'r') as stream:
    data_loaded = yaml.safe_load(stream)
optimized_powers = data_loaded["optimized_powers"]
initial_powers = data_loaded["initial_powers"]

ndirs = 72
nspeeds = 1
wind_directions, freq, wind_speeds = alturasRose(ndirs, nSpeeds=nspeeds)
freq = freq/np.sum(freq)
opt_aep = np.sum(initial_powers * freq * 8760)
print("just layout: ", np.max(aep_layout))
print("opt aep sequential: ", opt_aep/1E13)
print("codesign: ", np.max(aep_codesign))

# print(indices[np.argmax(aep_layout)])
# print(indices[np.argmax(aep_codesign)])
bins = np.linspace(1.85,2.05,25)
plt.hist(aep_layout,bins=bins,alpha=0.5)
plt.hist(aep_codesign,bins=bins,alpha=0.5)
plt.show()