"""
A script to compare the performance of wake steering where the yaw angles are determined by different methods:
- 0 (no yaw control)
- Baseline (yaw angles optimized directly with gradient-based optimizer)
- V1 (yaw angles determined by geometric yaw relationship)
- V2 (yaw angles determined by machine learning)

"""


import matplotlib.pyplot as plt
import numpy as np
import time
import joblib
import pathlib
import os
from floris.tools import FlorisInterface
from floris.utilities import rotate_coordinates_rel_west


def calculate_power_0(fi):

    fi.calculate_wake()
    farm_power = fi.get_farm_power()

    return farm_power[0, 0]


def calculate_power_baseline(fi, baseline_yaw):

    yaw_angles = np.zeros((1,1,len(turbine_x)))
    yaw_angles[0, 0, :] = baseline_yaw[:]
        
    fi.calculate_wake(yaw_angles=yaw_angles)
    farm_power = fi.get_farm_power()

    return farm_power[0, 0]


def process_layout_V1(turbine_x,turbine_y,rotor_diameter,spread=0.1):
    """
    returns the distance from each turbine to the nearest downstream waked turbine
    normalized by the rotor diameter. Right now "waked" is determind by a Jensen-like
    wake spread, but this could/should be modified to be the same as the trapezoid rule
    used to determine the yaw angles.

    turbine_x: turbine x coords (rotated)
    turbine_y: turbine y coords (rotated)
    rotor_diameter: turbine rotor diameter (float)
    spread=0.1: Jensen alpha wake spread value
    """
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

    return dx/rotor_diameter, dy/rotor_diameter


def get_yaw_angles_V1(x, y, left_x, top_left_y, right_x, top_right_y, top_left_yaw,
                   top_right_yaw, bottom_left_yaw, bottom_right_yaw):
    """
    _______2,5___________________________4,6
    |.......................................
    |......1,7...........................3,8
    |.......................................
    ________________________________________

    I realize this is kind of a mess and needs to be clarified/cleaned up. As it is now:
    
    x and y: dx and dy to the nearest downstream turbine in rotor diameteters with turbines rotated so wind is coming left to right
    left_x: where we start the trapezoid...now that I think about it this should just be assumed as 0
    top_left_y: trapezoid top left coord
    right_x: where to stop the trapezoid. Basically, to max coord after which the upstream turbine won't yaw
    top_right_y: trapezoid top right coord
    top_left_yaw: yaw angle associated with top left point
    top_right_yaw: yaw angle associated with top right point
    bottom_left_yaw: yaw angle associated with bottom left point
    bottom_right_yaw: yaw angle associated with bottom right point

    """

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


def geometric_yaw_V1(turbine_x, turbine_y, wind_direction, rotor_diameter,
                  left_x=0.0,top_left_y=1.0,
                  right_x=25.0,top_right_y=1.0,
                  top_left_yaw=30.0,top_right_yaw=0.0,
                  bottom_left_yaw=30.0,bottom_right_yaw=0.0):
    """
    turbine_x: unrotated x turbine coords
    turbine_y: unrotated y turbine coords
    wind_direction: float, degrees
    rotor_diameter: float
    """

    nturbs = len(turbine_x)
    turbine_coordinates_array = np.zeros((nturbs,3))
    turbine_coordinates_array[:,0] = turbine_x[:]
    turbine_coordinates_array[:,1] = turbine_y[:]
    
    rotated_x, rotated_y, _ = rotate_coordinates_rel_west(np.array([wind_direction]), turbine_coordinates_array)
    processed_x, processed_y = process_layout_V1(rotated_x[0][0],rotated_y[0][0],rotor_diameter)
    yaw_array = np.zeros(nturbs)
    for i in range(nturbs):
        yaw_array[i] = get_yaw_angles_V1(processed_x[i], processed_y[i], left_x, top_left_y, right_x, top_right_y, top_left_yaw,
                    top_right_yaw, bottom_left_yaw, bottom_right_yaw)

    return yaw_array


def calculate_power_V1(turbine_x, turbine_y, fi, rotor_diameter):

    yaw = geometric_yaw_V1(turbine_x, turbine_y, 270.0, rotor_diameter)

    yaw_angles = np.zeros((1,1,len(turbine_x)))
    yaw_angles[0, 0, :] = yaw[:]
        
    fi.calculate_wake(yaw_angles=yaw_angles)
    farm_power = fi.get_farm_power()

    return farm_power[0, 0]


def process_layout_V2(turbine_x, turbine_y, rotor_diameter, n_nearest=5, spread=0.08):
    """
    returns the distance from each turbine to the nearest downstream waked turbine
    normalized by the rotor diameter. Right now "waked" is determind by da Jensen-like
    wake spread, but this could/should be modified to be the same as the trapezoid rule
    used to determine the yaw angles.

    turbine_x: turbine x coords (rotated)
    turbine_y: turbine y coords (rotated)
    rotor_diameter: turbine rotor diameter (float)
    spread=0.1: Jensen alpha wake spread value
    """
    nturbs = len(turbine_x)
    dx_nearest = np.zeros((nturbs, n_nearest))
    dy_nearest = np.zeros((nturbs, n_nearest))

    for turbine_creating_the_wake in range(nturbs):
        dx_1d = np.zeros(nturbs)
        dy_1d = np.zeros(nturbs)
        nearest_downstream_x = np.zeros(nturbs) + 1E6
        nearest_downstream_y = np.zeros(nturbs) + 1E6
        for turbine_getting_waked in range(nturbs):
            dx_1d[turbine_getting_waked] = turbine_x[turbine_getting_waked] - turbine_x[turbine_creating_the_wake]
            dy_1d[turbine_getting_waked]  = turbine_y[turbine_getting_waked]- turbine_y[turbine_creating_the_wake]
            if dx_1d[turbine_getting_waked] > 0.0:
                r = spread*dx_1d[turbine_getting_waked] + rotor_diameter/2.0
                if abs(dy_1d[turbine_getting_waked]) < (r+rotor_diameter/2.0):
                    nearest_downstream_x[turbine_getting_waked] = dx_1d[turbine_getting_waked]
                    nearest_downstream_y[turbine_getting_waked] = dy_1d[turbine_getting_waked]
        

        x_sort_nearest = np.argsort(nearest_downstream_x)
        dx_nearest[turbine_creating_the_wake, :] = nearest_downstream_x[x_sort_nearest][:n_nearest]
        dy_nearest[turbine_creating_the_wake, :] = nearest_downstream_y[x_sort_nearest][:n_nearest]

        dx_nearest[turbine_creating_the_wake, :] = np.where(dx_nearest[turbine_creating_the_wake, :] == 1E6, 0.0, dx_nearest[turbine_creating_the_wake, :])
        dy_nearest[turbine_creating_the_wake, :] = np.where(dy_nearest[turbine_creating_the_wake, :] == 1E6, 0.0, dy_nearest[turbine_creating_the_wake, :])
        
    return dx_nearest, dy_nearest


def calculate_power_V2(turbine_x, turbine_y, fi, rotor_diameter, yaw_function, n_nearest):

    dx, dy = process_layout_V2(turbine_x, turbine_y, rotor_diameter, n_nearest=n_nearest, spread=0.08)
    dxdy = np.hstack((dx/rotor_diameter, dy/rotor_diameter))
    yaw = yaw_function.predict(dxdy)

    yaw_angles = np.zeros((1,1,len(turbine_x)))
    yaw_angles[0, 0, :] = yaw[:]
        
    fi.calculate_wake(yaw_angles=yaw_angles)
    farm_power = fi.get_farm_power()

    return farm_power[0, 0]


def calculate_power(turbine_x, turbine_y, fi, wind_speeds, rotor_diameter, yaw_function, baseline_yaw, n_nearest):

    fi.reinitialize(layout_x=turbine_x, layout_y=turbine_y, wind_speeds=wind_speeds)

    power_0 = calculate_power_0(fi)
    power_baseline = calculate_power_baseline(fi, baseline_yaw)
    power_V1 = calculate_power_V1(turbine_x, turbine_y, fi, rotor_diameter)
    power_V2 = calculate_power_V2(turbine_x, turbine_y, fi, rotor_diameter, yaw_function, n_nearest)

    return power_0, power_baseline, power_V1, power_V2    


if __name__=="__main__":

    n_nearest = 5
    function_filename = "model_nearest%s"%n_nearest #This model is created in the V2geometric_yaw file
    # function_filename = "model_nearest3_nospread"
    loaded_clf = joblib.load(function_filename)

    fi = FlorisInterface("cc.yaml")
    wind_directions = [270.0]
    fi.reinitialize(wind_directions=wind_directions)
    rotor_diameter = fi.floris.farm.rotor_diameters[0]

    power_0_array = []
    power_baseline_array = []
    power_V1_array = []
    power_V2_array = []
    
    counter = 0
    V1_sum = 0.0
    V2_sum = 0.0
    current_path = pathlib.Path(__file__).parent.resolve()


    """
    Loop through all of the data
    """
    # for path, currentDirectory, files in os.walk(os.path.join(current_path, "yaw_data_grid")):
    #     for file in files:
            
    #         filename = os.path.join(path, file)
    #         first_letter = file[0]

    #         if first_letter == "w":
    #             if file[2] != "1":
    #                 ws = float(file[2])
    #             else:
    #                 ws = float(file[2]+file[3])
    #         else:
    #             ws = 10.0

    #         data = np.load(filename)
    #         turbine_x = data['turbine_x']
    #         turbine_y = data['turbine_y']
    #         baseline_yaw_angles = data["yaw_angles"]

    #         nturbs = len(turbine_x)

    #         if nturbs == 25:
    #             counter += 1
    #             wind_speeds = [ws]
    #             power_0, power_baseline, power_V1, power_V2  = calculate_power(turbine_x, turbine_y, fi, wind_speeds, rotor_diameter, loaded_clf, baseline_yaw_angles, n_nearest)
    #             power_0_array.append(power_0)
    #             power_baseline_array.append(power_baseline)
    #             power_V1_array.append(power_V1)
    #             power_V2_array.append(power_V2)
    #             print(power_V2/power_V1)
    #             V1_sum += power_V1
    #             V2_sum += power_V2

    #             if counter > 91:
    #                 break
            
    # print("THE END: ", V2_sum / V1_sum)

    """
    Just loop through one wind speed for one grid
    """
    rotation = np.linspace(-90.0, 90.0, 91)
    temp_path = os.path.join(current_path, "yaw_data_grid")
    full_path = os.path.join(temp_path, "ws6") 
    # for i in range(len(rotation)):  
    #         filename = os.path.join(full_path, "ws6grid_5_5_%s.npz"%rotation[i])
    #         data = np.load(filename)
    #         turbine_x = data['turbine_x']
    #         turbine_y = data['turbine_y']
    #         baseline_yaw_angles = data["yaw_angles"]

    #         nturbs = len(turbine_x)

    #         wind_speeds = [6.0]
    #         power_0, power_baseline, power_V1, power_V2  = calculate_power(turbine_x, turbine_y, fi, wind_speeds, rotor_diameter, loaded_clf, baseline_yaw_angles, n_nearest)
    #         power_0_array.append(power_0)
    #         power_baseline_array.append(power_baseline)
    #         power_V1_array.append(power_V1)
    #         power_V2_array.append(power_V2)
    #         print(power_V2/power_V1)
    #         V1_sum += power_V1
    #         V2_sum += power_V2

    """
    Running all the power calculations takes a little while, saving to an array for easier plotting
    """
    power_0_array = [6900731.316752293, 7682274.893779839, 9493834.883475024, 11548828.65502961, 13548511.239289856, 15243064.243682953, 16204780.19185489, 16087911.15038129, 15504787.224639112, 15102420.22323832, 15354200.406988056, 15481659.673621764, 14302531.695581596, 13133322.7287303, 13486846.197130125, 14911406.52636456, 15601521.00963955, 15731785.831541033, 15741550.160646698, 15130859.790858768, 13704267.952966163, 11683777.290809907, 10146409.566692336, 10157477.9889832, 11719066.147630448, 13754901.095198218, 15183949.066309702, 15784578.499909082, 15731644.621264918, 15586812.013115814, 14880366.551342404, 13464638.434298776, 13142923.92671329, 14338810.918017942, 15497513.595363636, 15352187.535630576, 15107313.108208092, 15519637.927970083, 16050586.28870288, 16081634.235246243, 15163660.50782581, 13472782.47568068, 11473252.948784668, 9448211.072643753, 7664793.207728036, 6902034.978391536, 7682274.893779837, 9493834.883475022, 11548828.655029617, 13548511.239289854, 15243064.243682953, 16204780.191854892, 16087911.150381284, 15504873.96830625, 15102426.45805225, 15354200.439528344, 15481591.574341603, 14302531.695581596, 13133322.728730299, 13486846.197130127, 14911431.485347128, 15601521.00963955, 15731785.831541035, 15741524.296795266, 15130859.790858768, 13704220.630611219, 11683854.019762402, 10146409.566692336, 10157477.988983206, 11718709.353789376, 13754879.536169069, 15183899.789643431, 15784625.820379598, 15731644.621264922, 15586812.013115814, 14880390.00080107, 13464341.270726552, 13142923.92671329, 14338810.918017928, 15497513.595363632, 15352186.46514539, 15107313.108208092, 15519637.927970083, 16050586.288702877, 16081683.714056415, 15163658.983814966, 13472787.489866763, 11473218.523261854, 9448203.870595925, 7664793.207728039, 6899725.042044637]
    power_baseline_array = [8180709.557990591, 9907082.989297884, 11650494.795777652, 13475582.512432165, 15009096.35800206, 15918832.56885084, 16359774.859800784, 16088109.128855813, 15830142.78453159, 15326836.02629905, 15485969.95284623, 15583472.933972875, 15131718.151047254, 14109958.960458605, 14607421.89172899, 15411511.007511025, 15680129.408601051, 15764694.28985218, 15836203.139014162, 15626419.590618914, 14873366.259055302, 13551306.41717992, 11903412.040001204, 11966663.796687583, 13587788.27783251, 14909128.77472099, 15658171.553239822, 15841574.100376159, 15732014.699833106, 15673742.471385775, 15400162.83707264, 14577436.854449656, 14157762.50201439, 15143887.804438697, 15592234.51993184, 15481374.634844676, 15353283.558965702, 15841131.784731332, 16192025.474727213, 16342803.212760907, 15877307.654577717, 14951294.369063586, 13463663.848646581, 11656616.177231902, 9834567.542332435, 8190218.224048593, 9907082.98816017, 11650494.801761625, 13475531.677975435, 15009096.358709428, 15918832.62207303, 16359774.859391876, 16089083.084398698, 15829727.593929132, 15326824.78404816, 15485970.00420974, 15583149.508909497, 15131718.073653331, 14109958.934957895, 14607421.874155922, 15412089.458736366, 15680129.406890081, 15764775.448328251, 15832438.014797049, 15626419.592061454, 14866242.028362008, 13516752.355612073, 11903412.03902608, 11966663.801585522, 13632157.799551234, 14896595.655611651, 15647830.365583537, 15842246.769561687, 15738764.392413948, 15673742.467249043, 15398486.388603615, 14584321.381583026, 14157762.534224777, 15143887.804003455, 15592234.515029086, 15481374.42622318, 15353283.54617659, 15841122.838986179, 16192025.483242175, 16341616.951582307, 15877241.950774007, 14954101.428176, 13469922.522362685, 11684995.865035268, 9834567.63822054, 8128675.39383168]
    power_V1_array = [7889271.273156996, 9730281.193901513, 11511732.294900246, 13432951.609605635, 14593308.511683961, 14614477.698728781, 16359774.859800784, 16088109.128855813, 15830142.78453159, 15326836.02629905, 15485969.95284623, 15100334.592158426, 15036738.866417188, 13980987.792082243, 14541295.52536621, 15276356.05156741, 15680129.408601051, 15764694.28985218, 15836203.139014162, 14896712.919062577, 14679398.451806752, 13453873.62146773, 11706693.27178998, 11767860.764598656, 13487469.108257182, 14695526.228949213, 14913111.297940675, 15841574.100376159, 15732014.699833106, 15673742.471385775, 15262750.08329132, 14508044.86088074, 14044508.339965958, 15042283.998002991, 15126054.77869714, 15481374.634844676, 15353283.558965702, 15841131.784731332, 16192025.474727213, 16342803.212760907, 14623248.834958695, 14584179.559345817, 13378203.812549936, 11457453.931860741, 9671077.366804855, 7921721.161789954, 9730281.193901509, 11511732.294900248, 13432951.60960564, 14593308.511683956, 14614477.69872878, 16359774.859391876, 16089083.084398698, 15829727.593929132, 15326824.78404816, 15485970.00420974, 15109848.538557086, 15036738.866417188, 13980987.792082239, 14541295.525366213, 15274349.381050022, 15680129.406890081, 15764775.448328251, 15832438.014797049, 14896712.919062577, 14651411.416035676, 13428729.633190693, 11706693.271789983, 11767860.764598668, 13498956.084265284, 14660597.26762003, 14904289.280665668, 15842246.769561687, 15738764.392413948, 15673742.467249043, 15262138.99949008, 14508113.91599512, 14044508.339965966, 15042283.998002982, 15126054.778697144, 15481374.42622318, 15353283.54617659, 15841122.838986179, 16192025.483242175, 16341616.951582307, 14623170.739023631, 14589676.14925384, 13382440.96628952, 11459568.85580031, 9671077.366804857, 7401136.818318239]
    power_V2_array = [7955138.595319775, 9889883.765912354, 11646186.57843619, 13403866.521677548, 14995788.416336246, 15922638.851845479, 16371396.97521469, 16181903.763521003, 15776961.879700985, 15277700.763303293, 15434489.122287232, 15570593.940756246, 15079605.548097942, 13907995.067352764, 14526670.962245066, 15375167.336398236, 15674528.55335227, 15732389.467970472, 15775119.642234284, 15598556.653494835, 14831785.141886352, 13471826.077594392, 11170968.770190151, 11831601.227200452, 13504955.769428087, 14850345.787364267, 15640984.773434427, 15844038.7213481, 15744574.595280766, 15670272.724052737, 15358794.11201181, 14474137.37395057, 14074644.121833421, 15105385.297496067, 15584155.66584466, 15411780.208012417, 15348253.798310636, 15776748.86449776, 16177415.807284473, 16321773.540611735, 16005865.148026459, 14941700.558542484, 13342412.28602387, 11570099.700200403, 9795049.506291067, 7966226.489115909, 9889883.76591235, 11646186.57843619, 13403866.521677554, 14995788.416336246, 15922638.851845477, 16371396.975214694, 16181903.763521003, 15776941.141448807, 15277709.81856062, 15434489.147846395, 15570070.354805203, 15079605.548097944, 13907995.067352759, 14526670.96224507, 15376022.594622688, 15674528.553352268, 15732389.467970474, 15776017.077771228, 15598556.653494837, 14826566.69236949, 13447753.049498117, 11170968.770190157, 11831601.227200462, 13511190.155485725, 14841397.2611959, 15630593.739316028, 15845051.62560316, 15744574.595280768, 15670272.724052737, 15360437.16740279, 14473111.065838283, 14074644.12183343, 15105385.297496062, 15584155.66584466, 15411779.6162819, 15348253.798310636, 15776748.864497758, 16177415.807284469, 16321813.36727661, 16005772.2886243, 14944276.2565395, 13341927.276997196, 11572873.151801748, 9795049.506291071, 7917286.490665877]
    
    

    """
    Create figures for the TORQUE conference abstract
    """
    print(sum(power_baseline_array) / sum(power_0_array))
    print(sum(power_V1_array) / sum(power_0_array))
    print(sum(power_V2_array) / sum(power_0_array))
    power_0_array = np.array(power_0_array)
    power_baseline_array = np.array(power_baseline_array)
    power_V1_array = np.array(power_V1_array)
    power_V2_array = np.array(power_V2_array)
    
    # print(repr(power_0_array))
    # print(repr(power_baseline_array))
    # print(repr(power_V1_array))
    # print(repr(power_V2_array))
    # print("THE END: ", V2_sum / V1_sum)

    N = 46
    plt.figure(figsize=(6,2))
    lw = 1.0
    

    # plt.figure(figsize=(3,3))
    i1 = 0
    i2 = 46
    ax2 = plt.subplot(132)
    ax2.plot((power_baseline_array[i1:i2]-power_0_array[:N]) / power_0_array[:N] * 100.0, "--", label="fully optimized", color="C0", linewidth=lw)
    ax2.plot((power_V1_array[i1:i2]-power_0_array[:N]) / power_0_array[:N] * 100.0, "--", label="geometric yaw", color="k", linewidth=lw)
    ax2.plot((power_V2_array[i1:i2] - power_0_array[:N]) / power_0_array[:N] * 100.0, "-", label="machine learning", color="C1", linewidth=lw)

    ax3 = plt.subplot(133)
    ax3.plot((power_V2_array[i1:i2] - power_V1_array[:N])/power_V1_array[:N] * 100, "-", color="C1", linewidth=lw)
    ax3.set_yticks([-5, 0, 5, 10])

    # # ax2.plot((power_baseline_array[i1:i2]-power_0_array[:N]), "-", color="C1")
    # ax2.plot((power_V1_array[i1:i2]-power_baseline_array[:N]) / power_baseline_array[:N] * 100, "-", color="C0")
    # ax2.plot((power_V2_array[i1:i2] - power_baseline_array[:N]) / power_baseline_array[:N] * 100, "-", color="C1")
    # # ax2.plot((power_0_array[i1:i2] - power_baseline_array[:N]) / power_baseline_array[:N] * 100, "-", color="C3")

    # ax2.set_ylim(0.9,1.0025)

    
    ax1 = plt.subplot(131)
    # ax1.plot(power_baseline_array[:N] / max(power_baseline_array[:N]), "-", color="C0", label="fully\noptimized", linewidth=lw)
    # ax1.plot(power_V2_array[:N] / max(power_baseline_array[:N]), "-", color="C1", label="machine\nlearning", linewidth=lw)
    # ax1.plot(power_V1_array[:N] / max(power_baseline_array[:N]), "-", color="C3", label="geometric\nyaw", linewidth=lw)
    ax1.plot(power_0_array[:N] / max(power_0_array[:N]), "-", color="k", label="no yaw", linewidth=lw)
    ax1.set_ylim(0.6,1.01)

    lp=-2
    ax1.set_xlabel("wind direction", fontsize=8)
    ax1.set_ylabel("normalized baseline power", fontsize=8)

    ax2.set_xlabel("wind direction", fontsize=8)
    # ax2.set_ylabel("power comparison to\nfully optimized farm (%)")
    ax2.set_ylabel("power increase from\nwake steering (%)", fontsize=8)

    ax3.set_xlabel("wind direction", fontsize=8)
    # ax3.set_ylabel("machine learning/\ngeometric yaw", fontsize=8)
    ax3.set_ylabel("% improvement geometric\nyaw to machine learning", fontsize=8, labelpad=lp)

    ax2.legend(loc="upper center", fontsize=6, frameon=False)

    ax1.tick_params(axis='both', which='major', labelsize=8)
    ax1.tick_params(axis='both', which='minor', labelsize=8)
    ax2.tick_params(axis='both', which='major', labelsize=8)
    ax2.tick_params(axis='both', which='minor', labelsize=8)
    ax3.tick_params(axis='both', which='major', labelsize=8)
    ax3.tick_params(axis='both', which='minor', labelsize=8)

    a = 14
    d = 68
    ax1.text(a, 0.62, "A", fontsize=12, color="black", weight="bold")
    ax1.text(d + a, 0.62, "B", fontsize=12, color="black", weight="bold")
    ax1.text(2*d + a, 0.62, "C", fontsize=12, color="black", weight="bold")


    plt.subplots_adjust(top=0.98, bottom=0.2, left=0.1, right=0.99, wspace=0.4)
    # ax1.grid()
    # ax2.grid()
    plt.savefig("torque_abstract_performance.pdf")
    plt.show()

            # print(farm_power)

    # for path, currentDirectory, files in os.walk(os.path.join(current_path, "yaw_data_grid")):
    #     for file in files:
    #         filename = os.path.join(path, file)
    #         data = np.load(filename)

    #         turbine_x = data['turbine_x']
    #         turbine_y = data['turbine_y']
    #         yaw_angles = data["yaw_angles"]
    #         yaw = np.append(yaw, yaw_angles)

    #         dx, dy = process_layout(turbine_x, turbine_y, rotor_diameter, n_nearest=n_nearest, spread=0.08)

    #         dxdy = np.hstack((dx/rotor_diameter, dy/rotor_diameter))
    #         dxdy_total = np.vstack((dxdy_total, dxdy))

