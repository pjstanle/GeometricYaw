import numpy as np
import pandas as pd
import os
import pathlib
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
import joblib
import math


def process_layout(turbine_x, turbine_y, rotor_diameter, array_size=None, n_nearest=5, spread=0.08):
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
    if array_size == None:
        array_size = len(turbine_x)
    nturbs = len(turbine_x)
    dx = np.zeros((nturbs, array_size))
    dy = np.zeros((nturbs, array_size))
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
        
        x_sort = np.argsort(dx_1d)
        dx[turbine_creating_the_wake, :nturbs] = dx_1d[x_sort]
        dy[turbine_creating_the_wake, :nturbs] = dy_1d[x_sort]

        x_sort_nearest = np.argsort(nearest_downstream_x)
        dx_nearest[turbine_creating_the_wake, :] = nearest_downstream_x[x_sort_nearest][:n_nearest]
        dy_nearest[turbine_creating_the_wake, :] = nearest_downstream_y[x_sort_nearest][:n_nearest]

        dx_nearest[turbine_creating_the_wake, :] = np.where(dx_nearest[turbine_creating_the_wake, :] == 1E6, 0.0, dx_nearest[turbine_creating_the_wake, :])
        dy_nearest[turbine_creating_the_wake, :] = np.where(dy_nearest[turbine_creating_the_wake, :] == 1E6, 0.0, dy_nearest[turbine_creating_the_wake, :])
        #TODO Need to correct this. I want this sorted from left to right of closest downstream to furthest downstream, to everything else

        # dx[turbine_creating_the_wake, :nturbs] = dx_1d
        # dy[turbine_creating_the_wake, :nturbs] = dy_1d

    return dx_nearest, dy_nearest


if __name__=="__main__":
    
    # check that it is returning values
    # turbine_x = np.array([0,700,1400,2100])
    # turbine_y = np.array([0,50,0,-100])

    # dx, dy = process_layout(turbine_x, turbine_y)



    # turbs_array = [10,20,30]
    # # turbs_array = [30]
    # array_size = max(turbs_array)
    # spacing_array = [10.0,9.0,8.0,7.0]
    # # spacing_array = [7.0]
    # nruns = 10
    # ws_array = [6,8,10]
    # # ws_array = [10]

    array_size = 40
    rotor_diameter = 126.4
    n_nearest = 5

    dxdy_total = np.zeros((0, 2*n_nearest))
    yaw = np.array([])

    current_path = pathlib.Path(__file__).parent.resolve()

    """
    Process all of the layouts to get dx, dy, and optimal yaw angles into the training matrix
    """

    # for path, currentDirectory, files in os.walk(os.path.join(current_path, "yaw_data")):
    #     for file in files:
    #         filename = os.path.join(path, file)
    #         data = np.load(filename)

    #         turbine_x = data['turbine_x']
    #         turbine_y = data['turbine_y']
    #         yaw_angles = data["yaw_angles"]
    #         yaw = np.append(yaw, yaw_angles)

    #         dx, dy = process_layout(turbine_x, turbine_y, rotor_diameter, array_size=array_size, n_nearest=n_nearest, spread=0.05)

    #         dxdy = np.hstack((dx/rotor_diameter, dy/rotor_diameter))
    #         dxdy_total = np.vstack((dxdy_total, dxdy))

    # for path, currentDirectory, files in os.walk(os.path.join(current_path, "yaw_data_grid")):
    #     for file in files:
    #         filename = os.path.join(path, file)
    #         data = np.load(filename)

    #         turbine_x = data['turbine_x']
    #         turbine_y = data['turbine_y']
    #         yaw_angles = data["yaw_angles"]
    #         yaw = np.append(yaw, yaw_angles)

    #         dx, dy = process_layout(turbine_x, turbine_y, rotor_diameter, array_size=array_size, n_nearest=n_nearest, spread=0.05)

    #         dxdy = np.hstack((dx/rotor_diameter, dy/rotor_diameter))
    #         dxdy_total = np.vstack((dxdy_total, dxdy))

    """
    Save the processed data because this takes a little while
    """
    # np.savez("processed_data_nearest_nospread", dxdy=dxdy_total, yaw_angles=yaw)



    filename = os.path.join(current_path, "processed_data_nearest5.npz")
    data = np.load(filename)
    dxdy = data['dxdy']
    yaw = data["yaw_angles"]
    print(np.shape(dxdy))


    """
    train the model
    """
    clf = MLPRegressor(hidden_layer_sizes=200, max_iter=2000)
    X_train, X_test, y_train, y_test = train_test_split(dxdy, yaw, test_size=0.6, random_state=0)
    clf.fit(X_train, y_train)
    """
    save and evaluate the model
    """
    function_filename = "model_nearest5"
    # joblib.dump(clf, function_filename)
    loaded_clf = joblib.load(function_filename)

    # score = loaded_clf.score(X_train, y_train)
    # print("score: ", score)

    score = loaded_clf.score(X_test, y_test)
    print("score: ", score)

    

    """
    create figure for TORQUE conference abstract
    """
    # ntest = 10000
    # itest = fit_array[nfit:nfit + ntest]


    # test_data = yaw[itest]
    # test_fit = np.zeros(ntest)
    # # for i in range(ntest):
    # test_fit = clf.predict(dxdy[itest, :])
    N = 2000
    test_data = y_test[:N]
    test_fit = loaded_clf.predict(X_test[:N])

    polys = np.polyfit(test_data, test_fit, 1)
    print(polys)

    def poly_func(coeffs, x=np.array([-30.0, 30.0])):
        y = np.zeros_like(x)
        degree = len(coeffs) - 1
        for i in range(len(x)):
            for j in range(len(coeffs)):
                y[i] += coeffs[-(j+1)] * x[i]**j

        return x, y

    plt.figure(figsize=(3, 2.3))
    plt.plot(test_data, test_fit, "o", markersize=1)
    x, y = poly_func(polys)
    plt.plot(x, y, color="C1", linewidth=2)
    plt.xlabel("fully optimized", fontsize=8)
    plt.ylabel("machine learning\nprediction", fontsize=8)
    plt.grid()
    plt.title(r"$R^2$: %s"%(round(score, 3)), fontsize=8)
    plt.tight_layout()

    plt.xticks((-30,-20,-10,0,10,20,30))
    plt.yticks((-30,-20,-10,0,10,20,30))
    plt.gca().tick_params(axis='both', which='major', labelsize=8)
    plt.gca().tick_params(axis='both', which='minor', labelsize=8)
    # plt.axis("equal")
    # plt.savefig("torque_abstract_model.pdf")
    plt.show()


    # for i in range(np.shape(dxdy_total)[0]):
    #     test_x = np.zeros((1,60))
    #     test_x[0, :] = dxdy_total[i, :]
        # print(clf.predict(test_x))
                    

    