'''
Created on Dec 10, 2017

Run a single experiment with either the predictive or the self trigger (to be defined in Defs.py).

@author: Dominik Baumann
MPI-IS, ICS
dbaumann(at)tuebingen.mpg.de

Copyright 2017 Max Planck Society. All rights reserved.
'''

from VehiclePlatoon import *


def run_single_experiment(trigger):
    # Initiliaze platoon and vehicles
    exp_platoon = platoon(x_init)
    vehicles = [vehicle(x_des_init, x_init, i, delta) for i in range(num_veh)]
    # Initialize matrices for plotting
    state_out = matlib.zeros((2 * num_veh, num_it))
    u_out = matlib.zeros((num_veh, num_it))
    comm_out = np.zeros((num_veh, num_it))
    t = np.arange(0, num_it * Ts, Ts)
    # Initialize input
    u = np.zeros((num_veh, 1))
    # Create noise matrices
    v = np.random.uniform(-v_max, v_max, (num_veh, num_it))
    w = np.random.uniform(-w_max, w_max, (num_veh, num_it))
    for i in range(0, num_it):
        # Propagate system and check for accidents
        y = exp_platoon.propagate(u, v[:, i][np.newaxis].T, w[:, i][np.newaxis].T)
        if exp_platoon.check_for_accidents():
            print('accident')
            break

        # Safe current state and input for plotting
        state_out[:, i] = exp_platoon.state[:, 0]
        u_out[:, i] = u
        index = 0
        for obj in vehicles:
            # Estimate local state and predict state of other vehicles
            obj.predict()
            obj.estimate_state(y)
            # Every vehicle computes its own local input
            u[obj.veh_id, 0] = obj.u[obj.veh_id, 0]
            # Check trigger
            if trigger == 1:
                obj.gamma_pred(i)
            else:
                obj.gamma_self()
        for obj in vehicles:
            if obj.gamma == 1:
                # Safe triggering decision
                comm_out[index, i] = 1
                # In case of triggering, reset covariance matrices and actualize beliefs of state and desired state
                obj.state_loc_pred = copy.deepcopy(obj.state_loc)
                obj.state_pred = copy.deepcopy(obj.state)
                obj.x_des_pred = copy.deepcopy(obj.x_des)
                obj.klm_loc_pred.Pol = copy.deepcopy(obj.klm_loc.Pcl)
                for veh in vehicles:
                    # If no packet drop, information is communicated to all other vehicles
                    if np.random.uniform() > pdr:
                        veh.state_pred[2 * index, 0] = obj.state[2 * index, 0]
                        veh.state_pred[2 * index + 1, 0] = obj.state[2 * index + 1, 0]
                        veh.state[2 * index, 0] = obj.state[2 * index, 0]
                        veh.state[2 * index + 1, 0] = obj.state[2 * index + 1, 0]
                        veh.x_des = copy.deepcopy(obj.x_des)
                        veh.x_des_pred = copy.deepcopy(obj.x_des)
                        # rederive control in case of communication
                        u[veh.veh_id, 0] = veh.calc_input(veh.state, veh.x_des)[veh.veh_id, 0]

            index += 1
    mpl.subplot(2, 1, 1)
    mpl.plot(t, state_out[::2, :].T)
    mpl.grid()
    mpl.subplot(2, 1, 2)
    mpl.plot(t, comm_out.transpose())
    mpl.grid()
    mpl.show()


run_single_experiment(trigger)
