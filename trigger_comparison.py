'''
Created on Dec 10, 2017

Comparing predictive and self trigger in Monte Carlo simulations of vehicle platooning

@author: Dominik Baumann
MPI-IS, ICS
dbaumann(at)tuebingen.mpg.de
'''

from VehiclePlatoon import *


def run_simulation_changing_surface():
    # Number of experiments
    exp = 11

    # Number of simulations per experiment
    mc = 100

    # Initialize arrays
    err_pred = np.zeros((1, mc))
    err_pred_mean = np.zeros((1, exp))
    err_pred_var = np.zeros((1, exp))
    rate_pred = np.zeros((1, mc))
    rate_pred_out = np.zeros((1, exp))
    err_self = np.zeros((1, mc))
    err_self_mean = np.zeros((1, exp))
    err_self_var = np.zeros((1, exp))
    rate_self = np.zeros((1, mc))
    rate_self_out = np.zeros((1, exp))
    state_out = matlib.zeros((2 * num_veh, num_it))
    state_pred_out = matlib.zeros((2 * num_veh, num_it))
    comm_out = np.zeros((num_veh, num_it))

    # Input vector
    u = np.zeros((num_veh, 1))

    # Change delta for every experiment
    for k in range(0, exp):
        print(k)
        if k == 0:
            delta_self = 0
            delta_pred = 0
        elif k == 1:
            delta_self = 0.0005
            delta_pred = 0.0005
        elif k == 2:
            delta_self = 0.001
            delta_pred = 0.001
        elif k == 3:
            delta_self = 0.002
            delta_pred = 0.0013
        elif k == 4:
            delta_self = 0.003
            delta_pred = 0.0016
        elif k == 5:
            delta_self = 0.005
            delta_pred = 0.002
        elif k == 6:
            delta_self = 0.0075
            delta_pred = 0.01
        elif k == 7:
            delta_self = 0.01
            delta_pred = 0.05
        elif k == 8:
            delta_self = 0.02
            delta_pred = 0.1
        elif k == 9:
            delta_self = 0.025
            delta_pred = 0.5
        elif k == 10:
            delta_self = 0.03
            delta_pred = 10

        for j in range(0, mc):
            # Create noise matrices
            v = np.random.uniform(-v_max, v_max, (num_veh, num_it))
            w = np.random.uniform(-w_max, w_max, (num_veh, num_it))
            # Initialize the platoon and the vehicles
            exp_platoon = platoon(x_init)
            vehicles = [vehicle(x_des_init, x_init, i, delta_pred) for i in range(num_veh)]
            for i in range(0, num_it):
                # Propagate system and check for accidents
                y = exp_platoon.propagate(u, v[:, i][np.newaxis].T, w[:, i][np.newaxis].T)
                if exp_platoon.check_for_accidents():
                    print('accident pred')
                    break
                # After 200m change A and B matrices (wet road after tunnel)
                for n in range(0, num_veh):
                    if exp_platoon.state[2 * n, 0] > 200 and state_out[2 * n, i - 1] < 200:
                        exp_platoon.Ad[2 * n, 2 * n + 1] = 1.5 * exp_platoon.Ad[2 * n, 2 * n + 1]
                        exp_platoon.Bd[2 * n:2 * n + 2, n] = 0.5 * exp_platoon.Bd[2 * n:2 * n + 2, n]
                # Safe current state for plotting
                state_out[:, i] = exp_platoon.state[:, 0]
                index = 0
                for obj in vehicles:
                    # Estimate local state and predict state of other vehicles
                    obj.predict()
                    obj.estimate_state(y)
                    # Every vehicle computes its own local input
                    u[obj.veh_id, 0] = obj.u[obj.veh_id, 0]
                    # Check trigger
                    obj.gamma_pred(i)
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
                    else:
                        comm_out[index, i] = 0
                    index += 1
                state_pred_out[:, i] = vehicles[0].state_pred[:]
            err_pred[0, j] = 0
            rate_pred[0, j] = 0
            # Derive error in distance and velocity
            for l in range(0, num_veh):
                err_pred[0, j] = err_pred[0, j] + np.abs(np.linalg.norm(state_out[2 * l + 1, 20::] - des_vel, ord=np.inf) / ((state_out.shape[1] - 20) * num_veh))
                rate_pred[0, j] = rate_pred[0, j] + np.sum(comm_out[l, 20::]) / ((state_out.shape[1] - 20) * num_veh)
                if l < num_veh - 2:
                    err_pred[0, j] = err_pred[0, j] + np.abs(np.linalg.norm(state_out[2 * l, 20::] - state_out[2 * l + 2, 20::] - dist, ord=np.inf) / ((state_out.shape[1] - 20) * (num_veh - 1)))

            # Initialize platoon and vehicles a second time for simulatin with self trigger
            exp_platoon = platoon(x_init)
            vehicles = [vehicle(x_des_init, x_init, i, delta_self) for i in range(num_veh)]

            for i in range(0, num_it):
                # Propagate system and check for accidents
                y = exp_platoon.propagate(u, v[:, i][np.newaxis].T, w[:, i][np.newaxis].T)
                if exp_platoon.check_for_accidents():
                    print('accident self')
                    break
                # After 200m change A and B matrices (wet road after leaving a tunnel)
                for n in range(0, num_veh):
                    if exp_platoon.state[2 * n, 0] > 200 and state_out[2 * n, i - 1] < 200:
                        exp_platoon.Ad[2 * n, 2 * n + 1] = 1.5 * exp_platoon.Ad[2 * n, 2 * n + 1]
                        exp_platoon.Bd[2 * n:2 * n + 2, n] = 0.5 * exp_platoon.Bd[2 * n:2 * n + 2, n]
                # Safe current state for plotting
                state_out[:, i] = exp_platoon.state[:, 0]
                index = 0
                for obj in vehicles:
                    # Estimate local and predict local state for every vehicle
                    obj.predict()
                    obj.estimate_state(y)
                    # Derive control inputs
                    u[obj.veh_id, 0] = obj.u[obj.veh_id, 0]
                    # Check trigger
                    obj.gamma_self()
                for obj in vehicles:
                    if obj.gamma == 1:
                        # Safe triggering decision
                        comm_out[index, i] = 1
                        # In case of triggering, reset covariance matrices and actualize beliefs of state and desired state
                        obj.state_pred = copy.deepcopy(obj.state_loc)
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
                    else:
                        comm_out[index, i] = 0

                    index += 1
                state_pred_out[:, i] = vehicles[0].state_pred[:]
            err_self[0, j] = 0
            rate_self[0, j] = 0
            # Derive error in distance and velocity
            for l in range(0, num_veh):
                err_self[0, j] = err_self[0, j] + np.abs(np.linalg.norm(state_out[2 * l + 1, 20::] - des_vel, ord=np.inf) / ((state_out.shape[1] - 20) * num_veh))
                rate_self[0, j] = rate_self[0, j] + np.sum(comm_out[l, 20::]) / ((state_out.shape[1] - 20) * num_veh)
                if l < num_veh - 2:
                    err_self[0, j] = err_self[0, j] + np.abs(np.linalg.norm(state_out[2 * l, 20::] - state_out[2 * l + 2, 20::] - dist, ord=np.inf) / ((state_out.shape[1] - 20) * (num_veh - 1)))
        # Take mean of Monte Carlo simulations
        err_pred_mean[0, k] = np.mean(err_pred)
        err_pred_var[0, k] = np.var(err_pred)
        rate_pred_out[0, k] = np.mean(rate_pred)
        err_self_mean[0, k] = np.mean(err_self)
        err_self_var[0, k] = np.var(err_self)
        rate_self_out[0, k] = np.mean(rate_self)
    mpl.errorbar(rate_pred_out.flatten(), err_pred_mean.flatten(), yerr=err_pred_var.flatten(), color="g")
    mpl.errorbar(rate_self_out.flatten(), err_self_mean.flatten(), yerr=err_self_var.flatten(), color="b")
    # mpl.plot(rate_pred_out.flatten(), err_pred_mean.flatten())
    # mpl.plot(rate_self_out.flatten(), err_self_mean.flatten())
    tikz_save('changing_surface.tex')
    mpl.grid()
    mpl.show()


if __name__ == '__main__':
    run_simulation_changing_surface()
