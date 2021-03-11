'''
Created on Nov 14, 2017

@author: Dominik Baumann
MPI-IS, ICS
dbaumann(at)tuebingen.mpg.de
'''

from Defs import *
import scipy.linalg

' Define an discrete LQR. System matrices are different from the global system (couplings)'


class controller:

    def __init__(self, num_veh, Ts):
        # Define continuous matrices with couplings
        self.A = np.zeros((2 * (num_veh - 1) + 1, 2 * (num_veh - 1) + 1))
        for i in range(0, num_veh - 1):
            self.A[2 * i + 1, 2 * i] = 1
            self.A[2 * i + 1, 2 * i + 2] = -1
        self.B = np.zeros((2 * num_veh - 1, num_veh))
        for i in range(0, num_veh):
            self.B[2 * i, i] = 1
        self.C = np.ones((2 * (num_veh - 1) + 1, 2 * (num_veh - 1) + 1))
        self.D = np.zeros((2 * num_veh - 1, num_veh))
        self.sysc = ctrl.ss(self.A, self.B, self.C, self.D)
        # Derive discrete matrices
        self.sysd = ctrl.sample_system(self.sysc, Ts)
        [self.Ad, self.Bd, self.Cd, self.Dd] = ctrl.ssdata(self.sysd)

    'Derive discrete-time LQR controller'

    def dlqr(self):
        # first, try to solve the ricatti equation
        X = np.matrix(scipy.linalg.solve_discrete_are(self.Ad, self.Bd, Q, R))

        # compute the LQR gain
        K = np.matrix(scipy.linalg.inv(self.Bd.T * X * self.Bd + R) * (self.Bd.T * X * self.Ad))

        eigVals, eigVecs = scipy.linalg.eig(self.Ad - self.Bd * K)

        return K, X, eigVals


' Define the whole platoon'


class platoon:

    def __init__(self, x_init):
        # Continuous matrices
        self.A = np.zeros((2 * num_veh, 2 * num_veh))
        self.B = np.zeros((2 * num_veh, num_veh))
        self.C = np.zeros((num_veh, 2 * num_veh))
        self.D = np.zeros((num_veh, num_veh))
        for i in range(0, num_veh):
            self.A[2 * i, 2 * i + 1] = 1
            self.B[2 * i + 1, i] = 1
            self.C[i, 2 * i ] = 1
        self.sysc = ctrl.ss(self.A, self.B, self.C, self.D)
        # Discretize system
        self.sysd = ctrl.sample_system(self.sysc, Ts)
        [self.Ad, self.Bd, self.Cd, self.Dd] = ctrl.ssdata(self.sysd)
        # Initialize state
        self.state = copy.deepcopy(x_init)

    'Propagate the system one discrete time step'

    def propagate(self, u, v, w):
        # state and measurement equation
        self.state = np.dot(self.Ad, self.state) + np.dot(self.Bd, (u + v))
        y = np.dot(self.Cd, self.state) + w
        return y

    'Check if inter-vehicle distance is below 0 for any two cars'

    def check_for_accidents(self):
        # If distance between vehicle i and i-1 is below zero, vehicles crashed
        for i in range(num_veh):
            if self.state[2 * i] - self.state[2 * i + 2] <= 0:
                return True
            else:
                return False


'Implementation of a Kalman filter'


class kalman:

    def __init__(self, A, B, C, x_des):
        # Save copies of system matrices
        self.A = copy.deepcopy(A)
        self.B = copy.deepcopy(B)
        self.C = copy.deepcopy(C)
        self.x_des = x_des
        # Initialize closed loop and open loop covariance matrices
        self.Pol = np.zeros_like(self.A)
        self.Pcl = np.zeros_like(self.A)
        # Define a matrix to transform Kalman filter state to control state
        self.K_conv = np.zeros((2 * num_veh - 1, 2 * num_veh))
        for i in range(0, num_veh - 1):
            self.K_conv[2 * i, 2 * i + 1] = 1
            self.K_conv[2 * i + 1, 2 * i] = 1
            self.K_conv[2 * i + 1, 2 * i + 2] = -1
        self.K_conv[-1, -1] = 1
        # Noise variance. Shape depends on system matrices (local or global).
        sigma_w = np.eye(B.shape[1]) * 0.1 / np.sqrt(3)
        sigma_v = np.eye(A.shape[0]) * 0.1 / np.sqrt(3)
        # Kalman matrices. Shape depends on system matrices (local or global).
        self.Qkal = np.power(sigma_v, 3) * np.eye(A.shape[0])
        self.Rkal = np.power(sigma_w, 3) * np.eye(B.shape[1])

    'Calculate the input if not given'

    def calc_input(self, state):
        return -np.dot(K, np.dot(self.K_conv, state) - self.x_des)

    'Prediction step'

    def OLupdate(self, state, u=np.array([np.infty, np.infty])):
        if sum(u) == np.infty:
            u = self.calc_input(state)
        x_upd = np.dot(self.A, state) + np.dot(self.B, u)
        self.Pol = np.dot(np.dot(self.A, self.Pol), self.A.transpose()) + self.Qkal
        return x_upd, self.Pol

    'Prediction step variance'

    def OLupdate_var(self):
        self.Pol = np.dot(np.dot(self.A, self.Pol), self.A.transpose()) + self.Qkal
        return self.Pol

    'Prediction step mean'

    def OLupdate_mean(self, state, u=np.array([np.infty, np.infty])):
        if sum(u) == np.infty:
            u = self.calc_input(state)
        x_upd = np.dot(self.A, state) + np.dot(self.B, u)
        return x_upd

    'Prediction and correction step'

    def CLupdate(self, state, meas, u=np.array([np.infty, np.infty])):
        if sum(u) == np.infty:
            u = self.calc_input(state)
        x_upd = np.dot(self.A, state) + np.dot(self.B, u)
        P_pred = np.dot(np.dot(self.A, self.Pcl), self.A.transpose()) + self.Qkal
        Skal = np.dot(np.dot(self.C, P_pred), self.C.transpose()) + self.Rkal
        Kkal = np.dot(np.dot(P_pred, self.C.transpose()), inv(Skal))
        self.Pcl = P_pred - np.dot(np.dot(Kkal, Skal), Kkal.transpose())
        x_upd_cl = x_upd + np.dot(Kkal, (meas - np.dot(self.C, x_upd)))
        self.K_kal = Kkal
        return x_upd_cl, self.Pcl

    'Get variance after correction step'

    def CLupdate_var(self):
        P_pred = np.dot(np.dot(self.A, self.Pcl), self.A.transpose()) + self.Qkal
        Skal = np.dot(np.dot(self.C, P_pred), self.C.transpose()) + self.Rkal
        Kkal = np.dot(np.dot(P_pred, self.C.transpose()), inv(Skal))
        self.K_kal = Kkal
        self.Pcl = P_pred - np.dot(np.dot(Kkal, Skal), Kkal.transpose())
        return self.Pcl


'Define the predictive trigger'


class predTrig:

    def __init__(self, klm_loc_cl, klm_loc_pred, delta, veh_id):
        # Define an array to store communication decisions
        self.gamma_arr = np.zeros(predHor)
        # Initialize last triggering instant
        self.latestTrig = 0
        # Local closed loop and open loop Kalman filter
        self.klm_loc_cl = klm_loc_cl
        self.klm_loc_pred = klm_loc_pred
        # Communication threshold
        self.delta = delta
        # Number of the vehicle in the platoon
        self.veh_id = veh_id

    'Check if communication is necessary'

    def trigger(self, step, state_klm, state_pred, P_pred_loc, P_klm_loc, K_loc):
        # Covariance matrices
        self.klm_loc_pred.Pol = copy.deepcopy(P_pred_loc)
        self.klm_loc_cl.Pol = copy.deepcopy(P_klm_loc)
        # Also create own variable for variance matrices
        self.Pcl_loc = P_klm_loc
        self.Pol_loc = P_pred_loc
        # Get local state
        self.state_loc = state_klm[2 * self.veh_id:2 * self.veh_id + 2]
        self.state_loc_pred = state_pred[2 * self.veh_id:2 * self.veh_id + 2]

        # Already trigger scheduled in the future?
        if self.latestTrig < step:
            mode = 1  # k > kappa_{k-1}
        else:
            mode = 2

        # Initialize matrices
        G = [np.zeros_like(self.klm_loc_cl.A) for _ in range(predHor)]
        G[0] = np.dot(self.klm_loc_cl.B, K_loc)
        I = [np.zeros_like(self.klm_loc_cl.Rkal) for _ in range(predHor)]
        K_klm = [np.zeros((2, 1)) for _ in range(predHor)]

        # No trigger scheduled
        if mode == 1:
            # Compute mean error
            err_mean = np.dot(np.power(self.klm_loc_cl.A - np.dot(self.klm_loc_cl.B, K_loc), predHor), self.state_loc - self.state_loc_pred)
            for i in range(0, predHor):
                self.Pol_loc = self.klm_loc_pred.OLupdate_var()
                self.Pcl_loc = self.klm_loc_cl.CLupdate_var()
                if i > 0:
                    G[i] = np.dot(self.klm_loc_cl.A, G[i - 1]) + np.dot(self.klm_loc_cl.B, np.dot(K_loc, np.linalg.matrix_power(self.klm_loc_cl.A - np.dot(self.klm_loc_cl.B, K_loc), i + 1)))
                I[i] = np.dot(self.klm_loc_cl.C, np.dot(self.klm_loc_cl.A, np.dot(self.Pcl_loc, np.dot(self.klm_loc_cl.A.T, self.klm_loc_cl.C.T)))) + np.dot(self.klm_loc_cl.C, np.dot(self.klm_loc_cl.Qkal, self.klm_loc_cl.C.T)) + self.klm_loc_cl.Rkal
                K_klm[i] = self.klm_loc_cl.K_kal
            inp_term = [np.dot(G[l], np.dot(K_klm[predHor - l - 1], np.dot(I[predHor - l - 1], np.dot(K_klm[predHor - l - 1].T, G[l].T)))) for l in range(0, predHor - 1)]
            err_var = self.Pol_loc + sum(inp_term)
            # Get mean and variance of the error
            err_mean = np.square(np.linalg.norm(self.state_loc - self.state_loc_pred))
            err_var = np.trace(err_var - self.Pcl_loc)
        # Future trigger already planned
        else:
            self.Pol_loc = self.Pcl_loc
            self.klm_loc_pred.Pol = self.Pcl_loc
            self.klm_loc_pred.Pcl = self.Pcl_loc
            num_CLupdates = self.latestTrig - step
            num_OLupdates = step + predHor - self.latestTrig
            for _ in range(0, num_CLupdates):
                self.Pol_loc = self.klm_loc_pred.CLupdate_var()
                self.klm_loc_pred.Pol = self.Pol_loc
                self.Pcl_loc = self.klm_loc_cl.CLupdate_var()
            for i in range(0, num_OLupdates):
                self.Pol_loc = self.klm_loc_pred.OLupdate_var()
                self.Pcl_loc = self.klm_loc_cl.CLupdate_var()
                if i > 0:
                    G[i] = np.dot(self.klm_loc_cl.A, G[i - 1]) + np.dot(self.klm_loc_cl.B, np.dot(K_loc, np.linalg.matrix_power(self.klm_loc_cl.A - np.dot(self.klm_loc_cl.B, K_loc), i + 1)))
                I[i] = np.dot(self.klm_loc_cl.C, np.dot(self.klm_loc_cl.A, np.dot(self.Pcl_loc, np.dot(self.klm_loc_cl.A.T, self.klm_loc_cl.C.T)))) + np.dot(self.klm_loc_cl.C, np.dot(self.klm_loc_cl.Qkal, self.klm_loc_cl.C.T)) + self.klm_loc_cl.Rkal
                K_klm[i] = self.klm_loc_cl.K_kal

            inp_term = [np.dot(G[l], np.dot(K_klm[num_OLupdates - l - 1], np.dot(I[num_OLupdates - l - 1], np.dot(K_klm[num_OLupdates - l - 1].T, G[l].T)))) for l in range(0, num_OLupdates - 1)]
            err_mean = 0
            err_var = np.trace(self.Pol_loc - self.Pcl_loc)
        # Check triggering condition
        err = err_mean + err_var
        if err >= self.delta:
            # Set trigger variable
            gamma = 1
            self.latestTrig = step + predHor
        else:
            gamma = 0
        self.gamma_arr = np.roll(self.gamma_arr, 1)
        self.gamma_arr[0] = gamma
        return self.gamma_arr[predHor - 1]


'Define the self trigger'


class selfTrig:

    def __init__(self, klm_pred, klm_cl, delta):
        # Kalman filter
        self.klm_pred = klm_pred
        self.klm_cl = klm_cl
        # Initialize next planned trigger with Mmax
        self.nextTrig = Mmax
        # Communication threshold
        self.delta = delta

    'Check if communication is necessary'

    def trigger(self, P_klm, K_loc):
        self.klm_cl.Pcl = copy.deepcopy(P_klm)
        self.klm_cl.Pol = copy.deepcopy(P_klm)
        self.klm_pred.Pol = copy.deepcopy(P_klm)
        if self.nextTrig < Mmax and self.nextTrig > 0:
            self.nextTrig -= 1
        else:
            trigger = Mmax
            G = [np.zeros_like(self.klm_cl.A) for _ in range(Mmax)]
            G[0] = np.dot(self.klm_cl.B, K_loc)
            I = [np.zeros_like(self.klm_cl.Rkal) for _ in range(Mmax)]
            K_klm = [np.zeros((2 * num_veh, num_veh)) for _ in range(Mmax)]
            for i in range(0, Mmax):
                Pol = self.klm_pred.OLupdate_var()
                Pcl = self.klm_cl.CLupdate_var()
                # Additional terms due to estimated input
                if i > 0:
                    G[i] = np.dot(self.klm_cl.A, G[i - 1]) + np.dot(self.klm_cl.B, np.dot(K_loc, np.linalg.matrix_power(self.klm_cl.A - np.dot(self.klm_cl.B, K_loc), i + 1)))
                I[i] = np.dot(self.klm_cl.C, np.dot(self.klm_cl.A, np.dot(Pcl, np.dot(self.klm_cl.A.T, self.klm_cl.C.T)))) + np.dot(self.klm_cl.C, np.dot(self.klm_cl.Qkal, self.klm_cl.C.T)) + self.klm_cl.Rkal
                K_klm[i] = self.klm_cl.K_kal
                inp_term = [np.dot(G[l], np.dot(K_klm[i - l - 1], np.dot(I[i - l - 1], np.dot(K_klm[i - l - 1].T, G[l].T)))) for l in range(0, i - 1)]
                err_var = Pol + sum(inp_term)

                # Check trigger condition
                err = np.trace(err_var - Pcl)
                if err >= self.delta:
                    trigger = i
                    break
            self.nextTrig = trigger
        if self.nextTrig == 0:
            gamma = 1
        else:
            gamma = 0

        return gamma


'Implementation of just one vehicle, implementing both the local and the global system'


class vehicle:

    def __init__(self, x_des, x_init, veh_id, delta):
        # global system
        self.A = np.zeros((2 * num_veh, 2 * num_veh))
        self.B = np.zeros((2 * num_veh, num_veh))
        self.C = np.zeros((num_veh, 2 * num_veh))
        self.D = np.zeros((num_veh, num_veh))
        for i in range(0, num_veh):
            self.A[2 * i, 2 * i + 1] = 1
            self.B[2 * i + 1, i] = 1
            self.C[i, 2 * i ] = 1
        self.sysc = ctrl.ss(self.A, self.B, self.C, self.D)
        self.sysd = ctrl.sample_system(self.sysc, Ts)
        [self.Ad, self.Bd, self.Cd, self.Dd] = ctrl.ssdata(self.sysd)
        # local system
        self.A_loc = np.matrix([[0, 1], [0, 0]])
        self.B_loc = np.matrix([[0], [1]])
        self.C_loc = np.matrix([[1, 0]])
        self.D_loc = np.matrix([[0]])
        self.sysc_loc = ctrl.ss(self.A_loc, self.B_loc, self.C_loc, self.D_loc)
        self.sysd_loc = ctrl.sample_system(self.sysc_loc, Ts)
        [self.Ad_loc, self.Bd_loc, self.Cd_loc, self.Dd_loc] = ctrl.ssdata(self.sysd_loc)
        # Store number of vehicle in the platoon
        self.veh_id = veh_id
        # Get desired state
        self.x_des = copy.deepcopy(x_des)
        self.x_des_pred = copy.deepcopy(x_des)
        # Initialize state
        self.state = copy.deepcopy(x_init)
        self.state_pred = copy.deepcopy(x_init)
        # Initialize local state
        self.state_loc = np.matrix([[self.state[2 * self.veh_id, 0]], [self.state[2 * self.veh_id + 1, 0]]])
        self.state_loc_pred = copy.deepcopy(self.state_loc)
        # Initialize input array
        self.u = np.asmatrix(np.zeros((num_veh, 1)))
        # Initialize Kalman filters
        self.klm_pred = kalman(self.Ad, self.Bd, self.Cd, self.x_des)
        self.klm_loc = kalman(self.Ad_loc, self.Bd_loc, self.Cd_loc, self.x_des)
        self.klm_loc_pred = kalman(self.Ad_loc, self.Bd_loc, self.Cd_loc, self.x_des_pred)
        # Get local controller
        self.K_loc = np.dot(K, self.klm_pred.K_conv)[veh_id, 2 * veh_id:2 * veh_id + 2]
        # Initialize triggers
        self.predTrigger = predTrig(kalman(self.Ad_loc, self.Bd_loc, self.Cd_loc, self.x_des), kalman(self.Ad_loc, self.Bd_loc, self.Cd_loc, self.x_des_pred), delta, veh_id)
        self.selfTrigger = selfTrig(kalman(self.Ad_loc, self.Bd_loc, self.Cd_loc, self.x_des), kalman(self.Ad_loc, self.Bd_loc, self.Cd_loc, self.x_des), delta)
        # Initialize communication decision
        self.gamma = 0

    'Calculate inputs'

    def calc_input(self, state, x_des):
        return -np.dot(K, np.dot(self.klm_pred.K_conv, state) - x_des)

    'Predict state of global system'

    def predict(self):
        u_pred = self.calc_input(self.state_pred, self.x_des_pred)
        _ = self.klm_loc_pred.OLupdate_var()
        [self.state_pred, _] = self.klm_pred.OLupdate(self.state_pred, u_pred)
        self.state = copy.deepcopy(self.state_pred)

    'Get Kalman filter estimate for own, local state and replace in global matrix'

    def estimate_state(self, y):
        [self.state_loc, _] = self.klm_loc.CLupdate(self.state_loc, y[self.veh_id], self.u[self.veh_id])
        self.state[2 * self.veh_id:2 * self.veh_id + 2] = copy.deepcopy(self.state_loc)
        self.u = self.calc_input(self.state, self.x_des)

    'Check trigger condition for self trigger'

    def gamma_self(self):
        self.gamma = self.selfTrigger.trigger(self.klm_loc.Pcl, self.K_loc)

    'Check trigger condition for predictive trigger'

    def gamma_pred(self, step):
        self.gamma = self.predTrigger.trigger(step, self.state, self.state_pred, self.klm_loc_pred.Pol, self.klm_loc.Pcl, self.K_loc)


# Get controller and controller gain
controller = controller(num_veh, Ts)
[K, S, eigv] = controller.dlqr()
