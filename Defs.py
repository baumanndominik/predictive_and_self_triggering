'''
Created on Dec 19, 2017

@author: Dominik Baumann
MPI-IS, ICS
dbaumann(at)tuebingen.mpg.de

Copyright 2017 Max Planck Society. All rights reserved.
'''

import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as mpl
import control as ctrl
from scipy.linalg import inv
import copy

# Number of vehicles
num_veh = 10
# Discrete time step
Ts = 0.1
# Number of iterations
num_it = 250
# Desired velocity
des_vel = 80 / 3.6
# Desired distance
dist = 10
# Prediction horizon
predHor = 5
# Maximum number of iterations for self trigger
Mmax = 300
# Communication cost
delta = 0.7
# Decide which trigger to use (1: Predictive, 2: Self Trigger), trigger_comparison always uses both triggers
trigger = 2
# Maximum noise
w_max = 0.1
v_max = 0.1
# Packet drop probability
pdr = 0.1
# Initial values for state and desired state
x_init = np.zeros((2 * num_veh, 1))
x_des_init = np.zeros((2 * num_veh - 1, 1))
x_init[0, 0] = 0
x_init[1, 0] = des_vel
x_des_init[0, 0] = des_vel
x_des_init[1, 0] = dist
for i in range(1, num_veh):
    x_init[2 * i + 1, 0] = des_vel
    x_init[2 * i, 0] = x_init[2 * i - 2, 0] - dist
    if i == num_veh - 1:
        x_des_init[-1, 0] = des_vel
    else:
        x_des_init[2 * i, 0] = des_vel
        x_des_init[2 * i + 1, 0] = dist

# LQR matrices
Q = np.eye(2 * (num_veh - 1) + 1)
R = 1000 * np.eye(num_veh)
