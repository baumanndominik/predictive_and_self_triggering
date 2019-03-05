========================================================================================
Source code of the simulation example for the paper "Resource-aware IoT Control: Saving
Communication through Predictive Triggering", published in the IEEE Internet of Things 
Journal (https://ieeexplore.ieee.org/document/8624412). The code simulates a bunch of 
vehicles platooning with a desired inter-vehicle distance and velocity. All vehicles are 
able to communicate their state to all other. The decision, when to communicate, is taken 
by either the predictive or the self trigger.

The simulations were created using Python 3.6 and also tested with Python 3.4. To run 
the simulations, it is required to have the following Python libraries installed:
- numpy with numpy.matlib
- scipy.linalg
- matplotlib.pyplot
- control
- copy.

All parameters that may be changed to obtain different kinds of simulations are stored in
Defs.py. The parameters are for example the number of vehicles in the platoon, noise
parameters, the trigger, the desired velocity, and inter-vehicle distance, etc. Three 
different kinds of simulations are provided: 
- single_experiment.py starts a single simulation.
- run_braking_experiment.py also starts a single simulation, but after 10s the first
vehicle in the platoon starts to break. The noise is here generated using a seed such
that the results presented in the paper can be recreated.
- trigger_comparison.py starts a Monte Carlo simulation comparing both triggers for
different communication thresholds.
The file VehiclePlatoon.py implements the classes necessary to run the simulation.

The simulations can be executed from the command line.

Dominik Baumann
MPI-IS, ICS
dbaumann(at)tuebingen.mpg.de

Copyright 2017 Max Planck Society. All rights reserved.
=========================================================================================

