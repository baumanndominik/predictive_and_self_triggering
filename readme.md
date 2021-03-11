# Predictive and Self Triggering

This repository is the official implementation of the [paper](https://arxiv.org/abs/1901.07531) "Resource-aware IoT control: Saving comunication through predictive triggering" by Sebastian Trimpe and Dominik Baumann, published in the IEEE Internet of Things Journal.

# Requirements

The code was developed using Python 3.6 and also tested with Python 3.4. The following libraries are required:

* numpy with numpy.matlib
* scipy
* matplotlib
* control
* copy

# Simulation

The code implements the simulation example discussed in Sec. IX of the paper. It consists of a bunch of vehicles platooning with a desired inter-vehicle distance and velocity. All vehicles can only measure their own absolute positions and, thus, need to communicate with the other vehicles to maintain the desired inter-vehicle distance. The decision when to communicate is taken by either the predictive or the self trigger.

# Execution

To execute the code, run the command

```
python single_experiment.py
```
to start a single simulation, 

```
python run_braking_experiment
```
for a single simulation in which the first vehicle in the platoon starts to brake after 10s (the seed is set such that the results from the paper can be recreated), and

```
python trigger_comparison.py
```
to start a Monte Carlo simulation comparing predictive and self triggering for different communication thresholds.
Source code of the simulation example for the paper "Resource-aware IoT Control: Saving
Communication through Predictive Triggering", published in the IEEE Internet of Things 
Journal (https://ieeexplore.ieee.org/document/8624412). The code simulates a bunch of 
vehicles platooning with a desired inter-vehicle distance and velocity. All vehicles are 
able to communicate their state to all other. The decision, when to communicate, is taken 
by either the predictive or the self trigger.
