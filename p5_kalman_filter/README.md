# Extended Kalman Filter Project
Self-Driving Car Engineer Nanodegree Program

In this project we will utilize an Extended kalman filter to estimate 
the state of a moving object of interest with noisy lidar and radar 
measurements generated by a provided simulator. It is based on the [seed 
project](https://github.com/udacity/CarND-Extended-Kalman-Filter-Project) 
by Udacity.  

## Code Structure

- main.cpp, communicating with the simulator via uWebSocketIO, which include 
communication in two directions: (1) main to simulator for sending the predicted
state and (2) simulator to main for sending the measurement observed by the 
sensors.
- kalman_filter.*, implementing the extended kalman filter based on both measurement
from lidar and radar.  
- FusionEKF.*, setting parameters for the kalman filter and dictating how kalman
filter is used based on the given sensor type: lidar v.s. radar. 
- tools.*, utilities for computing RMSE (root mean square error) and Jacobian matrix


## Compilation & Running

1. mkdir build
2. cd build
3. cmake ..
4. make
5. ./ExtendedKF
6. open the simulator

## Simulation

![](result/dataset1.gif)
![](result/dataset2.gif)
