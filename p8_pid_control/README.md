# CarND-Controls-PID

This udacity project implements a PID controller for autonomous cars. 
Here is the [original project](https://github.com/udacity/CarND-PID-Control-Project) 
containing all details. 

## Build Instructions

1. Make a build directory: `mkdir build && cd build`
2. Compile: `cmake .. && make`
3. Run it: `./pid`. 

## PID Controller

Our PID controller implements the basic PID algorithm. See implementation
at src/PID.*. Below details the effect of each component of the algorithm.

- *P*roportional component controls the cross track error (CTE) proportionally.
It tries to steer the vechile toward the expected center line. The bigger it is,
the stronger the control signal will be. If it is used alone, the vehicle will
oscilate around the center line. 

- *I*ntegral component controls the accumulating error. It reduces the steady 
state error, i.e., it makes the controler drive the vehicle along the expected
line even if the vehicle biases towards left/right. 

- *D*erivative component controls the rate of change of error. It counteracts
the proportional componeny to avoid overshooting the center line, thus it weakens
the oscilation.

We selects the hyperparameters manually. We first tried the parameters used
in the lesson (tau_p = 0.2, tau_i = 0.004, tau_d = 3.0), but the car drives pulls
too hard at turns, so we reduce the tau_p to 0.1. Then we manually tune tau_d
to 2.0 to reduce the obvious oscilation. Our final parameter choice (0.1, 0.004, 2.0)
enable the car drive smoothly a lap of the track. 

## Demo Video

![Demo](demo/demo.gif)
