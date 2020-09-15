# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

## Goal

In this project your goal is to safely navigate around a virtual highway with other traffic that is driving +-10 MPH of the 50 MPH speed limit. You will be provided the car's localization and sensor fusion data, there is also a sparse map list of waypoints around the highway. The car should try to go as close as possible to the 50 MPH speed limit, which means passing slower traffic when possible, note that other cars will try to change lanes too. The car should avoid hitting other cars at all cost as well as driving inside of the marked road lanes at all times, unless going from one lane to another. The car should be able to make one complete loop around the 6946m highway. Since the car is trying to go 50 MPH, it should take a little over 5 minutes to complete 1 loop. Also the car should not experience total acceleration over 10 m/s^2 and jerk that is greater than 10 m/s^3.

## Details of the Project

See details of the project [here](https://github.com/udacity/CarND-Path-Planning-Project)

## Project Rubric

The implementation passes the rubric points below:

- [x] Code compiles correctly
- [x] The car is able to drive at least 4.32 miles without incident.
- [x] The car drives according to the speed limit.
- [x] Max Acceleration and Jerk are not Exceeded.
- [x] Car does not have collisions.
- [x] The car stays in its lane, except for the time between changing lanes.
- [x] The car is able to change lanes when feasible. 

## Demo

![!Demo](demo/demo.gif)

## Reflection

We implement a basic path planning algorithm. It consists of 3 parts: 
prediction, decision, and trajectory generation. 

In prediction step, we check the presence of cars in the 3 feasible lanes for our
vehicle. Specifically, we check 

- In the lane where our vehicle runs if there is a another vehicle blocking us, 
i.e., the vehicle is in front of us and its distance from us is less than 20 meters
- If there is a vehicle on the right of us blocking us from shifting to the 
right lane, i.e., its distance with us is less than 20 meters
- If there is a vehicle on the left of us blocking us from shifting to the left,
i.e., its distance with us is less than 20 meters

In the decision step, we need to decide 

- If we need to change lane based on the above observation.
- If we need to change speed, i.e., we hope to be as fast as possible (close to 
50 mph) but at the same time be not too close to other vehicles.

In the trajectory generation step, we generate smooth trajectories using the 
decisions above as well as the past trajectory information. 

In the demo video, our vehicle drives well. But there situations this planning model
does not work well. For example, our vehicle is on the left most lane with one vehicle
blocking its way in the front, on the right another vehicle runs with it at the 
same speed, and no vehicles run on the right most lane. Though our vehicle could 
have slowed down and shift to the right most lane to quickly get away from the traffic, 
it unfortunately does not. To deal with this issue, we must also consider the occupancy
of each lane and the slow-down strategy. For another, our basic model does not consider
the velocity of other vehicles, so it might not work well if there is crazy drivers
who accelerate/decelerate fiercely. A better model would consider velocity in the planning. 
In summary, much improvement could be made by a well-designed finite state machine. 
