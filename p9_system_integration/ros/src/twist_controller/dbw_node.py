#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''

class DBWNode(object):
    def __init__(self):

        rospy.init_node('dbw_node')

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        # Create a controller
        self.controller = Controller(vehicle_mass=vehicle_mass,
                                     fuel_capacity=fuel_capacity,
                                     brake_deadband=brake_deadband,
                                     decel_limit=decel_limit,
                                     accel_limit=accel_limit,
                                     wheel_radius=wheel_radius,
                                     wheel_base=wheel_base,
                                     steer_ratio=steer_ratio,
                                     max_lat_accel=max_lat_accel,
                                     max_steer_angle=max_steer_angle)

        # Create publishers
        self.steer_publisher = rospy.Publisher('/vehicle/steering_cmd',
                SteeringCmd, queue_size=1)
        self.throttle_publisher = rospy.Publisher('/vehicle/throttle_cmd',
                ThrottleCmd, queue_size=1)
        self.brake_publisher = rospy.Publisher('/vehicle/brake_cmd',
                BrakeCmd, queue_size=1)

        # Create subscribers
        rospy.Subscriber('/current_velocity', TwistStamped, self.set_current_velocity)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.set_velocity)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.set_dbw_enabled)

        # Initialize member variables
        self.current_velocity = None
        self.linear_velocity = None
        self.angular_velocity = None
        self.dbw_enabled = None
        self.throttle = 0
        self.steering = 0
        self.brake = 0

        self.exec_at_hz(50)

    def exec_at_hz(self, hz):
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            # Publish the predicted throttle, brake, and steering using
            # twist_controller only if dbw is enabled
            if not None in (self.current_velocity,
                    self.linear_velocity, self.angular_velocity):
                self.throttle, self.brake, self.steering = (
                        self.controller.control(self.current_velocity,
                                                self.dbw_enabled,
                                                self.linear_velocity,
                                                self.angular_velocity))
            if self.dbw_enabled:
                self.publish(self.throttle, self.brake, self.steering)

            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_publisher.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_publisher.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_publisher.publish(bcmd)

    def set_current_velocity(self, msg):
        self.current_velocity = msg.twist.linear.x

    def set_dbw_enabled(self, msg):
        self.dbw_enabled = msg

    def set_velocity(self, msg):
        self.linear_velocity = msg.twist.linear.x
        self.angular_velocity = msg.twist.angular.z


if __name__ == '__main__':
    DBWNode()
