#!/usr/bin/env python

import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Int32
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''


# Number of waypoints we will publish. You can change this number
LOOKAHEAD_WPS = 50
MAX_DECEL = 0.5

class WaypointUpdater(object):

    def __init__(self):

        # Initialize the node
        rospy.init_node('waypoint_updater')

        # Create subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.set_pose)
        rospy.Subscriber('/base_waypoints', Lane, self.set_waypoints)
        rospy.Subscriber('/traffic_waypoint', Int32, self.set_traffic_info)

        # Create a publisher
        self.final_waypoints_publisher = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        # Initialize member variables
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.stopline_wp_idx = -1

        self.exec_at_hz(5)

    def calculate_distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def decelerate_waypoints(self, waypoints, closest_idx):
        decel_waypoints = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose

            stop_idx = max(self.stopline_wp_idx - closest_idx - 2, 0)
            dist = self.calculate_distance(waypoints, i, stop_idx)
            velocity = math.sqrt(2 * MAX_DECEL * dist)
            if velocity < 1:
                velocity = 0

            p.twist.twist.linear.x = min(velocity, wp.twist.twist.linear.x)
            decel_waypoints.append(p)

        return decel_waypoints

    def exec_at_hz(self, hz):
        """
        Execute code at a given frequency, hz
        """
        rate = rospy.Rate(hz)
        while not rospy.is_shutdown():
            if self.pose and self.waypoint_tree:
                idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(idx)
            rate.sleep()

    def get_closest_waypoint_idx(self):
        """
        Get the index of the closest waypoint
        """
        # Get the closest waypoint
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        idx = self.waypoint_tree.query([x, y], 1)[1]

        # Get the next waypoint if the above found is behind the vehicle
        closest = np.array(self.waypoints_2d[idx])
        closest_prev = np.array(self.waypoints_2d[idx - 1])
        current = np.array([x, y])
        if np.dot(closest - closest_prev, current - closest) > 0:
            return (idx + 1) % len(self.waypoints_2d)
        return idx

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def publish_waypoints(self, idx):
        # Publish final waypoints
        lane = Lane()
        limit = idx + LOOKAHEAD_WPS
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[idx: limit]

        if self.stopline_wp_idx != -1 and self.stopline_wp_idx < limit:
            lane.waypoints = self.decelerate_waypoints(lane.waypoints, idx)

        self.final_waypoints_publisher.publish(lane)

    def set_pose(self, msg):
        self.pose = msg

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def set_waypoints(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        assert self.waypoint_tree != None

    def set_traffic_info(self, msg):
        self.stopline_wp_idx = msg.data


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
