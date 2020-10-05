#!/usr/bin/env python

import rospy
import math
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
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
LOOKAHEAD_WPS = 200


class WaypointUpdater(object):

    def __init__(self):

        # Initialize member variables
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None

        # Initialize the node
        rospy.init_node('waypoint_updater')

        # Create subscribers
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below

        # Create a publisher
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.exec_at_hz(50)

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

    def exec_at_hz(self, hz):
        rate = rospy.Rate(hz)
        """
        Execute code at a given frequency, hz
        """
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(idx)

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

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def publish_waypoints(self, idx):
        # Publish final waypoints
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[idx: idx + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        # Update the pose
        self.pose = msg

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[wp.pose.pose.position.x, wp.pose.pose.position.y] for wp in waypoints.waypoints]
            self.waypoint_tree = KDTree(self.waypoints_2d)
        assert self.waypoint_tree != None

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
