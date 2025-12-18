#!/usr/bin/env python3

import math
import rclpy
from rclpy.node import Node
import tf2_ros
import numpy as np
import transforms3d.euler as euler
from geometry_msgs.msg import Twist, PointStamped
from plannedcntrl.trajectory import plan_curved_trajectory
import time


class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller')

        # Publisher and TF setup
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Controller gains
        self.Kp_lin = 0.8
        self.Kp_ang = 1.5

        # Subscriber
        self.create_subscription(PointStamped, '/goal_point', self.planning_callback, 10)

        self.get_logger().info('TurtleBot controller node initialized.')

    # ------------------------------------------------------------------
    def controller(self, waypoint):
        goal_x, goal_y, goal_yaw = waypoint
        while rclpy.ok():
<<<<<<< HEAD
            try:
                trans = self.tf_buffer.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                self.get_logger().warn('TF lookup failed, retrying...')
                rclpy.spin_once(self, timeout_sec=0.1)
                continue
=======
            rclpy.spin_once(self, timeout_sec=0.1)
            # TODO: Transform the waypoint from the odom/world frame into the robot's base_link frame 
            # before computing errors — you'll need this so x_err and yaw_err are in the robot's coordinate system.
>>>>>>> 44ae72a1ab33be1cf2ed01c83d4cde1291702437

            # Extract current pose
            x_robot = trans.transform.translation.x
            y_robot = trans.transform.translation.y
            q = trans.transform.rotation
            roll, pitch, yaw = euler.quat2euler([q.w, q.x, q.y, q.z])

            # Compute errors
            dx = goal_x - x_robot
            dy = goal_y - y_robot
            distance = math.sqrt(dx**2 + dy**2)
            desired_yaw = math.atan2(dy, dx)
            yaw_err = math.atan2(math.sin(desired_yaw - yaw), math.cos(desired_yaw - yaw))

            # Stop condition
            if distance < 0.05:
                self.stop_robot()
                self.get_logger().info("Waypoint reached, moving to next.")
                return

            # Compute velocity commands
            cmd = Twist()
            cmd.linear.x = self.Kp_lin * distance
            cmd.angular.z = self.Kp_ang * yaw_err

            # Clip velocities
            cmd.linear.x = np.clip(cmd.linear.x, -0.2, 0.2)
            cmd.angular.z = np.clip(cmd.angular.z, -1.5, 1.5)

            # Publish command
            self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)

    # ------------------------------------------------------------------
    def stop_robot(self):
        self.pub.publish(Twist())

    # ------------------------------------------------------------------
    def planning_callback(self, msg: PointStamped):
        trajectory = plan_curved_trajectory((msg.point.x, msg.point.y))
        for waypoint in trajectory:
            self.controller(waypoint)
        self.stop_robot()
        self.get_logger().info("All waypoints reached.")

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
