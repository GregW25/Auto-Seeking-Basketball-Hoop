#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory
from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from planning.ik import IKPlanner

TABLE_SAFETY_HEIGHT = 0 # Z height filter

class UR7e_HoopCatch(Node):
    def __init__(self):
        super().__init__('hoop_catch_node')

        # subscribe to the topic to get the red sphere position
        self.impact_sub = self.create_subscription(
            Marker, 
            '/impact_point_marker', 
            self.impact_callback, 
            1
        )
        
        self.joint_state_sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            1
        )

        self.exec_ac = ActionClient(
            self, 
            FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.is_moving = False
        self.current_joint_state = None
        self.ik_planner = IKPlanner()

        # Non-blocking wait for server 
        self.get_logger().info("Waiting for trajectory controller...")
        if not self.exec_ac.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Trajectory controller not available! Robot will not move.")
        else:
            self.get_logger().info(f"Ready to catch! (Safety Floor: Z > {TABLE_SAFETY_HEIGHT}m)")

    def joint_state_callback(self, msg: JointState):
        self.current_joint_state = msg

    def impact_callback(self, msg: Marker):
        if msg.action != Marker.ADD:
            return

        if self.is_moving:
            return  
        
        if self.current_joint_state is None:
            self.get_logger().warn("Waiting for joint states...", throttle_duration_sec=2.0)
            return

        # extract position
        target_point = msg.pose.position

        # offset here
        target_point.y -= 0.2  # 1cm in the negative Y direction
        
        # table safety check
        if target_point.z < TABLE_SAFETY_HEIGHT:
            self.get_logger().warn(f"Ignoring target too low: {target_point.z:.3f}m < {TABLE_SAFETY_HEIGHT}m")
            return

        self.get_logger().info(f"Ball detected at: ({target_point.x:.2f}, {target_point.y:.2f}, {target_point.z:.2f})")

        # Compute IK
        target_joints = self.ik_planner.compute_ik(
            self.current_joint_state, 
            target_point.x, 
            target_point.y, 
            target_point.z
        )

        if target_joints is None:
            self.get_logger().warn("No IK solution found for impact point.")
            return

        # plan path
        plan = self.ik_planner.plan_to_joints(target_joints)

        if plan and plan.joint_trajectory:
            self.execute_trajectory(plan.joint_trajectory)
        else:
            self.get_logger().warn("IK found, but path planning failed.")

    def execute_trajectory(self, joint_traj):
        self.is_moving = True
        
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj
        
        self.get_logger().info("Moving to catch...")
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_accepted)

    def _on_goal_accepted(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Move rejected by controller.")
            self.is_moving = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_move_complete)

    def _on_move_complete(self, future):
        self.get_logger().info("Catch attempt finished. Resetting.")
        self.is_moving = False

def main(args=None):
    rclpy.init(args=args)
    node = UR7e_HoopCatch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()