#!/usr/bin/env python3

import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from visualization_msgs.msg import Marker
from sensor_msgs.msg import JointState

from control_msgs.action import FollowJointTrajectory

from moveit_msgs.srv import GetMotionPlan, GetPositionIK
from moveit_msgs.msg import Constraints, JointConstraint

from planning.ik_pyroki import IKPlanner


UR_ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Filters 
TABLE_SAFETY_HEIGHT = 0.05
ZERO_POINT_EPS = 0.02
MIN_VALID_RADIUS = 0.10

# Planning
GOAL_TOL = 0.05
ALLOWED_PLANNING_TIME = 5.0
PLAN_CALL_TIMEOUT = 7.0   # must be > allowed_planning_time


def wrap_to_pi(q: np.ndarray) -> np.ndarray:
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def filter_joint_state_to_arm(js: JointState) -> JointState:
    name_to_idx = {n: i for i, n in enumerate(js.name)}
    out = JointState()
    out.header = js.header
    out.name = list(UR_ARM_JOINTS)
    out.position = []
    for n in UR_ARM_JOINTS:
        if n not in name_to_idx:
            raise ValueError(f"Missing arm joint '{n}' in /joint_states")
        out.position.append(float(js.position[name_to_idx[n]]))
    return out


def fmt_joints(names, pos) -> str:
    return ", ".join([f"{n}={float(v): .3f}" for n, v in zip(names, pos)])


class UR7e_Catch_Pyroki_MoveIt(Node):
    def __init__(self):
        super().__init__('catch_pyroki_moveit')

        self.marker_sub = self.create_subscription(
            Marker, '/impact_point_marker', self.marker_callback, 1
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 1
        )

        self.plan_client = self.create_client(GetMotionPlan, '/plan_kinematic_path')

        # Shadow only, for matching original behavior/logs (not actually used)
        self.fake_ik_client = self.create_client(GetPositionIK, '/compute_ik')

        self.exec_ac = ActionClient(
            self, FollowJointTrajectory,
            '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        )

        self.ik_planner = IKPlanner()

        self.current_joint_state = None
        self.is_busy = False  # True while planning or executing
        self.last_target = None  # for de-bounce

        # Match original waiting behavior
        self.get_logger().info("Waiting for /compute_ik service...")
        while not self.fake_ik_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /compute_ik service...")

        self.get_logger().info("Waiting for /plan_kinematic_path service...")
        while not self.plan_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /plan_kinematic_path service...")

        self.get_logger().info("Waiting for trajectory controller action server...")
        self.exec_ac.wait_for_server()

        self.get_logger().info("Catch node ready (Pyroki IK + MoveIt planning).")

    def joint_state_callback(self, msg: JointState):
        self.current_joint_state = msg

    def marker_callback(self, msg: Marker):
        if msg.action != Marker.ADD:
            return
        if self.is_busy:
            return
        if self.current_joint_state is None:
            self.get_logger().warn("Waiting for /joint_states...", throttle_duration_sec=2.0)
            return

        p = msg.pose.position
        r = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)

        # Minimal filtering to avoid garbage targets
        if r < ZERO_POINT_EPS:
            return
        if p.z < TABLE_SAFETY_HEIGHT:
            return
        if r < MIN_VALID_RADIUS:
            return

        #ignore repeated same target (prevents repeated planning spam)
        key = (round(p.x, 3), round(p.y, 3), round(p.z, 3))
        if self.last_target == key:
            return
        self.last_target = key

        self.get_logger().info(f"Target xyz: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f}), r={r:.3f}")

        try:
            start_arm_js = filter_joint_state_to_arm(self.current_joint_state)
        except Exception as e:
            self.get_logger().error(f"Cannot filter /joint_states: {e}")
            return

        goal_arm_js = self.ik_planner.compute_ik(start_arm_js, p.x, p.y, p.z)
        if goal_arm_js is None:
            self.get_logger().warn("Pyroki IK failed.")
            return

        # wrap goal angles to [-pi, pi]
        q_goal = np.array(goal_arm_js.position, dtype=float)
        if not np.isfinite(q_goal).all():
            self.get_logger().error("Goal contains NaN/Inf, skipping.")
            return
        goal_arm_js.position = wrap_to_pi(q_goal).tolist()

        self.get_logger().info(f"Start joints: {fmt_joints(start_arm_js.name, start_arm_js.position)}")
        self.get_logger().info(f"Goal joints : {fmt_joints(goal_arm_js.name, goal_arm_js.position)}")

        # (non-blocking wait + timeout)
        self.is_busy = True
        traj = self.plan_with_moveit_nonblocking(start_arm_js, goal_arm_js)
        if traj is None:
            self.is_busy = False
            return

        self.execute_trajectory(traj)

    def plan_with_moveit_nonblocking(self, start_arm_js: JointState, goal_arm_js: JointState):
        req = GetMotionPlan.Request()
        req.motion_plan_request.group_name = 'ur_manipulator'
        req.motion_plan_request.allowed_planning_time = float(ALLOWED_PLANNING_TIME)
        req.motion_plan_request.planner_id = "RRTConnectkConfigDefault"

        # Start state (arm joints only)
        req.motion_plan_request.start_state.joint_state = start_arm_js

        # Goal constraints (arm joints only)
        goal = Constraints()
        for name, pos in zip(goal_arm_js.name, goal_arm_js.position):
            goal.joint_constraints.append(
                JointConstraint(
                    joint_name=name,
                    position=float(pos),
                    tolerance_above=float(GOAL_TOL),
                    tolerance_below=float(GOAL_TOL),
                    weight=1.0
                )
            )
        req.motion_plan_request.goal_constraints.append(goal)

        self.get_logger().info("Requesting MoveIt plan...")

        future = self.plan_client.call_async(req)

        # wait with periodic logging + hard timeout
        t0 = self.get_clock().now()
        last_log_sec = -1

        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.2)

            dt = (self.get_clock().now() - t0).nanoseconds * 1e-9
            sec = int(dt)
            if sec != last_log_sec and sec >= 1:
                last_log_sec = sec
                self.get_logger().info(f"Still waiting for MoveIt plan... {dt:.1f}s")

            if dt > float(PLAN_CALL_TIMEOUT):
                self.get_logger().error(
                    f"MoveIt plan call timed out after {dt:.1f}s. "
                    "MoveIt/move_group may be stuck or overloaded. Skipping this target."
                )
                return None

        result = future.result()
        if result is None:
            self.get_logger().error("MoveIt planning service returned None.")
            return None

        code = result.motion_plan_response.error_code.val
        if code != 1:
            self.get_logger().warn(f"MoveIt failed to plan. Error code: {code}")
            return None

        traj = result.motion_plan_response.trajectory.joint_trajectory
        if not traj.points:
            self.get_logger().warn("MoveIt returned empty trajectory.")
            return None

        self.get_logger().info(f"MoveIt plan OK: points={len(traj.points)}")
        return traj

    def execute_trajectory(self, joint_traj):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_traj

        self.get_logger().info("Executing trajectory...")
        send_future = self.exec_ac.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_sent)

    def _on_goal_sent(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory rejected by controller.")
            self.is_busy = False
            return

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_done)

    def _on_done(self, future):
        self.get_logger().info("Motion complete.")
        self.is_busy = False


def main(args=None):
    rclpy.init(args=args)
    node = UR7e_Catch_Pyroki_MoveIt()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
