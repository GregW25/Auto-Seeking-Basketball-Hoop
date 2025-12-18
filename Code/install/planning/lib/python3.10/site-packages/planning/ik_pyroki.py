#!/usr/bin/env python3
"""
IK helper using Pyroki ONLY.
12.16 02：29
This module:
- Computes IK using Pyroki
- Does NOT plan trajectories
- Is designed to be paired with MoveIt motion execution

Public API:
- compute_ik(current_joint_state, x, y, z, qx=0, qy=1, qz=0, qw=0)
"""

import os
import sys
from typing import Optional, List, Dict

import numpy as np
import pyroki as pk
# Note: This relative import works when running as a package module (ros2 run)
from . import pyroki_snippets as pks
from yourdfpy import URDF

# ROS imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


# Update this path if necessary to match your actual file system
URDF_PATH = "/home/cc/ee106a/fa25/class/ee106a-aah/ros_workspaces/lab7/src/planning/planning/ur7e.urdf"


class IKPlanner:
    def __init__(self, urdf_path: str = URDF_PATH) -> None:
        print("[IKPlanner(Pyroki)] Initializing Pyroki IK solver...")

        if not os.path.isfile(urdf_path):
            raise FileNotFoundError(f"URDF not found: {urdf_path}")

        urdf = URDF.load(
            urdf_path,
            load_meshes=False,
            build_scene_graph=True,
        )

        # Patch base_link if needed
        if getattr(urdf, "base_link", None) is None:
            children = {j.child for j in urdf.joint_map.values() if j.child}
            roots = list(set(urdf.link_map.keys()) - children)
            if len(roots) != 1:
                raise RuntimeError("Cannot determine unique base_link")
            urdf.base_link = roots[0]

        self.robot: pk.Robot = pk.Robot.from_urdf(urdf)
        # Verify this link name matches your URDF
        self.target_link_name = "tool0"

        self.actuated_names: List[str] = list(self.robot.joints.actuated_names)
        self.num_actuated = self.robot.joints.num_actuated_joints

        print("[IKPlanner(Pyroki)] Ready. Actuated joints:")
        for j in self.actuated_names:
            print(f"  - {j}")

    # --------------------------------------------------------

    def _jointstate_to_cfg(self, joint_state: JointState) -> np.ndarray:
        name_to_idx = {n: i for i, n in enumerate(joint_state.name)}
        return np.array(
            [joint_state.position[name_to_idx[n]] for n in self.actuated_names],
            dtype=float,
        )

    def _cfg_to_jointstate_like(
        self, template_js: JointState, q_sol: np.ndarray
    ) -> JointState:

        name_to_val = dict(zip(self.actuated_names, q_sol.tolist()))

        js = JointState()
        js.header = template_js.header
        js.name = list(template_js.name)
        js.position = [name_to_val[n] for n in js.name]

        return js

    # --------------------------------------------------------

    def compute_ik(
        self,
        current_joint_state: JointState,
        x: float,
        y: float,
        z: float,
        qx: float = 0.0,
        qy: float = 1.0,
        qz: float = 0.0,
        qw: float = 0.0,
    ) -> Optional[JointState]:
        """
        Compute IK using Pyroki.
        Returns a JointState compatible with MoveIt.
        """

        target_position = np.array([x, y, z], dtype=float)
        target_wxyz = np.array([qw, qx, qy, qz], dtype=float)

        print(
            f"[IKPlanner(Pyroki)] Solving IK for "
            f"pos=({x:.3f}, {y:.3f}, {z:.3f})"
        )

        q_sol = pks.solve_ik(
            robot=self.robot,
            target_link_name=self.target_link_name,
            target_position=target_position,
            target_wxyz=target_wxyz,
        )

        if q_sol is None:
            print("[IKPlanner(Pyroki)] IK failed.")
            return None

        q_sol = np.asarray(q_sol, dtype=float)

        print("[IKPlanner(Pyroki)] IK success.")
        return self._cfg_to_jointstate_like(current_joint_state, q_sol)


def main(args=None):
    """
    Main entry point for the ROS node.
    Initializes rclpy, creates the IKPlanner to verify setup, and keeps the node alive.
    """
    rclpy.init(args=args)
    node = rclpy.create_node("ik_pyroki_node")
    
    try:
        # Initialize the planner class
        planner = IKPlanner()
        node.get_logger().info("IKPlanner initialized successfully and ready.")
        
        # Keep the process alive so it doesn't exit immediately
        rclpy.spin(node)
        
    except FileNotFoundError as e:
        node.get_logger().error(f"Configuration Error: {e}")
    except Exception as e:
        node.get_logger().error(f"Unexpected Error: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()