#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class ConstantTransformPublisher(Node):
    def __init__(self):
        super().__init__('constant_tf_publisher')
        self.br = StaticTransformBroadcaster(self)

        #baymax is #6
        self.declare_parameter('ar_marker', 'ar_marker_6')
        marker = self.get_parameter('ar_marker').get_parameter_value().string_value

        # Homogeneous transform G_ar->base_link
        G = np.array([
            [-1, 0, 0, 0.0],
            [ 0, 0, 1, 0.16],
            [ 0, 1, 0, -0.13],
            [ 0, 0, 0, 1.0]
        ])

        # Create TransformStamped
        self.transform = TransformStamped()
        # ---------------------------
        # Fill out TransformStamped message
        # --------------------------
        # Extract rotation (3x3) and translation (3x1)

        # Extract rotation (upper-left 3x3) and translation (first 3 of last column)
        R_mat = G[:3, :3]
        t = G[:3, 3]

        # Convert rotation matrix to quaternion
        rot = R.from_matrix(R_mat)
        q = rot.as_quat()  # [x, y, z, w]

        # Create TransformStamped
        self.transform = TransformStamped()
        self.transform.header.frame_id = 'ar_marker_6'   # parent frame
        self.transform.child_frame_id = 'base_link'      # child frame

        # Fill translation
        self.transform.transform.translation.x = float(t[0])
        self.transform.transform.translation.y = float(t[1])
        self.transform.transform.translation.z = float(t[2])

        # Fill rotation
        self.transform.transform.rotation.x = float(q[0])
        self.transform.transform.rotation.y = float(q[1])
        self.transform.transform.rotation.z = float(q[2])
        self.transform.transform.rotation.w = float(q[3])

        self.timer = self.create_timer(0.05, self.broadcast_tf)

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

def main():
    rclpy.init()
    node = ConstantTransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
