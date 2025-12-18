#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class ToolToCameraPublisher(Node):
    def __init__(self):
        super().__init__('tool_to_camera_tf_publisher')
        self.br = StaticTransformBroadcaster(self)
        
        # translation (in meters)
        X_t = 0.0
        Y_t = 0.05
        Z_t = 0.0

        # R matrix (3x3)
        r11, r12, r13 = 0.0, 1.0, 0.0  
        r21, r22, r23 = 1.0, 0.0, 0.0   
        r31, r32, r33 = 0.0, 0.0, -1.0   

        G = np.array([
            [r11, r12, r13, X_t], 
            [r21, r22, r23, Y_t], 
            [r31, r32, r33, Z_t], 
            [0.0, 0.0, 0.0, 1.0]
        ])

        R_mat = G[:3, :3]
        t = G[:3, 3]

        # convert rotation matrix to quaternion
        rot = R.from_matrix(R_mat)
        q = rot.as_quat()  # [x, y, z, w]

        # create TransformStamped
        self.transform = TransformStamped()
        
        # frame ids
        self.transform.header.frame_id = 'tool0'      # parent frame
        self.transform.child_frame_id = 'camera_link' # child frame

        # fill translation
        self.transform.transform.translation.x = float(t[0])
        self.transform.transform.translation.y = float(t[1])
        self.transform.transform.translation.z = float(t[2])

        # fill rotation
        self.transform.transform.rotation.x = float(q[0])
        self.transform.transform.rotation.y = float(q[1])
        self.transform.transform.rotation.z = float(q[2])
        self.transform.transform.rotation.w = float(q[3])

        self.timer = self.create_timer(0.05, self.broadcast_tf)
        self.get_logger().info(f"Publishing Static TF: {self.transform.header.frame_id} -> {self.transform.child_frame_id}")

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

def main():
    rclpy.init()
    node = ToolToCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()