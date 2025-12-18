#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster
from scipy.spatial.transform import Rotation as R
import numpy as np

class Cam2TransformPublisher(Node):
    def __init__(self):
        super().__init__('cam2_tf_publisher')
        self.br = StaticTransformBroadcaster(self)
        # Parent Frame: camera1_link
        # Child Frame:  camera2_link
   
        X_t = 0.0
        Y_t = 0.08  # 4 inches in -Y direction
        Z_t = 0.0

        G = np.array([
            [ 0.154509,  0.951057, -0.267617, X_t], 
            [-0.512236, -0.154508, -0.844832, Y_t], 
            [-0.844832,  0.267617,  0.463292, Z_t], 
            [ 0.0,       0.0,       0.0,      1.0]
        ])

        R_mat = G[:3, :3]
        t = G[:3, 3]

        # Convert rotation matrix to quaternion
        rot = R.from_matrix(R_mat)
        q = rot.as_quat()  # [x, y, z, w]

        # Create TransformStamped
        self.transform = TransformStamped()
        self.transform.header.frame_id = 'camera1'   # parent frame
        self.transform.child_frame_id = 'camera_link'    # child frame

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

    def broadcast_tf(self):
        self.transform.header.stamp = self.get_clock().now().to_msg()
        self.br.sendTransform(self.transform)

def main():
    rclpy.init()
    node = Cam2TransformPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()