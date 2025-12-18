import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import pyrealsense2 as rs
import numpy as np

class CustomRealsenseNode(Node):
    def __init__(self):
        super().__init__('custom_realsense_node')
        
        # --- Configuration ---
        # Ensure this matches the link name in your existing URDF/TF tree
        self.frame_id = "camera_link"
        
        # --- Publishers ---
        self.color_pub = self.create_publisher(Image, '/camera/color/image_raw', 10)
        self.depth_aligned_pub = self.create_publisher(Image, '/camera/aligned_depth_to_color/image_raw', 10)

        # --- Realsense Setup ---
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Auto-negotiate streams (Fixes "Couldn't resolve requests" error)
        self.config.enable_stream(rs.stream.color)
        self.config.enable_stream(rs.stream.depth)
        
        # Align depth to color (so they share the same TF frame)
        self.align = rs.align(rs.stream.color)

        try:
            self.pipeline.start(self.config)
            self.get_logger().info("RealSense Camera started.")
        except Exception as e:
            self.get_logger().error(f"Failed to start camera: {e}")
            exit(1)

        self.bridge = CvBridge()
        # Timer set to approx 30 FPS (0.033s)
        self.timer = self.create_timer(1.0 / 30, self.timer_callback)

    def timer_callback(self):
        try:
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            now = self.get_clock().now().to_msg()

            # Publish Color
            color_msg = self.bridge.cv2_to_imgmsg(color_image, encoding="bgr8")
            color_msg.header.stamp = now
            color_msg.header.frame_id = self.frame_id
            self.color_pub.publish(color_msg)

            # Publish Depth
            depth_msg = self.bridge.cv2_to_imgmsg(depth_image, encoding="16UC1")
            depth_msg.header.stamp = now
            depth_msg.header.frame_id = self.frame_id
            self.depth_aligned_pub.publish(depth_msg)

        except RuntimeError:
            pass

    def stop(self):
        self.pipeline.stop()

def main(args=None):
    rclpy.init(args=args)
    node = CustomRealsenseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()