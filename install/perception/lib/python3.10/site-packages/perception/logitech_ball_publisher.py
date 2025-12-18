import rclpy
import cv2
import os
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
import math
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy # <-- MUST BE PRESENT

# --- CONSTANTS FOR BALL DETECTION (Monocular Depth) ---
SPORTS_BALL_ID = 32
# This is the real-world area of the sports ball, you must measure/estimate this.
# Example: 0.011 m^2 for a small ball. This value is critical for monocular depth.
BALL_AREA = 0.011 
# The frame ID for the output point, often 'base_link' for TurtleBot/robot base.
OUTPUT_FRAME_ID = 'camera_depth_optical_frame' 
# --- CONSTANTS END ---

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()

        # Load YOLO model
        package_share_dir = get_package_share_directory('perception')
        # Ensure 'yolo11s-seg.pt' is the correct path for your ball model
        model_path = os.path.join(package_share_dir, 'utilities', 'yolo11s-seg.pt')
        self.model = YOLO(model_path)
        self.get_logger().info(f"YOLO Model classes: {self.model.names}")

        self.SPORTS_BALL_ID = SPORTS_BALL_ID
        self.BALL_AREA = BALL_AREA

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE, # CRITICAL: Requests the last message sent
            depth=1
        )
        

        # 2. Subscribe to the Image topic with the correct name (default QoS is fine)
        self.image_sub = self.create_subscription(
            Image, 
            '/camera1/image_raw',  # <-- CONFIRMED TOPIC
            self.image_callback, 
            qos_profile
        )

        # 3. Subscribe to Camera Info with the correct name AND custom QoS
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            '/camera1/camera_info', # <-- CONFIRMED TOPIC
            self.camera_info_callback, 
            qos_profile # <-- MUST USE THE CUSTOM PROFILE!
        )

        # Publishers
        self.ball_position_pub = self.create_publisher(PointStamped, '/goal_point', 1)
        self.debug_image_pub = self.create_publisher(Image, '/detection_image_out', 1)
        
        # Camera Intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.get_logger().info('Image Subscriber Node initialized for Monocular Camera')


    def image_callback(self, msg: Image):
        # Only proceed if camera intrinsics have been loaded
        if self.fx is None:
            self.get_logger().warn('Waiting for camera intrinsics...')
            return

        # 1. Convert ROS Image message to OpenCV format
        try:
            cv_image_color = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return
        
        # 2. Run YOLO detection - Filter by class ID
        # CRITICAL CHANGE 3: Run YOLO on the color image
        results = self.model(cv_image_color, classes=[self.SPORTS_BALL_ID], verbose=False)
        
        ball_detected_in_frame = False

        # --- PROCESS DETECTIONS ---
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()

                for i, mask in enumerate(masks):
                    
                    # Compute pixel count and centroid
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    pixel_count = np.sum(mask_binary) # Total pixels in the mask

                    M = cv2.moments(mask_binary)

                    if M['m00'] > 0 and pixel_count > 0:
                        u = int(M['m10'] / M['m00']) # Centroid x (column)
                        v = int(M['m01'] / M['m00']) # Centroid y (row)
                        ball_detected_in_frame = True
                    else:
                        continue 

                    # ---------------------------------------------
                    # CRITICAL CHANGE 4: MONOCULAR DEPTH CALCULATION
                    # Formula: Z = sqrt((fx * fy * Real_Area) / Pixel_Area)
                    # ---------------------------------------------
                    # Note: Using fx * fy * Real_Area / Pixel_Area is an approximation 
                    # that assumes the object is frontal-parallel to the camera.
                    
                    if pixel_count > 0: 
                        # Z is depth in meters
                        # We use the focal length product (fx*fy) to relate pixel area to world area
                        Z = math.sqrt((self.fx * self.fy * self.BALL_AREA) / pixel_count) 
                    else:
                        Z = 0.0
                    
                    if Z == 0.0:
                        self.get_logger().warn(f"Invalid depth reading (0) for ball {i+1}. Skipping.")
                        continue
                    
                    self.get_logger().info(f'Ball {i+1}: Monocular Depth={Z:.3f}m')
                         
                    # 4. Find X, Y, Z in the Camera Frame (using u, v and Z)
                    # Formula: X = Z * (u - cx) / fx, Y = Z * (v - cy) / fy
                    X = (((u - self.cx) * Z) / self.fx)
                    Y = (((v - self.cy) * Z) / self.fy)
    
                    # 5. Apply Transformation (using the matrix from your old cone code)
                    # This transforms the point from the Camera Frame to the Robot's Base Frame
                    G = np.array([[0, 0, 1, 0.115],
                                  [-1, 0, 0, 0],
                                  [0, -1, 0, 0],
                                  [0, 0, 0, 1]])
                    
                    goal_point = (G @ np.array([X, Y, Z, 1]).reshape(4, 1)).flatten()

                    # 6. Publish the 3D result
                    point_cam = PointStamped()
                    point_cam.header.stamp = msg.header.stamp
                    point_cam.header.frame_id = OUTPUT_FRAME_ID # e.g., 'base_link'
                    point_cam.point.x = goal_point[0]
                    point_cam.point.y = goal_point[1]
                    point_cam.point.z = goal_point[2]
                    self.ball_position_pub.publish(point_cam)

                    self.get_logger().info(
                        f"[Ball {i+1}] u={u:.1f}, v={v:.1f} | Depth={Z:.3f} m | " 
                        f"Position ({OUTPUT_FRAME_ID}): X={goal_point[0]:.3f}, Y={goal_point[1]:.3f}, Z={goal_point[2]:.3f}"
                    )
                    
                    # DEBUG VISUALIZATION LOGIC
                    cv2.circle(cv_image_color, (u, v), 5, (0, 0, 255), -1) 
                    
        # 7. Publish the Debug Image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image_color, encoding="bgr8")
            debug_msg.header = msg.header 
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

        # 8. Log "No balls spotted"
        if not ball_detected_in_frame:
             self.get_logger().info('No balls spotted')


    def camera_info_callback(self, msg: CameraInfo):
        self.get_logger().info("Recieved Camera Info")
        K = msg.k
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]
        
        # Destroy subscription to prevent log spam
        self.destroy_subscription(self.camera_info_sub) 
        self.get_logger().info(f"Camera intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        
        

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()