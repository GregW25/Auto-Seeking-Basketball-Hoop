import rclpy
import cv2
import os
import message_filters
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO
import math

# --- CONSTANTS ---
SPORTS_BALL_ID = 32
DEPTH_SCALE = 1000.0 # Standard for depth images (mm to meters)

# Corrected Real-World Area for a Tennis Ball (D=0.067m, R=0.0335m). A_real = pi * R^2
# This constant is used in the 2D Area-Based Depth Estimation formula (Equation 10 in lab)
TENNIS_BALL_AREA = math.pi * (0.067 / 2.0)**2 

DEPTH_TOLERANCE_WINDOW = 0.15 # 15% tolerance for filtering measured depth around Z_pred

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')

        self.bridge = CvBridge()

        # Load YOLO model
        package_share_dir = get_package_share_directory('perception')
        model_path = os.path.join(package_share_dir, 'utilities', 'yolo11s-seg.pt')
        self.model = YOLO(model_path)

        self.get_logger().info(f"YOLO Model classes: {self.model.names}")

        self.SPORTS_BALL_ID = SPORTS_BALL_ID
        
        # Color Image (for YOLO detection)
        color_sub = message_filters.Subscriber(self, Image, '/camera/camera/color/image_raw')

        # Depth Image (for 3D distance reading)
        depth_sub = message_filters.Subscriber(self, Image, '/camera/camera/depth/image_rect_raw')
        
        # Camera Info (to get intrinsics fx, fy, cx, cy)
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            '/camera/camera/color/camera_info',
            self.camera_info_callback,
            1
        )

        # Approximate Time Synchronizer (ATS)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], 
            queue_size=10, 
            slop=0.1
        )

        # Register the new callback function for synchronized messages
        self.ts.registerCallback(self.image_depth_callback)

        self.ball_position_pub = self.create_publisher(PointStamped, '/goal_point', 1)
        self.debug_image_pub = self.create_publisher(Image, '/detection_image_out', 1)
        self.fx = self.fy = self.cx = self.cy = None


        self.get_logger().info('Image Subscriber Node initialized')


    def image_depth_callback(self, color_msg: Image, depth_msg: Image):
        if self.fx is None:
            self.get_logger().warn('Waiting for camera intrinsics...')
            return

        # 1. Convert ROS Image messages to OpenCV format
        try:
            cv_image_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_image_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        # Get resolutions for scaling
        h_color, w_color = cv_image_color.shape[:2] # e.g., 1080x1920
        h_depth, w_depth = cv_image_depth.shape[:2] # e.g., 480x848

        # Calculate scaling factors from color to depth resolution
        scale_x = w_depth / w_color
        scale_y = h_depth / h_color
        
        # 2. Run YOLO detection - Filter by class ID
        results = self.model(cv_image_color, classes=[self.SPORTS_BALL_ID], verbose=False)
        
        ball_detected_in_frame = False
        output_frame_id = color_msg.header.frame_id

        # --- PROCESS DETECTIONS ---
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()

                for i, mask in enumerate(masks):
                    
                    # 1. Prepare mask at COLOR resolution (for centroid and A_pixels)
                    mask_binary = (mask > 0.5).astype(np.uint8)
                    mask_resized_color = cv2.resize(
                        mask_binary, 
                        (w_color, h_color), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    # Compute moments (M['m00'] is the area A_pixels in COLOR pixels)
                    M = cv2.moments(mask_resized_color)

                    A_pixels = M['m00'] # Area of the mask in pixels

                    if A_pixels <= 5: # Skip if the mask area is too small
                        continue 
                        
                    # Compute centroid (u, v) in COLOR frame coordinates
                    u = int(M['m10'] / M['m00'])
                    v = int(M['m01'] / M['m00'])
                    ball_detected_in_frame = True

                    # --- Corrected 2D Area-Based Predicted Depth (Z_pred) Calculation (Equation 10) ---
                    # Z_pred = sqrt((fx * fy * A_real) / A_pixels)
                    
                    Z_pred_squared = (self.fx * self.fy * TENNIS_BALL_AREA) / A_pixels
                    
                    if Z_pred_squared <= 0:
                        self.get_logger().warn(f"Calculated Z_pred_squared is zero or negative. Skipping.")
                        continue

                    Z_pred = math.sqrt(Z_pred_squared)
                    
                    # Define the tolerance window (Z_min and Z_max in meters)
                    Z_min = Z_pred * (1.0 - DEPTH_TOLERANCE_WINDOW)
                    Z_max = Z_pred * (1.0 + DEPTH_TOLERANCE_WINDOW)
                    
                    # CRITICAL FIX: Create a mask at DEPTH resolution
                    mask_resized_depth = cv2.resize(
                        mask_binary, 
                        (w_depth, h_depth), 
                        interpolation=cv2.INTER_NEAREST
                    )
                    
                    # 3. Extract depth values from the depth-resolution mask
                    ball_depth_samples = cv_image_depth[mask_resized_depth == 1]
                    
                    # Filter out invalid depth values (0 means no reading/far away)
                    valid_depth_samples = ball_depth_samples[ball_depth_samples > 0]
                    
                    if len(valid_depth_samples) < 5:
                         Z = 0.0
                         self.get_logger().warn(f"Too few valid depth samples ({len(valid_depth_samples)}). Skipping.")
                         continue
                    else:
                         # --- Filter using Z_pred Window ---
                         # Convert Z_min and Z_max to millimeters for comparison
                         Z_min_mm = Z_min * DEPTH_SCALE 
                         Z_max_mm = Z_max * DEPTH_SCALE
                         
                         # Filter the samples to only those within the area-based window
                         filtered_samples = valid_depth_samples[
                             (valid_depth_samples >= Z_min_mm) & 
                             (valid_depth_samples <= Z_max_mm)
                         ]
                         
                         if len(filtered_samples) < 5:
                            # Fallback: If filtering rejects almost everything, use the area-based prediction
                            self.get_logger().warn(f"Measured depths failed Z_pred filter. Using Z_pred ({Z_pred:.3f}m).")
                            Z = Z_pred # Z is already in meters
                         else:
                            # Use the MEDIAN of the filtered measured samples for the final Z
                            raw_depth_val = np.median(filtered_samples)
                            Z = raw_depth_val / DEPTH_SCALE 
                            
                         self.get_logger().info(
                            f'Ball {i+1}: Z_pred={Z_pred:.3f}m | Final Depth={Z:.3f}m ' 
                            f'({len(filtered_samples)} samples used)'
                         )
                         
                    # 4. Find X, Y, Z in the Camera Frame (use UNscaled u, v with UNscaled fx, fy)
                    X = (((u - self.cx)*Z)/self.fx)
                    Y = (((v - self.cy)*Z)/self.fy)
    
                    # 5. Publish the 3D result
                    point_cam = PointStamped()
                    point_cam.header.stamp = color_msg.header.stamp
                    point_cam.header.frame_id = output_frame_id
                    point_cam.point.x = X
                    point_cam.point.y = Y
                    point_cam.point.z = Z
                    self.ball_position_pub.publish(point_cam)

                    self.get_logger().info(
                        f"[Ball {i+1}] u={u:.1f}, v={v:.1f} | Position: X={X:.3f}, Y={Y:.3f}, Z={Z:.3f}"
                    )

                    
        # 6. Publish the Debug Image
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(cv_image_color, encoding="bgr8")
            debug_msg.header = color_msg.header 
            self.debug_image_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish debug image: {e}")

        # 7. Log "No balls spotted"
        if not ball_detected_in_frame:
             self.get_logger().info('No balls spotted')


    def camera_info_callback(self, msg):
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