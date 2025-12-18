import rclpy
import cv2
import os
import math
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import sensor_msgs_py.point_cloud2 as pc2
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

# COCO Class ID for 'sports ball' is 37
SPORTS_BALL_ID = 32

class YOLORSPCDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_rs_pc_detector')

        self.bridge = CvBridge()
        
        # --- 1. Load YOLOv11 Segmentation Model ---
        package_share_dir = get_package_share_directory('perception')
        # Ensure you have placed 'yolo11n-seg.pt' in the 'perception/utilities/' folder.
        model_path = os.path.join(package_share_dir, 'utilities', 'yolo11n-seg.pt') 
        self.model = YOLO(model_path)
        self.target_class_id = SPORTS_BALL_ID
        
        # Storage for the latest PointCloud2 message
        self.latest_pc_msg = None
        
        # --- 2. Subscribers ---
        # Image (for 2D detection) - using common RealSense topic
        self.image_sub = self.create_subscription(
            Image, 
            '/camera/color/image_raw', 
            self.image_callback, 
            1
        )
        
        # PointCloud2 (for 3D localization) - using common RealSense topic
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.pc_callback,
            1
        )
        
        # --- 3. Publisher ---
        # Publish the detected ball's position in the camera frame
        self.ball_position_pub = self.create_publisher(PointStamped, '/ball_position_camera_frame', 1) 

        self.get_logger().info('YOLO-RealSense Detector Node initialized and awaiting data.')

    def pc_callback(self, msg: PointCloud2):
        """Stores the latest PointCloud2 message."""
        self.latest_pc_msg = msg

    def image_callback(self, msg: Image):
        """Runs YOLO on the image and processes 3D data if a ball is found."""
        if self.latest_pc_msg is None:
            self.get_logger().warn('No PointCloud2 data received yet.')
            return

        # Use the point cloud that arrived *before* or *around* this image
        pc_msg = self.latest_pc_msg
        
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # Run YOLO prediction, filtering for only the 'sports ball' class (ID 37)
        results = self.model(cv_image, classes=[self.target_class_id], verbose=False)

        for result in results:
            if result.masks is not None and result.masks.data.cpu().numpy().size > 0:
                # Assuming one ball is found (take the first mask)
                mask = result.masks.data.cpu().numpy()[0]
                
                # Convert the float mask to a binary mask (H x W)
                mask_binary = (mask > 0.5) 
                
                # --- 3D Localization using Point Cloud ---
                
                # 1. Read the point cloud data
                # We read all (x, y, z) fields and skip NaN points
                points_xyz = []
                for p in pc2.read_points(pc_msg, field_names=('x','y','z'), skip_nans=False):
                    points_xyz.append(p)

                # 2. Reshape the flattened list of points to its organized H x W structure
                points_xyz = np.array(points_xyz)
                # The point cloud should be the same dimensions as the image
                try:
                    H, W = pc_msg.height, pc_msg.width
                    if H == 0 or W == 0:
                         H, W = cv_image.shape[:2] # Fallback to image size if PC not organized

                    points_xyz = points_xyz.reshape((H, W, 3))
                except ValueError:
                    self.get_logger().error(
                        f"Point cloud reshape failed: PC has {points_xyz.shape[0]} points, expected {H*W}."
                        " Ensure PointCloud2 is properly aligned with the image."
                    )
                    return
                
                # 3. Use the 2D mask to filter the 3D points
                # Ensure mask is the correct size before indexing
                if mask_binary.shape[0] != H or mask_binary.shape[1] != W:
                    # Resize the mask if its resolution is different from the PC/image
                    mask_binary = cv2.resize(mask_binary.astype(np.uint8), (W, H)).astype(bool)

                # Flatten the 3D array and apply the flattened mask
                filtered_points_flat = points_xyz[mask_binary]
                
                # 4. Calculate Centroid
                if filtered_points_flat.shape[0] > 0:
                    ball_centroid = np.nanmean(filtered_points_flat, axis=0)
                    
                    # Compute standard deviation for better logging/debugging
                    std_dev = np.nanstd(filtered_points_flat[:, 2])

                    # --- 5. Publish Result ---
                    point_stamped = PointStamped()
                    point_stamped.header = msg.header
                    
                    # The frame_id is already set by the RealSense to its camera frame (e.g., 'camera_link')
                    # which is what you requested (position from the camera).
                    point_stamped.header.frame_id = pc_msg.header.frame_id 
                    
                    point_stamped.point.x = float(ball_centroid[0])
                    point_stamped.point.y = float(ball_centroid[1])
                    point_stamped.point.z = float(ball_centroid[2])
                    
                    self.ball_position_pub.publish(point_stamped)
                    
                    self.get_logger().info(
                        f"[Ball Detected] Position ({point_stamped.header.frame_id}): "
                        f"X={point_stamped.point.x:.3f}m, Y={point_stamped.point.y:.3f}m, Z={point_stamped.point.z:.3f}m "
                        f"(Points: {filtered_points_flat.shape[0]}, Z-Std: {std_dev:.3f})"
                    )
                
                # Clear the point cloud after use to force a new sync
                self.latest_pc_msg = None
                return

        # If loop finishes without finding a ball
        self.get_logger().info('No sports ball spotted in the latest image.')
        
def main(args=None):
    rclpy.init(args=args)
    node = YOLORSPCDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()