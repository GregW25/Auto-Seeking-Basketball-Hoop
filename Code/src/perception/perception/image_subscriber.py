import rclpy
import cv2
import message_filters
import numpy as np
import math
import time
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge, CvBridgeError

# config
REAL_BALL_DIAMETER_M = 0.0889 
DEPTH_SCALE = 1000.0           
DEPTH_MEDIAN_TOLERANCE_MM = 300.0 # Relaxed for flight

# HSV values
HUE_MIN = 22    
HUE_MAX = 40    
SAT_MIN = 100   
SAT_MAX = 255   
VAL_MIN = 65   
VAL_MAX = 255

# logic constants
MAX_TRACKING_DISTANCE = 4.0 # Extended for flight
MOTION_SWITCH_THRESHOLD = 0.5 # m/s

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('blob_detector_subscriber')

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        
        # velocity calcs
        self.prev_pos = None     # (x, y, z)
        self.prev_time = None    # timestamp
        self.current_velocity = 0.0

        # topics
        color_topic = '/camera/camera/color/image_raw'
        depth_topic = '/camera/camera/depth/image_rect_raw'
        info_topic  = '/camera/camera/color/camera_info'

        # subscribers
        color_sub = message_filters.Subscriber(self, Image, color_topic)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic)
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo, 
            info_topic,
            self.camera_info_callback,
            1
        )

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], 
            queue_size=10, 
            slop=0.05
        )
        self.ts.registerCallback(self.image_depth_callback)

        # publishers
        self.ball_position_pub = self.create_publisher(PointStamped, '/goal_point', 1)
        self.debug_image_pub = self.create_publisher(Image, '/detection_image_out', 1)
        self.mask_image_pub = self.create_publisher(Image, '/ball/mask_out', 1)

        self.get_logger().info(f"Blob Detector: MOTION-AWARE FLIGHT MODE")


    def image_depth_callback(self, color_msg: Image, depth_msg: Image):
        if self.fx is None:
            return

        try:
            cv_image_color = self.bridge.imgmsg_to_cv2(color_msg, desired_encoding='bgr8')
            cv_image_depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f"CvBridge error: {e}")
            return

        h_depth, w_depth = cv_image_depth.shape[:2] 
        h_color, w_color = cv_image_color.shape[:2]
        output_frame_id = color_msg.header.frame_id
        current_time = time.time()

        # color detection
        hsv_image = cv2.cvtColor(cv_image_color, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([HUE_MIN, SAT_MIN, VAL_MIN])
        upper_bound = np.array([HUE_MAX, SAT_MAX, VAL_MAX])
        
        mask_binary_color = cv2.inRange(hsv_image, lower_bound, upper_bound)
        
        # close connects gaps
        kernel = np.ones((5,5), np.uint8)
        mask_binary_color = cv2.morphologyEx(mask_binary_color, cv2.MORPH_CLOSE, kernel)
        mask_binary_color = cv2.dilate(mask_binary_color, None, iterations=1)
        
        contours, _ = cv2.findContours(mask_binary_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            self._publish_debug(color_msg, cv_image_color, mask_binary_color)
            return

        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)

        if M["m00"] < 50:
            self._publish_debug(color_msg, cv_image_color, mask_binary_color)
            return

        u = int(M["m10"] / M["m00"])
        v = int(M["m01"] / M["m00"])
        
        area_pixels = M["m00"]
        D_pixels = 2 * math.sqrt(area_pixels / math.pi)
        
        # sensor depth
        sensor_z = 0.0
        blob_mask = np.zeros_like(mask_binary_color)
        cv2.drawContours(blob_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
        blob_mask_scaled = cv2.resize(blob_mask, (w_depth, h_depth), interpolation=cv2.INTER_NEAREST)
        ball_depth_samples = cv_image_depth[blob_mask_scaled > 0]
        valid_depth_samples = ball_depth_samples[ball_depth_samples > 0]

        if len(valid_depth_samples) >= 3: 
            sensor_z = np.median(valid_depth_samples) / DEPTH_SCALE

        # area depth
        area_z = 0.0
        if D_pixels > 0:
            area_z = (REAL_BALL_DIAMETER_M * self.fx) / D_pixels

        # motion logic
        Z = 0.0
        method = "NONE"

        # check Velocity to decide strategy
        is_moving_fast = False
        if self.prev_pos is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 0:
                # Calculate simple 3D distance
                dx = 0
                dy = 0
                dz = abs(area_z - self.prev_pos[2]) # Use Area Z for speed diff
                dist_moved = math.sqrt(dx*dx + dy*dy + dz*dz)
                velocity = dist_moved / dt
                
                # smooth-ish velocity - check later
                self.current_velocity = (0.7 * self.current_velocity) + (0.3 * velocity)
                
                if self.current_velocity > MOTION_SWITCH_THRESHOLD:
                    is_moving_fast = True

        # decision split
        if is_moving_fast and area_z > 0.1:
            # in flight -> area
            Z = area_z
            method = "FLIGHT(Area)"
        
        elif sensor_z > 0.1 and area_z > 0.1:
            # slow -> average
            if abs(sensor_z - area_z) > 1.0:
                Z = sensor_z # Mismatch? Trust sensor
            else:
                Z = (sensor_z + area_z) / 2.0
                method = "FUSED"
        
        elif area_z > 0.1:
            Z = area_z
            method = "AREA(Fallback)"
            
        elif sensor_z > 0.1:
            Z = sensor_z
            method = "SENSOR"

        # output
        draw_color = (0, 0, 255)
        status_text = f"{method}"

        if Z > 0.001:
            # update traj hsitory
            self.prev_pos = (0, 0, Z)
            self.prev_time = current_time

            if Z <= MAX_TRACKING_DISTANCE:
                draw_color = (0, 255, 0)
                
                X = (((u - self.cx)*Z)/self.fx)
                Y = (((v - self.cy)*Z)/self.fy)
                
                point_cam = PointStamped()
                point_cam.header.stamp = color_msg.header.stamp
                point_cam.header.frame_id = output_frame_id
                point_cam.point.x = X
                point_cam.point.y = Y
                point_cam.point.z = Z
                self.ball_position_pub.publish(point_cam)
                
                log_msg = f"[{method}] Z={Z:.2f}m"
                if is_moving_fast:
                    log_msg += f" | FAST ({self.current_velocity:.1f} m/s)"
                self.get_logger().info(log_msg)

        # debug
        cv2.circle(cv_image_color, (u, v), int(D_pixels / 2), draw_color, 2)
        cv2.putText(cv_image_color, f"{status_text} v={self.current_velocity:.1f}", (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
        
        self._publish_debug(color_msg, cv_image_color, mask_binary_color)


    def _publish_debug(self, header_src, color_img, mask_img):
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(color_img, encoding="bgr8")
            debug_msg.header = header_src.header 
            self.debug_image_pub.publish(debug_msg)

            mask_msg = self.bridge.cv2_to_imgmsg(mask_img, "mono8")
            mask_msg.header = header_src.header
            self.mask_image_pub.publish(mask_msg)
        except CvBridgeError:
            pass

    def camera_info_callback(self, msg):
        K = msg.k
        self.fx = K[0]
        self.fy = K[4]
        self.cx = K[2]
        self.cy = K[5]
        self.destroy_subscription(self.camera_info_sub) 
        self.get_logger().info(f"Camera intrinsics: fx={self.fx:.2f}")

def main(args=None):
    rclpy.init(args=args)
    image_subscriber = ImageSubscriber()
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()