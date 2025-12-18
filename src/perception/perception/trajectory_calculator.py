import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker 
import collections
import math
import numpy as np
import tf2_ros
import tf2_geometry_msgs

# constants
G_CONST = 9.81
ACCEL_X = 0.0
ACCEL_Y = 0.0
ACCEL_Z = -G_CONST 

# intial config
NUM_PREDICTION_POINTS = 50 
PREDICTION_TIME_STEP = 0.05 
MIN_VELOCITY_THRESHOLD = 0.5 
MAX_POSSIBLE_SPEED = 20.0     
COOLDOWN_DURATION = 1.0 

# window settings
MIN_POINTS_TO_START = 4   
MAX_SMOOTHING_WINDOW = 10 

class TrajectoryCalculator(Node):
    def __init__(self):
        super().__init__('trajectory_calculator')
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.target_frame = 'base_link'

        # target plane config
        self.PLANE_A = 0.0 
        self.PLANE_B = .5
        self.PLANE_C = 0
        self.PLANE_D = -0.3562
        
        norm_mag = math.sqrt(self.PLANE_A**2 + self.PLANE_B**2 + self.PLANE_C**2)
        self.nx = self.PLANE_A / norm_mag
        self.ny = self.PLANE_B / norm_mag
        self.nz = self.PLANE_C / norm_mag
        self.d_norm = self.PLANE_D / norm_mag
        
        self.CONSTRAINT_Z_MIN = -0.0410 

        self.position_history = collections.deque(maxlen=MAX_SMOOTHING_WINDOW) 
        
        self.trajectory_pub = self.create_publisher(Marker, '/trajectory_marker', 10)
        self.plane_pub = self.create_publisher(Marker, '/plane_marker', 10)
        self.impact_pub = self.create_publisher(Marker, '/impact_point_marker', 10)
        self.constraint_pub = self.create_publisher(Marker, '/constraint_plane_marker', 10)

        self.predicted_point_active = False 
        self.cooldown_timer = None          

        self.ball_sub = self.create_subscription(
            PointStamped,
            '/goal_point',
            self.position_callback,
            10
        )
        
        self.timer = self.create_timer(1.0, self.publish_markers)
        self.get_logger().info('Trajectory Calculator: Weighted Least Squares Mode')

    def publish_markers(self):
        self.publish_plane_marker()
        self.publish_constraint_marker()

    def publish_plane_marker(self):
        plane_marker = Marker()
        plane_marker.header.stamp = self.get_clock().now().to_msg()
        plane_marker.header.frame_id = self.target_frame
        plane_marker.ns = "target_plane"
        plane_marker.id = 1
        plane_marker.type = Marker.CUBE 
        plane_marker.action = Marker.ADD

        center_dist = -self.d_norm
        plane_marker.pose.position.x = center_dist * self.nx
        plane_marker.pose.position.y = center_dist * self.ny
        plane_marker.pose.position.z = center_dist * self.nz

        vec_start = np.array([0.0, 0.0, 1.0])
        vec_target = np.array([self.nx, self.ny, self.nz])
        v_cross = np.cross(vec_start, vec_target)
        v_dot = np.dot(vec_start, vec_target)
        
        if np.linalg.norm(v_cross) < 1e-6:
            if v_dot > 0:
                plane_marker.pose.orientation.w = 1.0
            else:
                plane_marker.pose.orientation.x = 1.0
                plane_marker.pose.orientation.w = 0.0
        else:
            q_xyz = v_cross
            q_w = 1.0 + v_dot
            q_norm = math.sqrt(q_xyz[0]**2 + q_xyz[1]**2 + q_xyz[2]**2 + q_w**2)
            plane_marker.pose.orientation.x = q_xyz[0] / q_norm
            plane_marker.pose.orientation.y = q_xyz[1] / q_norm
            plane_marker.pose.orientation.z = q_xyz[2] / q_norm
            plane_marker.pose.orientation.w = q_w / q_norm

        plane_marker.scale.x = 2.0 
        plane_marker.scale.y = 2.0 
        plane_marker.scale.z = 0.005 
        plane_marker.color.r = 0.0
        plane_marker.color.g = 0.5
        plane_marker.color.b = 1.0
        plane_marker.color.a = 0.5 
        self.plane_pub.publish(plane_marker)

    def publish_constraint_marker(self):
        constraint_marker = Marker()
        constraint_marker.header.stamp = self.get_clock().now().to_msg()
        constraint_marker.header.frame_id = self.target_frame
        constraint_marker.ns = "constraint_plane"
        constraint_marker.id = 3
        constraint_marker.type = Marker.CUBE
        constraint_marker.action = Marker.ADD
        constraint_marker.pose.position.x = 0.0
        constraint_marker.pose.position.y = 0.0
        constraint_marker.pose.position.z = self.CONSTRAINT_Z_MIN
        constraint_marker.pose.orientation.w = 1.0
        constraint_marker.scale.x = 3.0
        constraint_marker.scale.y = 3.0
        constraint_marker.scale.z = 0.005
        constraint_marker.color.r = 1.0
        constraint_marker.color.g = 0.0
        constraint_marker.color.b = 0.0
        constraint_marker.color.a = 0.3 
        self.constraint_pub.publish(constraint_marker)

    def clear_impact_point(self):
        clear_marker = Marker()
        clear_marker.header.frame_id = self.target_frame
        clear_marker.ns = "impact_point"
        clear_marker.id = 2
        clear_marker.action = Marker.DELETE
        self.impact_pub.publish(clear_marker)
        self.predicted_point_active = False
        if self.cooldown_timer:
            self.cooldown_timer.cancel()
            self.cooldown_timer = None

    def position_callback(self, msg: PointStamped):
        # transform
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, 
                msg.header.frame_id, 
                rclpy.time.Time()
            )
            p_transformed = tf2_geometry_msgs.do_transform_point(msg, transform)
        except Exception as e:
            return

        new_point = {
            'x': p_transformed.point.x, 
            'y': p_transformed.point.y, 
            'z': p_transformed.point.z, 
            'time_sec': msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        }

        # outlier filtering
        if len(self.position_history) > 0:
            last = self.position_history[-1]
            dt_check = new_point['time_sec'] - last['time_sec']
            if dt_check > 0:
                dist = math.sqrt(
                    (new_point['x'] - last['x'])**2 + 
                    (new_point['y'] - last['y'])**2 + 
                    (new_point['z'] - last['z'])**2
                )
                speed = dist / dt_check
                if speed > MAX_POSSIBLE_SPEED:
                    return 

        self.position_history.append(new_point)

        # window logic
        if len(self.position_history) < MIN_POINTS_TO_START: 
            return

        window = list(self.position_history)
        
        # weighted least squares
        T = np.array([p['time_sec'] for p in window])
        X = np.array([p['x'] for p in window])
        Y = np.array([p['y'] for p in window])
        Z = np.array([p['z'] for p in window])
        
        T_rel = T - T[-1]

        # Weight Calculation
        num_points = len(window)
        weights = np.exp(np.linspace(-2.0, 0.0, num_points))

        # Linear Fit X/Y with Weights
        slope_x, intercept_x = np.polyfit(T_rel, X, 1, w=weights)
        slope_y, intercept_y = np.polyfit(T_rel, Y, 1, w=weights)
        
        # gravity-compensated fit Z
        Z_linearized = Z + 0.5 * G_CONST * (T_rel**2)
        slope_z, intercept_z = np.polyfit(T_rel, Z_linearized, 1, w=weights)

        vx, vy, vz = slope_x, slope_y, slope_z
        x0, y0, z0 = intercept_x, intercept_y, intercept_z
        
        # check velocity
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed < MIN_VELOCITY_THRESHOLD:
            self._publish_delete_marker(self.target_frame, "trajectory")
            self.clear_impact_point() 
            return 
        
        # calculate intersection
        nx, ny, nz, d = self.nx, self.ny, self.nz, self.d_norm
        p2 = 0.5 * (nx * ACCEL_X + ny * ACCEL_Y + nz * ACCEL_Z)
        p1 = nx * vx + ny * vy + nz * vz
        p0 = nx * x0 + ny * y0 + nz * z0 + d

        t_impact = -1.0
        
        if abs(p2) < 1e-6:
            if abs(p1) > 1e-6:
                t = -p0 / p1
                if t > 0: t_impact = t
        else: 
            delta = p1**2 - 4 * p2 * p0
            if delta >= 0:
                t1 = (-p1 + math.sqrt(delta)) / (2 * p2)
                t2 = (-p1 - math.sqrt(delta)) / (2 * p2)
                positive_times = [t for t in [t1, t2] if t > 0]
                if positive_times:
                    t_impact = min(positive_times)

        # visualization logic
        predicted_impact = None
        
        if t_impact > 0.001 and t_impact < 3.0: 
            predicted_x = x0 + vx * t_impact + 0.5 * ACCEL_X * (t_impact**2)
            predicted_y = y0 + vy * t_impact + 0.5 * ACCEL_Y * (t_impact**2)
            predicted_z = z0 + vz * t_impact + 0.5 * ACCEL_Z * (t_impact**2)

            if predicted_z < self.CONSTRAINT_Z_MIN:
                t_impact = -1.0
            
            if t_impact > 0:
                predicted_impact = Point(x=predicted_x, y=predicted_y, z=predicted_z)
                
                if not self.predicted_point_active:
                    self._publish_impact_point(self.target_frame, predicted_impact)
                    self.predicted_point_active = True
                    if self.cooldown_timer: self.cooldown_timer.cancel()
                    self.cooldown_timer = self.create_timer(COOLDOWN_DURATION, self.clear_impact_point)
                
        # draw Trajectory
        trajectory_marker = Marker()
        trajectory_marker.header.stamp = self.get_clock().now().to_msg()
        trajectory_marker.header.frame_id = self.target_frame
        trajectory_marker.ns = "trajectory"
        trajectory_marker.id = 0
        trajectory_marker.type = Marker.LINE_STRIP 
        trajectory_marker.action = Marker.ADD 
        trajectory_marker.scale.x = 0.01 
        trajectory_marker.color.r = 0.0
        trajectory_marker.color.g = 1.0
        trajectory_marker.color.b = 0.0
        trajectory_marker.color.a = 1.0

        trajectory_marker.points.append(Point(x=x0, y=y0, z=z0))
        
        t = PREDICTION_TIME_STEP 
        max_t = t_impact if (t_impact > 0.001 and t_impact < 3.0) else NUM_PREDICTION_POINTS * PREDICTION_TIME_STEP
        
        while t <= max_t:
            px = x0 + vx * t + 0.5 * ACCEL_X * (t**2)
            py = y0 + vy * t + 0.5 * ACCEL_Y * (t**2)
            pz = z0 + vz * t + 0.5 * ACCEL_Z * (t**2)
            trajectory_marker.points.append(Point(x=px, y=py, z=pz))
            t += PREDICTION_TIME_STEP
            
            if t > max_t and predicted_impact:
                 trajectory_marker.points.append(predicted_impact)

        if len(trajectory_marker.points) > 1:
            self.trajectory_pub.publish(trajectory_marker)
        else:
            self._publish_delete_marker(self.target_frame, "trajectory")

    def _publish_impact_point(self, frame_id, point: Point):
        impact_marker = Marker()
        impact_marker.header.stamp = self.get_clock().now().to_msg()
        impact_marker.header.frame_id = frame_id
        impact_marker.ns = "impact_point"
        impact_marker.id = 2
        impact_marker.type = Marker.SPHERE 
        impact_marker.action = Marker.ADD
        impact_marker.pose.position = point
        impact_marker.scale.x = 0.08
        impact_marker.scale.y = 0.08
        impact_marker.scale.z = 0.08
        impact_marker.color.r = 1.0
        impact_marker.color.g = 0.2
        impact_marker.color.b = 0.0
        impact_marker.color.a = 1.0 
        self.impact_pub.publish(impact_marker)

    def _publish_delete_marker(self, frame_id, ns):
        delete_marker = Marker()
        delete_marker.header.frame_id = frame_id
        delete_marker.ns = ns
        delete_marker.id = 0
        delete_marker.action = Marker.DELETE
        self.trajectory_pub.publish(delete_marker) 
        
def main(args=None):
    rclpy.init(args=args)
    trajectory_calculator = TrajectoryCalculator()
    try:
        rclpy.spin(trajectory_calculator)
    except KeyboardInterrupt:
        pass
    
    trajectory_calculator.clear_impact_point()
    rclpy.spin_once(trajectory_calculator, timeout_sec=0.1)
    trajectory_calculator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()