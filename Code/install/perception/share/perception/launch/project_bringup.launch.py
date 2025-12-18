from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.events import Shutdown
from launch.actions import IncludeLaunchDescription  
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, EmitEvent
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # RealSense (include rs_launch.py)
    """
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '1920x1080x30',
        }.items(),
    )
    """

    # Image Subscriber node
    image_subscriber_node = Node(
        package='perception',
        executable='image_subscriber',
        name='image_subscriber',
        output='screen',
        parameters=[{
        }]
    )

    ar_marker_launch_arg = DeclareLaunchArgument(
        'ar_marker',
        default_value='ar_marker_6'
    )
    ar_marker = LaunchConfiguration('ar_marker')


    # Aruco TF node: Canmera1 -> aruco
    # -------------------------------------------------
    # This TF is static because the "world" frame does not move.
    # It is necessary to define the "world" frame for MoveIt to work properly as this is the defualt planning frame.
    # -------------------------------------------------
    aruco_tf_node = Node(
        package='perception',
        executable='aruco_tf',
        name='aruco_base_tf_publisher',
        output='screen',
        parameters=[{
            'ar_marker': ar_marker,
        }]
    )

    # cam TF: camera1 -> camera_link
    # -------------------------------------------------
    # This TF is static because the "world" frame does not move.
    # It is necessary to define the "world" frame for MoveIt to work properly as this is the defualt planning frame.
    # -------------------------------------------------
    cam_tf_node = Node(
        package='perception',
        executable='cam2_tf',
        name='cam1_cam2_tf_publisher',
        output='screen',
        parameters=[{
            'ar_marker': ar_marker,
        }]
    )

    trajectory_calculator_node = Node(
        package='perception',  
        executable='trajectory_calculator',
        name='trajectory_calculator',
        output='screen',
        parameters=[{
        }]
    )


    # -------------------------
    # Global shutdown on any process exit
    # -------------------------
    shutdown_on_any_exit = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='SOMETHING BONKED'))]
        )
    )
    
    return LaunchDescription([
        ar_marker_launch_arg,
        image_subscriber_node,
        aruco_tf_node,
        cam_tf_node,
        trajectory_calculator_node,
        shutdown_on_any_exit
    ])
