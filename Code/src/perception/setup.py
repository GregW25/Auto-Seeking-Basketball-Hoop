from setuptools import find_packages, setup
from glob import glob

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/utilities',
            ['perception/utilities/yolo11s-seg.pt']),
        (('share/' + package_name + '/launch'), glob('launch/*.launch.py'))
    ],
    install_requires=['setuptools', 'rclpy', 'ultralytics', 'numpy'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='danielmunicio360@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_rs_pc_detector = perception.yolo_rs_pc_detector:main',
            'trajectory_calculator = perception.trajectory_calculator:main',
            'aruco_tf = perception.aruco_base_tf_publisher:main',
            'cam2_tf = perception.cam1_cam2_tf_publisher:main',
            'image_subscriber = perception.image_subscriber:main',
            'tool_tf = perception.tool_cam_tf_publisher:main'
        ],
    },
)
