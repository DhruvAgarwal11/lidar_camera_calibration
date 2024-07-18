import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Declare launch arguments
    camera = LaunchConfiguration('camera', default='/sensors/camera')
    size = LaunchConfiguration('size', default='8x8')
    square = LaunchConfiguration('square', default='0.0318')
    k_coefficients = LaunchConfiguration('k_coefficients', default='2')
    bagfile = LaunchConfiguration('bagfile', default='/home/dhruvagarwal/webcamLidar0/webcamLidar0_0.db3')

    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value=camera, description='Camera topic base name'),
        DeclareLaunchArgument('size', default_value=size, description='Checkerboard size'),
        DeclareLaunchArgument('square', default_value=square, description='Checkerboard square size in meters'),
        DeclareLaunchArgument('k_coefficients', default_value=k_coefficients, description='Number of k coefficients for camera calibration'),

        #Include the play_rosbag.launch.py file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('lidar_camera_calibration'), 'launch', 'play_rosbag.launch.py'
            )]),
            launch_arguments={'bagfile': bagfile}.items(),
        ),

        # Run camera calibration
        Node(
            package='camera_calibration',
            executable='cameracalibrator',
            name='cameracalibrator',
            output='screen',
            arguments=[
                '--size', LaunchConfiguration('size'),
                '--square', LaunchConfiguration('square'),
                '--k-coefficients', LaunchConfiguration('k_coefficients'),
                '--no-service-check'
            ],
            remappings=[
                ('image', PathJoinSubstitution([LaunchConfiguration('camera'), 'image_color'])),
                ('camera', PathJoinSubstitution([LaunchConfiguration('camera'), 'camera_info'])),
            ]
        ),
    ])


