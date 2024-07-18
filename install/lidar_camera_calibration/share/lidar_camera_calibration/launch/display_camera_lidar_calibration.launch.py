from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    lidar_camera_calibration_dir = get_package_share_directory('lidar_camera_calibration')
    play_rosbag_launch = os.path.join(lidar_camera_calibration_dir, 'launch', 'play_rosbag.launch.py')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera',
            default_value='/sensors/camera',
            description='Camera topic namespace'
        ),
        LogInfo(
            msg=f"Loading play_rosbag_launch file from: {play_rosbag_launch}"
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(play_rosbag_launch),
            launch_arguments={'bagfile': '/home/dhruvagarwal/webcamLidar0/webcamLidar0_0_updated.db3/webcamLidar0_0_updated.db3_0.db3'}.items(),
        ),
        Node(
            package='image_proc',
            executable='image_proc',
            name='image_proc_node1'
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='world_velodyne_tf',
            arguments=['0', '0', '0', '0', '0', '0', 'world', 'velodyne'],
            output='screen'
        ),
        Node(
            package='lidar_camera_calibration',
            executable='calibrate_camera_lidar.py',
            name='calibrate_camera_lidar',
            output='screen',
            parameters=[{
                'camera_info_topic': '/sensors/camera/camera_info',
                'image_color_topic': '/sensors/camera/image_rect_color',
                'velodyne_points_topic': '/sensors/velodyne_points',
                'camera_lidar_topic': '/sensors/camera/camera_lidar',
                'project_mode': True
            }]
        ),
        Node(
            package='image_view',
            executable='image_view',
            name='camera_lidar_projection',
            output='screen',
            remappings=[('image', '/sensors/camera/camera_lidar')]
        ),
    ])

