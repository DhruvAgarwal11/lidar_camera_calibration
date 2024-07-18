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
    bagfile = LaunchConfiguration('bagfile', default='/home/dhruvagarwal/webcamLidar0/webcamLidar0_0_updated.db3/webcamLidar0_0_updated.db3_0.db3')

    return LaunchDescription([
        DeclareLaunchArgument('camera', default_value=camera, description='Camera topic base name'),
        DeclareLaunchArgument('bagfile', default_value=bagfile, description='Bag file to play'),

        # Include the play_rosbag.launch.py file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([os.path.join(
                get_package_share_directory('lidar_camera_calibration'), 'launch', 'play_rosbag.launch.py'
            )]),
            launch_arguments={'bagfile': bagfile}.items(),
        ),

        # Nodelet manager equivalent
        Node(
            package='rclcpp_components',
            executable='component_container_mt',
            name='lidar_camera_manager',
            output='screen'
        ),

        # image_proc node
        Node(
            package='image_proc',
            executable='image_proc',
            name='image_proc_node1'
        ),

        # Run image_proc/rectify node
        Node(
            package='image_proc',
            executable='rectify_node',
            name='rectify_color',
            output='screen',
            remappings=[
                ('image_mono', [camera, '/image_color']),
                ('camera_info', [camera, '/camera_info']),
                ('image_rect', [camera, '/image_rect_color']),
            ]
        ),

        # Run image_view to display the unrectified image
        Node(
            package='image_view',
            executable='image_view',
            name='unrectified',
            output='screen',
            remappings=[
                ('image', [camera, '/image_color'])
            ]
        ),

        # Run image_view to display the rectified image
        Node(
            package='image_view',
            executable='image_view',
            name='rectified',
            output='screen',
            remappings=[
                ('image', [camera, '/image_rect_color'])
            ]
        ),
    ])

