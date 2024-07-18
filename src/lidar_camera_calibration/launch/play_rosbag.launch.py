import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch.actions import ExecuteProcess

def generate_launch_description():
    bagfile = LaunchConfiguration('bagfile', default='2016-11-22-14-32-13_test_updated.bag')

    return LaunchDescription([
        DeclareLaunchArgument(
            'bagfile',
            default_value='2016-11-22-14-32-13_test_updated.bag',
            description='Path to the rosbag file'
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bagfile, '--clock', '--loop'],
            output='screen'
        )
    ])

