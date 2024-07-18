#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.0.1
Date    : Jan 18, 2019

Description:
Script to update the camera calibration data into the ROSBAG file
Ensure that this file has executable permissions

Example Usage:
$ python3 update_camera_info.py rosbag.db3 calibration.yaml

Notes:
Make sure this file has executable permissions:
$ chmod +x update_camera_info.py
'''

# Built-in modules
import os
import sys
import yaml

# ROS 2 modules
import rclpy
from rclpy.node import Node
import rosbag2_py
from sensor_msgs.msg import CameraInfo
from rclpy.serialization import serialize_message, deserialize_message


def load_calibration_data(filename):
    # Open calibration file
    with open(filename, 'r') as stream:
        try:
            calibration = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            rclpy.logging.get_logger('update_camera_info').error(f'Error loading calibration data: {exc}')
            sys.exit(1)
    return calibration

class UpdateCameraInfoNode(Node):
    def __init__(self):
        super().__init__('update_camera_info')
    
    def update_bag(self, bag_file, calib_file):
        # Load ROSBAG file
        self.get_logger().info(f'Bag Filename: {bag_file}')
        storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions('', '')
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Output file
        folder = os.path.dirname(bag_file)
        output_name = os.path.splitext(os.path.basename(bag_file))[0] + '_updated.db3'
        output_file = os.path.join(folder, output_name)
        
        writer_storage_options = rosbag2_py.StorageOptions(uri=output_file, storage_id='sqlite3')
        writer = rosbag2_py.SequentialWriter()
        writer.open(writer_storage_options, converter_options)
        
        # Load calibration data
        calibration = load_calibration_data(calib_file)

        # Update calibration data
        self.get_logger().info(f'Updating /sensors/camera/camera_info data...')
        
        topics = reader.get_all_topics_and_types()
        for topic in topics:
            writer.create_topic(topic)

        while reader.has_next():
            (topic, data, t) = reader.read_next()
            if topic == '/sensors/camera/camera_info':
                msg = deserialize_message(data, CameraInfo)
                msg.d = calibration['distortion_coefficients']['data']
                msg.k = calibration['camera_matrix']['data']
                msg.r = calibration['rectification_matrix']['data']
                msg.p = calibration['projection_matrix']['data']
                data = serialize_message(msg)
            writer.write(topic, data, t)
        self.get_logger().info('Done')

        # Close bag file
        #reader.close()
        #writer.close()

def main(args=None):
    rclpy.init(args=args)
    node = UpdateCameraInfoNode()
    
    if len(sys.argv) == 3:
        bag_file = sys.argv[1]
        calib_file = sys.argv[2]
    else:
        node.get_logger().error('Usage: update_camera_info.py <bag_file> <calibration_file>')
        sys.exit(1)

    node.update_bag(bag_file, calib_file)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

