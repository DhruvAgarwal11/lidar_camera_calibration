from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'lidar_camera_calibration'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name), glob('launch/*.py')),
        (os.path.join('share', package_name), glob('launch/*.launch.py')),
        (os.path.join('share', package_name), glob('launch/*.launch')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your_email@example.com',
    description='Lidar Camera Calibration Package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'calibrate_camera_lidar = lidar_camera_calibration.scripts.calibrate_camera_lidar:main',
            'test_projection = lidar_camera_calibration.scripts.test_projection:main'
        ],
    },
)

