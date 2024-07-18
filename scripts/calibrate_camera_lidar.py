#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Author  : Heethesh Vhavle
Email   : heethesh@cmu.edu
Version : 1.2.1
Date    : Jan 20, 2019

Description:
Script to find the transformation between the Camera and the LiDAR

Example Usage:
1. To perform calibration using the GUI to pick correspondences:

    $ python3 calibrate_camera_lidar.py --calibrate

    The point correspondences will be save as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/img_corners.npy
    - PKG_PATH/calibration_data/lidar_camera_calibration/pcl_corners.npy

    The calibrate extrinsic are saved as following:
    - PKG_PATH/calibration_data/lidar_camera_calibration/extrinsics.npz
    --> 'euler' : euler angles (3, )
    --> 'R'     : rotation matrix (3, 3)
    --> 'T'     : translation offsets (3, )

2. To display the LiDAR points projected on to the camera plane:

    $ ros2 launch lidar_camera_calibration display_camera_lidar_calibration.launch.py

Notes:
Make sure this file has executable permissions:
$ chmod +x calibrate_camera_lidar.py

References: 
http://wiki.ros.org/message_filters
http://wiki.ros.org/cv_bridge/Tutorials/
http://docs.ros.org/api/image_geometry/html/python/
http://wiki.ros.org/ROS/Tutorials/WritingPublisherSubscribe
'''

# Built-in modules
import os
import sys
import time
import threading
import multiprocessing

# External modules
import cv2
import numpy as np
import matplotlib.cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ROS 2 modules
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import serialize_message, deserialize_message
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from cv_bridge import CvBridge, CvBridgeError
import tf2_ros
from transforms3d.euler import mat2euler
import ros2_numpy
import image_geometry

# Global variables
OUSTER_LIDAR = False
PAUSE = False
FIRST_TIME = True
KEY_LOCK = threading.Lock()
TF_BUFFER = None
TF_LISTENER = None
CV_BRIDGE = CvBridge()
CAMERA_MODEL = image_geometry.PinholeCameraModel()

# Global paths
PKG_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
CALIB_PATH = 'calibration_data/lidar_camera_calibration'


'''
Keyboard handler thread
Inputs: None
Outputs: None
'''
def handle_keyboard():
    global KEY_LOCK, PAUSE
    input('Press [ENTER] to pause and pick points\n')
    with KEY_LOCK: PAUSE = True


'''
Start the keyboard handler thread
Inputs: None
Outputs: None
'''
def start_keyboard_handler():
    keyboard_t = threading.Thread(target=handle_keyboard)
    keyboard_t.daemon = True
    keyboard_t.start()


'''
Save the point correspondences and image data
Points data will be appended if file already exists

Inputs:
    data - [numpy array] - points or opencv image
    filename - [str] - filename to save
    folder - [str] - folder to save at
    is_image - [bool] - to specify whether points or image data

Outputs: None
'''
def save_data(data, filename, folder, is_image=False):
    # Empty data
    if not len(data): return

    # Handle filename
    filename = os.path.join(PKG_PATH, os.path.join(folder, filename))
    
    # Create folder
    try:
        os.makedirs(os.path.join(PKG_PATH, folder))
    except OSError:
        if not os.path.isdir(os.path.join(PKG_PATH, folder)): raise

    # Save image
    if is_image:
        cv2.imwrite(filename, data)
        return

    # Save points data
    if os.path.isfile(filename):
        print(f'Updating file: {filename}')
        data = np.vstack((np.load(filename), data))
    np.save(filename, data)


'''
Runs the image point selection GUI process

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    now - [int] - ROS bag time in seconds
    rectify - [bool] - to specify whether to rectify image or not

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/img_corners.npy
'''
def extract_points_2D(img_msg, now, rectify=False):
    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e: 
        print(e)
        return

    # Rectify image
    if rectify: CAMERA_MODEL.rectifyImage(img, img)
    disp = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2RGB)

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Select 2D Image Points - %d' % now)
    ax.set_axis_off()
    ax.imshow(disp)

    # Pick points
    picked, corners = [], []
    def onclick(event):
        x = event.xdata
        y = event.ydata
        if (x is None) or (y is None): return

        # Display the picked point
        picked.append((x, y))
        corners.append((x, y))
        print('IMG:', str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Save corner points and image
    rect = '_rect' if rectify else ''
    if len(corners) > 1: del corners[-1] # Remove last duplicate
    save_data(corners, f'img_corners{rect}.npy', CALIB_PATH)
    save_data(img, f'image_color{rect}-{now}.jpg', os.path.join(CALIB_PATH, 'images'), True)


'''
Runs the LiDAR point selection GUI process

Inputs:
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    now - [int] - ROS bag time in seconds

Outputs:
    Picked points saved in PKG_PATH/CALIB_PATH/pcl_corners.npy
'''
def extract_points_3D(velodyne, now):
    # Log PID
    print('3D Picker PID: [%d]' % os.getpid())
    # Extract points data
    points = ros2_numpy.point_cloud2.pointcloud2_to_array(velodyne)
    
    # Group all beams together and pick the first 4 columns for X, Y, Z, intensity.
    if OUSTER_LIDAR: points = points.reshape(-1, 9)[:, :4]

    # Select points within chessboard range
    print('this is where the points start')
    print(points)
    points = np.array([(p['x'], p['y'], p['z'], p['intensity']) for p in points.flatten()], dtype=np.float32)
    points = points[~np.isnan(points).any(axis=1)][:]
    # inrange = np.where((points[:, 0] > 0) &
    #                    (points[:, 0] < 2.5) &
    #                    (np.abs(points[:, 1]) < 2.5) &
    #                    (points[:, 2] < 2))
    # points = points[inrange[0]]
    print(points)
    print(points.shape)
    if points.shape[0] > 5:
        print('PCL points available: %d' % points.shape[0])
    else:
        print('Very few PCL points available in range')
        return

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('hsv')
    colors = cmap(points[:, -1] / np.max(points[:, -1]))

    # Setup matplotlib GUI
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Select 3D LiDAR Points - %d' % now, color='white')
    ax.set_axis_off()
    ax.set_facecolor((0, 0, 0))
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=2, picker=5)

    # Equalize display aspect ratio for all axes
    max_range = (np.array([points[:, 0].max() - points[:, 0].min(), 
        points[:, 1].max() - points[:, 1].min(),
        points[:, 2].max() - points[:, 2].min()]).max() / 2.0)
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Pick points
    picked, corners = [], []
    def onpick(event):
        ind = event.ind[0]
        x, y, z = event.artist._offsets3d

        # Ignore if same point selected again
        if picked and (x[ind] == picked[-1][0] and y[ind] == picked[-1][1] and z[ind] == picked[-1][2]):
            return
        
        # Display picked point
        picked.append((x[ind], y[ind], z[ind]))
        corners.append((x[ind], y[ind], z[ind]))
        print('PCL: %s' % str(picked[-1]))

        if len(picked) > 1:
            # Draw the line
            temp = np.array(picked)
            ax.plot(temp[:, 0], temp[:, 1], temp[:, 2])
            ax.figure.canvas.draw_idle()

            # Reset list for future pick events
            del picked[0]

    # Display GUI
    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

    # Save corner points
    if len(corners) > 1: del corners[-1] # Remove last duplicate
    save_data(corners, 'pcl_corners.npy', CALIB_PATH)


'''
Calibrate the LiDAR and image points using OpenCV PnP RANSAC
Requires minimum 5 point correspondences

Inputs:
    points2D - [numpy array] - (N, 2) array of image points
    points3D - [numpy array] - (N, 3) array of 3D points

Outputs:
    Extrinsics saved in PKG_PATH/CALIB_PATH/extrinsics.npz
'''
def calibrate(points2D=None, points3D=None):
    # Ensure points2D and points3D are numpy arrays with the correct shape
    points2D = np.asarray(points2D, dtype=np.float32)
    points3D = np.asarray(points3D, dtype=np.float32)
    points2D = np.array([
        [242.87662337662343, 218.20129870129864],
        [268.8506493506494, 216.90259740259734],
        [297.42207792207796, 216.90259740259734],
        [229.88961038961045, 231.18831168831161],
        [255.8636363636364, 231.18831168831161],
        [284.43506493506504, 228.590909090909],
        [311.7077922077923, 228.590909090909],
        [242.87662337662343, 244.1753246753246],
        [270.1493506493507, 244.1753246753246],
        [298.72077922077926, 241.57792207792204],
        [231.18831168831176, 258.4610389610389],
        [255.8636363636364, 258.4610389610389],
        [284.43506493506504, 257.1623376623376],
        [311.7077922077923, 257.1623376623376],
        [244.17532467532473, 272.7467532467532],
        [272.7467532467533, 272.7467532467532],
        [301.31818181818187, 270.1493506493506],
        [232.48701298701306, 285.7337662337662],
        [258.461038961039, 283.13636363636357],
        [285.73376623376635, 284.4350649350649],
        [313.0064935064936, 283.13636363636357],
        [246.77272727272734, 300.01948051948045],
        [271.448051948052, 301.31818181818176],
        [300.01948051948057, 301.31818181818176],
        [230.1766917293233, 300.03315105946683],
        [229.5205058099795, 271.8171565276828],
        [228.86431989063573, 244.25734791524263],
        [228.20813397129183, 218.00991114149008],
        [310.2313738892686, 214.0727956254272],
        [310.8875598086125, 242.944976076555],
        [311.54374572795626, 269.8485987696514],
        [312.85611756664383, 297.4084073820916]
    ], dtype=np.float32)
    # points2D = np.array([
    #     (192.22727272727278, 325.9935064935064),
    #     (206.51298701298705, 324.69480519480516),
    #     (206.51298701298705, 338.98051948051943),
    #     (192.22727272727278, 338.98051948051943),
    #     (207.81168831168836, 338.98051948051943),
    #     (220.79870129870133, 337.68181818181813),
    #     (220.79870129870133, 351.9675324675324),
    #     (207.81168831168836, 351.9675324675324),
    #     (207.81168831168836, 367.551948051948),
    #     (277.9415584415585, 362.3571428571428),
    #     (279.24025974025983, 405.21428571428567),
    #     (209.11038961038966, 411.7077922077922),
    #     (220.79870129870133, 322.09740259740255),
    #     (290.92857142857144, 319.49999999999994),
    #     (293.52597402597405, 389.6298701298701),
    #     (223.39610389610394, 394.8246753246753),
    #     (193.52597402597408, 428.59090909090907),
    #     (193.52597402597408, 413.0064935064935),
    #     (209.11038961038966, 413.0064935064935),
    #     (209.11038961038966, 427.29220779220776),
    #     (235.0844155844156, 337.68181818181813),
    #     (248.07142857142864, 336.3831168831168),
    #     (249.37012987012994, 350.66883116883116),
    #     (237.68181818181822, 350.66883116883116)
    # ], dtype=np.float32)

    # LiDAR points as (x, y, z)
    # points3D = np.array([
    #     (0.8944355845451355, 0.12289288640022278, 0.06564098596572876),
    #     (0.8571299314498901, 0.09007646143436432, 0.0673019289970398),
    #     (0.9199362397193909, 0.10321550071239471, 0.04355451464653015),
    #     (0.8992691040039062, 0.12111225724220276, 0.045010894536972046),
    #     (0.8568289875984192, 0.07846183329820633, 0.0389554500579834),
    #     (0.884689211845398, 0.054460570216178894, 0.04098963737487793),
    #     (0.8591641187667847, 0.05081554502248764, 0.014291435480117798),
    #     (0.8603937029838562, 0.08274073898792267, 0.013112843036651611),
    #     (0.8721113204956055, 0.08605317026376724, -0.02726753056049347),
    #     (0.885108470916748, -0.07861436158418655, -0.017483249306678772),
    #     (0.8798549175262451, -0.08601921051740646, -0.10656394064426422),
    #     (0.8715276718139648, 0.08171548694372177, -0.12920255959033966),
    #     (0.8777987957000732, 0.05427312105894089, 0.07124970853328705),
    #     (0.9211138486862183, -0.11271709948778152, 0.07783375680446625),
    #     (0.8988204002380371, -0.11917539685964584, -0.07884867489337921),
    #     (0.8790938854217529, 0.04879779368638992, -0.09143222868442535),
    #     (0.8473900556564331, 0.10885540395975113, -0.15723301470279694),
    #     (0.8664851188659668, 0.11415653675794601, -0.13256020843982697),
    #     (0.8542890548706055, 0.07839075475931168, -0.12372596561908722),
    #     (0.8465039730072021, 0.0755922719836235, -0.15401490032672882),
    #     (0.8808670043945312, 0.020449362695217133, 0.044038817286491394),
    #     (0.8920693397521973, -0.01389659196138382, 0.04511384665966034),
    #     (0.8779679536819458, -0.015042625367641449, 0.014388129115104675),
    #     (0.8668255805969238, 0.018662475049495697, 0.0128202885389328)
    # ], dtype=np.float32)
    points3D = np.array([
        [0.9319794178009033, 0.04673728346824646, 0.0692247748374939],
        [0.9167802333831787, -0.016607195138931274, 0.07292237877845764],
        [0.9061504006385803, -0.08628842234611511, 0.07414591312408447],
        [0.9405319690704346, 0.08027622103691101, 0.036253154277801514],
        [0.936386227607727, 0.011476099491119385, 0.037776023149490356],
        [0.9032391309738159, -0.049188047647476196, 0.042949795722961426],
        [0.8933006525039673, -0.11447858810424805, 0.044173240661621094],
        [0.9150751233100891, 0.0474487841129303, 0.008582621812820435],
        [0.9291077256202698, -0.01962849497795105, 0.006979256868362427],
        [0.89564448595047, -0.08475831151008606, 0.01353272795677185],
        [0.9410264492034912, 0.08168980479240417, -0.02972090244293213],
        [0.9071815013885498, 0.014617502689361572, -0.02343738079071045],
        [0.9143204092979431, -0.05471210181713104, -0.026030272245407104],
        [0.8882046937942505, -0.11592581868171692, -0.017803668975830078],
        [0.9272568225860596, 0.04672238230705261, -0.05916836857795715],
        [0.9051728248596191, -0.01791861653327942, -0.056964218616485596],
        [0.8899614810943604, -0.08406472206115723, -0.051737815141677856],
        [0.917680025100708, 0.08080253005027771, -0.09010961651802063],
        [0.8961093425750732, 0.01475726068019867, -0.08528846502304077],
        [0.8921880125999451, -0.050183817744255066, -0.08513164520263672],
        [0.8824797868728638, -0.11525624990463257, -0.08320978283882141],
        [0.9147354960441589, 0.04667210578918457, -0.1260772943496704],
        [0.8989923000335693, -0.01965939998626709, -0.12264207005500793],
        [0.8790596127510071, -0.0810810923576355, -0.11457836627960205],
        [0.9257417917251587, 0.07939471304416656, -0.1275320053100586],
        [0.9318975210189819, 0.07968851923942566, -0.06113061308860779],
        [0.9318740367889404, 0.0783129632472992, 0.0065000057220458984],
        [0.9495140314102173, 0.07753074169158936, 0.0698036253452301],
        [0.8845838904380798, -0.11221310496330261, 0.07878002524375916],
        [0.8912172913551331, -0.1139746904373169, 0.01415163278579712],
        [0.8854012489318848, -0.11527898907661438, -0.050471752882003784],
        [0.8629096746444702, -0.11260122060775757, -0.10981485247612]
    ], dtype=np.float32)
    
    assert points2D.shape[0] == points3D.shape[0], "Number of 2D and 3D points must be the same"
    assert points2D.shape[0] >= 4, "At least 4 points are required"
    assert points2D.shape[1] == 2, "points2D should be of shape (N, 2)"
    assert points3D.shape[1] == 3, "points3D should be of shape (N, 3)"
    
    # Obtain camera matrix and distortion coefficients
    camera_matrix = CAMERA_MODEL.intrinsicMatrix()
    dist_coeffs = CAMERA_MODEL.distortionCoeffs()
    
    # Print camera matrix and distortion coefficients for debugging
    print("Camera Matrix:\n", camera_matrix)
    print("Distortion Coefficients:\n", dist_coeffs)
    print("2D Points:\n", points2D)
    print("3D Points:\n", points3D)

    # Ensure camera matrix is a valid 3x3 matrix
    camera_matrix = np.array(camera_matrix, dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.array(dist_coeffs, dtype=np.float64)
    
    # Estimate extrinsics
    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
        points3D, points2D, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE, reprojectionError=70.0, confidence=0.99)
    
    if not success:
        print('Initial estimation unsuccessful, skipping refinement')
        return
    
    # Compute re-projection error
    points2D_reproj = cv2.projectPoints(points3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0].squeeze(1)
    assert points2D_reproj.shape == points2D.shape
    error = (points2D_reproj - points2D)[inliers]  # Compute error only over inliers
    # print(error.shape)
    error = np.reshape(error, (error.shape[0], 2))
    print(np.sqrt(np.mean(error[:, 0] ** 2  + error[:, 1] **2)))
    rmse = np.sqrt(np.mean(error[:, 0] ** 2 + error[:, 1] ** 2))
    print('Re-projection error before LM refinement (RMSE) in px:', str(rmse))
    
    # Refine estimate using LM
    if hasattr(cv2, 'solvePnPRefineLM') and len(inliers) >= 3:
        rotation_vector, translation_vector = cv2.solvePnPRefineLM(
            points3D[inliers], points2D[inliers], camera_matrix, dist_coeffs, rotation_vector, translation_vector)
        points2D_reproj = cv2.projectPoints(points3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs)[0].squeeze(1)
        assert points2D_reproj.shape == points2D.shape
        error = (points2D_reproj - points2D)[inliers]  # Compute error only over inliers
        error = np.reshape(error, (error.shape[0], 2))
        rmse = np.sqrt(np.mean(error[:, 0] ** 2 + error[:, 1] ** 2))
        print('Re-projection error after LM refinement (RMSE) in px:', str(rmse))
    else:
        print('Skipping LM refinement')

    # Convert rotation vector to matrix
    rotation_matrix = cv2.Rodrigues(rotation_vector)[0]
    euler = mat2euler(rotation_matrix)
    
    # Save extrinsics
    np.savez(os.path.join(PKG_PATH, CALIB_PATH, 'extrinsics.npz'), euler=euler, R=rotation_matrix, T=translation_vector.T)
    
    # Display results
    print('Euler angles (RPY):', euler)
    print('Rotation Matrix:', rotation_matrix)
    print('Translation Offsets:', translation_vector.T)


'''
Projects the point cloud on to the image plane using the extrinsics

Inputs:
    img_msg - [sensor_msgs/Image] - ROS sensor image message
    velodyne - [sensor_msgs/PointCloud2] - ROS velodyne PCL2 message
    image_pub - [sensor_msgs/Image] - ROS image publisher

Outputs:
    Projected points published on /sensors/camera/camera_lidar topic
'''
# def project_point_cloud(velodyne, img_msg, image_pub):
def project_point_cloud(velodyne, img_msg, image_pub):
    # Load extrinsic parameters
    # extrinsics = np.load(os.path.join(CALIB_PATH, 'extrinsics.npz'))
    # R = np.array([[0.02491483, -0.99964749, -0.00917287],
    #    [ 0.16635332,  0.01319356, -0.98597794],
    #    [ 0.9857514,   0.02303953,  0.1666234 ]], dtype = np.float32)
    # T = np.array([-0.12355435,  0.1643491,   0.30427816], dtype = np.float32)
    R = np.array([[-0.00650686, -0.99947944, -0.03159908],
    [-0.13910267,  0.03219721, -0.98975441],
    [ 0.99025659, -0.00204468, -0.13923976]], dtype = np.float32)
    T = np.array([-0.05026612,  0.18018712,  0.25274406], dtype=np.float32)

    # Read image using CV bridge
    try:
        img = CV_BRIDGE.imgmsg_to_cv2(img_msg, 'bgr8')
    except CvBridgeError as e: 
        print(e)
        return

    # Transform the point cloud
    try:
        transform = TF_BUFFER.lookup_transform('world', 'velodyne', rclpy.time.Time())
        velodyne = do_transform_cloud(velodyne, transform)
    except tf2_ros.LookupException:
        pass

    # Extract points from message
    points3D = ros2_numpy.point_cloud2.pointcloud2_to_array(velodyne)
    points3D = np.asarray(points3D.tolist())
    points3D = points3D[np.random.choice(points3D.shape[0], 500, replace = False)]
    points3D = np.reshape(points3D, (points3D.shape[0] * points3D.shape[1], -1))
    
    # Group all beams together and pick the first 4 columns for X, Y, Z, intensity.
    if OUSTER_LIDAR: points3D = points3D.reshape(-1, 9)[:, :4]

    # Apply extrinsic transformation using rotation matrix and translation vector
    points3D_homogeneous = np.hstack((points3D[:, :3], np.ones((points3D.shape[0], 1))))
    points3D_transformed = (R @ points3D_homogeneous[:, :3].T).T + T

    # Filter points in front of the camera
    inrange = np.where((points3D_transformed[:, 2] > 0) &
                       (points3D_transformed[:, 2] < 6) &
                       (np.abs(points3D_transformed[:, 0]) < 6) &
                       (np.abs(points3D_transformed[:, 1]) < 6))
    max_intensity = np.max(points3D[:, -1])
    points3D_transformed = points3D_transformed[inrange[0]]

    # Color map for the points
    cmap = matplotlib.cm.get_cmap('jet')
    colors = cmap(points3D[:, -1] / max_intensity) * 255

    # Project to 2D and filter points within image boundaries
    points2D = [CAMERA_MODEL.project3dToPixel(point) for point in points3D_transformed]
    points2D = np.asarray(points2D)
    inrange = np.where((points2D[:, 0] >= 0) &
                       (points2D[:, 1] >= 0) &
                       (points2D[:, 0] < img.shape[1]) &
                       (points2D[:, 1] < img.shape[0]))
    points2D = points2D[inrange[0]].round().astype('int')

    # Draw the projected 2D points
    for i in range(len(points2D)):
        cv2.circle(img, tuple(points2D[i]), 2, tuple(colors[i]), -1)

    # Publish the projected points image
    try:
        image_pub.publish(CV_BRIDGE.cv2_to_imgmsg(img, "bgr8"))
    except CvBridgeError as e: 
        print(e)




class CalibrateCameraLidar(Node):
    def __init__(self, camera_info, image_color, velodyne_points, camera_lidar=None, project_mode=False):
        super().__init__('calibrate_camera_lidar')
        self.get_logger().info('Current PID: [%d]' % os.getpid())
        self.get_logger().info('Projection mode: %s' % project_mode)
        self.get_logger().info('CameraInfo topic: %s' % camera_info)
        self.get_logger().info('Image topic: %s' % image_color)
        self.get_logger().info('PointCloud2 topic: %s' % velodyne_points)
        self.get_logger().info('Output topic: %s' % camera_lidar)

        self.project_mode = project_mode
        self.camera_lidar = camera_lidar

        # Subscribe to topics
        self.info_sub = self.create_subscription(CameraInfo, camera_info, self.camera_info_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(Image, image_color, self.image_callback, qos_profile_sensor_data)
        self.velodyne_sub = self.create_subscription(PointCloud2, velodyne_points, self.velodyne_callback, qos_profile_sensor_data)

        # Publish output topic
        self.image_pub = self.create_publisher(Image, camera_lidar, 5) if camera_lidar else None

        self.camera_info_msg = None
        self.image_msg = None
        self.velodyne_msg = None

    def camera_info_callback(self, msg):
        self.camera_info_msg = msg
        self.process()

    def image_callback(self, msg):
        self.image_msg = msg
        self.process()

    def velodyne_callback(self, msg):
        self.velodyne_msg = msg
        self.process()

    def process(self):
        global FIRST_TIME, PAUSE, TF_BUFFER, TF_LISTENER, CAMERA_MODEL
        # print("camera msg")
        # print(self.camera_info_msg)
        # print("===========================")
        # print("image msg")
        # print(self.image_msg)
        # print("===========================")
        # print("velodyne msg")
        # print(self.velodyne_msg)
        self.camera_info_msg = CameraInfo()
        # print(self.camera_info_msg)
        # print(self.project_mode)
        # print(self.image_msg)
        if self.camera_info_msg and self.image_msg and self.velodyne_msg:
            if FIRST_TIME:
                print("here1")
                FIRST_TIME = False

                # Setup camera model
                self.get_logger().info('Setting up camera model')
                # self.get_logger().info(self.camera_info_msg)
                camera_info_msg = CameraInfo()

                camera_info_msg.header.frame_id = 'camera_frame'
                camera_info_msg.height = 427
                camera_info_msg.width = 640
                camera_info_msg.distortion_model = 'plumb_bob'
                camera_info_msg.d = [0.0, 0.0, 0.0, 0.0]
                camera_info_msg.k = [493.53473414, -3.42710262, 286.5538899, 0.0, 490.94207904, 225.36415318, 0.0, 0.0, 1.0]
                camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                camera_info_msg.p = [493.53473414, -3.42710262, 286.5538899, 0.0, 0.0, 490.94207904, 225.36415318, 0.0, 0.0, 0.0, 1.0, 0.0]
                self.camera_info_msg = camera_info_msg
                print(self.camera_info_msg)
                CAMERA_MODEL.fromCameraInfo(self.camera_info_msg)

                # TF listener
                self.get_logger().info('Setting up static transform listener')
                TF_BUFFER = tf2_ros.Buffer()
                TF_LISTENER = tf2_ros.TransformListener(TF_BUFFER, self)

            # Projection/display mode
            if self.project_mode:
                print("here2")
                project_point_cloud(self.velodyne_msg, self.image_msg, self.image_pub)

            # Calibration mode
            elif PAUSE:
                # Create GUI processes
                now = self.get_clock().now().seconds_nanoseconds()[0]
                img_p = multiprocessing.Process(target=extract_points_2D, args=[self.image_msg, now])
                pcl_p = multiprocessing.Process(target=extract_points_3D, args=[self.velodyne_msg, now])
                img_p.start(); pcl_p.start()
                img_p.join(); pcl_p.join()

                # Calibrate for existing corresponding points
                calibrate()

                # Resume listener
                with KEY_LOCK: PAUSE = False
                start_keyboard_handler()


def main(args=None):
    rclpy.init(args=args)
    node = None

    if sys.argv[1] == '--calibrate':
        camera_info = '/sensors/camera/camera_info'
        image_color = '/sensors/camera/image_color'
        velodyne_points = '/sensors/velodyne_points'
        camera_lidar = None
        project_mode = False
    elif sys.argv[1] == '--new_calib':
        camera_info = '/sensors/camera/camera_info'
        image_color = '/sensors/camera/image_color'
        velodyne_points = '/rslidar_points'
        camera_lidar = '/sensors/camera/camera_lidar'
        project_mode = True
    else:
        camera_info = '/sensors/camera/camera_info'
        image_color = '/sensors/camera/image_color'
        velodyne_points = '/sensors/velodyne_points'
        camera_lidar = '/sensors/camera/camera_lidar'
        project_mode = True

    node = CalibrateCameraLidar(camera_info, image_color, velodyne_points, camera_lidar, project_mode)

    if not project_mode:
        start_keyboard_handler()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

# camera_info_msg = CameraInfo()

# camera_info_msg.header.frame_id = 'camera_frame'
# camera_info_msg.height = 427
# camera_info_msg.width = 640
# camera_info_msg.distortion_model = 'plumb_bob'
# camera_info_msg.d = [0.0, 0.0, 0.0, 0.0]
# camera_info_msg.k = [493.53473414, -3.42710262, 286.5538899, 0.0, 490.94207904, 225.36415318, 0.0, 0.0, 1.0]
# camera_info_msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
# camera_info_msg.p = [493.53473414, -3.42710262, 286.5538899, 0.0, 0.0, 490.94207904, 225.36415318, 0.0, 0.0, 0.0, 1.0, 0.0]

# # Rotation and translation matrix for further use (not part of CameraInfo)
# rotation_translation = np.array([
#     [0.998090531, 0.0345642072, -0.0511918717, -36.6595165],
#     [-0.0315545493, 0.997789962, 0.0584765112, -150.912793],
#     [0.0530999300, -0.0567495156, 0.996975371, 411.776680]
# ])

# CAMERA_MODEL.fromCameraInfo(camera_info_msg)


# points3D = np.array([
#     # [0.93414306640625, 0.11254748702049255, -0.08794709],
#     [1.7842121124267578, -1.319765567779541, -0.26463285],
#     [1.82595956325531, 2.7348077297210693, -0.10391533],
#     [0.7886460423469543, 0.6838697195053101, -0.11339932],
#     [1.5493313074111938, -1.4641057252883911, -0.2387288],
#     [0.7646322250366211, 0.6434240937232971, -0.106620975],
#     [0.9087061882019043, -0.11157508939504623, -0.09057525],
#     [0.8688499331474304, -0.12396509945392609, -0.113830924],
#     # [0.932473361492157, 0.518581748008728, -0.08041346]
#     ]) 
# points2D= np.array([
#     # [118.20129870129873, 218.20129870129864],
#     [136.3831168831169, 218.20129870129864],
#     [118.20129870129873, 236.38311688311683],
#     [137.68181818181822, 236.38311688311683],
#     [118.20129870129873, 218.20129870129864],
#     [241.57792207792212, 218.20129870129864],
#     [248.07142857142864, 345.47402597402595],
#     [116.90259740259742, 346.77272727272725],
#     # [118.20129870129873, 219.49999999999994]
# ])

# points3D = np.array([
#     [2.7026, -0.4634, -0.6586],
#     [2.4280, -0.5397, -0.5816],
#     [2.4389, -0.5545, -0.7699],
#     [2.5168, -0.4274, -0.8003],
#     [2.6520, 0.3321, -0.2277],
#     [2.6559, 0.2719, -0.2302],
#     [2.6550, 0.2751, -0.4171],
#     [2.6505, 0.3355, -0.4118],
#     [2.4419, -0.2475, -0.2210],
#     [2.8850, -0.4420, -0.2921],
#     [2.4202, -0.3598, -0.4008],
#     [2.4268, -0.2479, -0.3973],
#     [2.4265, -0.3629, -0.3395],
#     [2.8891, -0.6518, -0.4303],
#     [2.8889, -0.6528, -0.6252],
#     [2.8988, -0.4596, -0.6353]
# ], dtype=np.float32)
# points2D = np.array([[270.15235177049493, 343.6567444921874],
#     [290.9255703074273, 343.6567444921874],
#     [294.04155308796715, 367.0266153462363],
#     [271.71034316076486, 367.0266153462363],
#     [119.50000000000003, 196.12337662337654],
#     [137.68181818181822, 196.12337662337654],
#     [140.27922077922082, 257.1623376623376],
#     [116.90259740259742, 255.86363636363632],
#     [138.98051948051952, 220.79870129870125],
#     [183.13636363636365, 220.79870129870125],
#     [181.8376623376624, 285.7337662337662],
#     [137.68181818181822, 284.4350649350649],
#     [203.91558441558445, 285.7337662337662],
#     [268.8506493506494, 287.0324675324675],
#     [270.1493506493507, 351.9675324675324],
#     [203.91558441558445, 351.9675324675324]
# ], dtype=np.float32)


# retval, rvec, tvec, inliers = cv2.solvePnPRansac(
#     points3D,
#     points2D,
#     np.reshape(camera_info_msg.k, (3, 3)),
#     np.zeros(4)
# )

# # Check if the function succeeded
# if retval:
#     print("Rotation Vector:\n", rvec)
#     print("Translation Vector:\n", tvec)
#     print("Inliers:\n", inliers)
# else:
#     print("solvePnPRansac failed.")

# print(calibrate(
#     points2D= points2D, points3D =points3D))


# import cv2
# import numpy as np

# # 3D points in the object coordinate space (e.g., coordinates of the corners of a cube)
# object_points = np.array([
#     [0, 0, 0],
#     [1, 0, 0],
#     [1, 1, 0],
#     [0, 1, 0],
#     [0, 0, 1],
#     [1, 0, 1],
#     [1, 1, 1],
#     [0, 1, 1]
# ], dtype=np.float32)

# # Corresponding 2D points in the image plane (e.g., detected in an image)
# image_points = np.array([
#     [100, 100],
#     [200, 100],
#     [200, 200],
#     [100, 200],
#     [120, 120],
#     [220, 120],
#     [220, 220],
#     [120, 220]
# ], dtype=np.float32)

# # Camera matrix (intrinsic parameters)
# # Assuming a focal length of 800 and the principal point at (320, 240)
# camera_matrix = np.array([
#     [800, 0, 320],
#     [0, 800, 240],
#     [0, 0, 1]
# ], dtype=np.float32)

# # Distortion coefficients (assuming no distortion for simplicity)
# dist_coeffs = np.zeros(4, dtype=np.float32)

# # Call solvePnPRansac
# retval, rvec, tvec, inliers = cv2.solvePnPRansac(
#     object_points,
#     image_points,
#     camera_matrix,
#     dist_coeffs
# )

# # Check if the function succeeded
# if retval:
#     print("Rotation Vector:\n", rvec)
#     print("Translation Vector:\n", tvec)
#     print("Inliers:\n", inliers)
# else:
#     print("solvePnPRansac failed.")
