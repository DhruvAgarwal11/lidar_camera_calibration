import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.cm
import rclpy
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import ros2_numpy
from scipy.spatial.transform import Rotation as R

def main():
    # Initialize CV Bridge
    CV_BRIDGE = CvBridge()

    # Initialize TF Buffer and Listener
    TF_BUFFER = Buffer()
    TF_LISTENER = TransformListener(TF_BUFFER)

    # Provided calibration data
    euler_angles = np.array([2.338816443976044, -0.574322815165147, -0.904944999401531], dtype=np.float32)
    R_matrix = np.array([[ 0.51862066, -0.78770242, -0.33250189],
                        [-0.66022288, -0.12185,    -0.74111964],
                        [ 0.54326638,  0.60388531, -0.58325309]], dtype=np.float32)
    T_vector = np.array([-2.08619504,  1.61377784,  0.77802758], dtype=np.float32)

    def project_point_cloud(velodyne, img_msg, image_pub):
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
        
        # Group all beams together and pick the first 4 columns for X, Y, Z, intensity.
        if OUSTER_LIDAR:
            points3D = points3D.reshape(-1, 9)[:, :4]
        
        # Apply extrinsic transformation using rotation matrix and translation vector
        points3D_homogeneous = np.hstack((points3D[:, :3], np.ones((points3D.shape[0], 1))))
        points3D_transformed = (R_matrix @ points3D_homogeneous[:, :3].T).T + T_vector

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

if __name__ == '__main__':
    main()