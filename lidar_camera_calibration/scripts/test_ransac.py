import cv2
import numpy as np

# Camera matrix from your camera info message
camera_matrix = np.array([
    493.53473414, -3.42710262, 286.5538899,
    0.0, 490.94207904, 225.36415318,
    0.0, 0.0, 1.0
]).reshape(3, 3)

# Distortion coefficients (ensure it has the correct size)
dist_coeffs = np.zeros(5)  # Adjusted to have 5 coefficients

# 3D points in the object coordinate space
points3D = np.array([
    [0.93414306640625, 0.11254748702049255, -0.08794709],
    [1.7842121124267578, -1.319765567779541, -0.26463285],
    [1.82595956325531, 2.7348077297210693, -0.10391533],
    [0.7886460423469543, 0.6838697195053101, -0.11339932],
    [0.7646322250366211, 0.6434240937232971, -0.106620975],
    [0.9087061882019043, -0.11157508939504623, -0.09057525],
    [0.8688499331474304, -0.12396509945392609, -0.113830924]
])

# Corresponding 2D points in the image plane
points2D = np.array([
    [118.20129870129873, 218.20129870129864],
    [136.3831168831169, 218.20129870129864],
    [118.20129870129873, 236.38311688311683],
    [137.68181818181822, 236.38311688311683],
    [241.57792207792212, 218.20129870129864],
    [248.07142857142864, 345.47402597402595],
    [116.90259740259742, 346.77272727272725]
])

# Ensure you have enough points and that they match correctly
assert points3D.shape[0] == points2D.shape[0] and points3D.shape[0] >= 6, "Not enough points or mismatch between 3D and 2D points"

# Call solvePnPRansac with RANSAC parameters
retval, rvec, tvec, inliers = cv2.solvePnPRansac(
    points3D,
    points2D,
    camera_matrix,
    dist_coeffs,
    iterationsCount=100,  # Adjust if needed
    reprojectionError=70.0,  # Adjust if needed
    confidence=.99,
    flags=cv2.SOLVEPNP_ITERATIVE
)

# Check if the function succeeded
if retval:
    print("solvePnPRansac succeeded.")
    print("Rotation Vector:\n", rvec)
    print("Translation Vector:\n", tvec)
    print("Inliers:\n", inliers)
else:
    print("solvePnPRansac failed.")
