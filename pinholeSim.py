import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the camera parameters
f_x = 800   # Focal length in x
f_y = 800   # Focal length in y
c_x = 320   # Optical center in x
c_y = 240   # Optical center in y

# Pinhole camera projection matrix
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])

# Define a 3D point in the inertial frame
point_3D_inertial = np.array([10, 5, 30])

# Pan and tilt matrices
def get_pan_tilt_matrix(pan, tilt):
    # Pan rotation matrix
    pan_matrix = np.array([[np.cos(pan), -np.sin(pan), 0],
                           [np.sin(pan), np.cos(pan), 0],
                           [0, 0, 1]])

    # Tilt rotation matrix
    tilt_matrix = np.array([[1, 0, 0],
                            [0, np.cos(tilt), -np.sin(tilt)],
                            [0, np.sin(tilt), np.cos(tilt)]])
    
    # Combined pan-tilt rotation
    return np.dot(tilt_matrix, pan_matrix)

# Define the pan and tilt angles
pan_angle = np.radians(30)  # 30 degrees pan
tilt_angle = np.radians(15) # 15 degrees tilt

# Get the rotation matrix for pan and tilt
R_pan_tilt = get_pan_tilt_matrix(pan_angle, tilt_angle)

# Transform the 3D point in the inertial frame to the body frame
point_3D_body = np.dot(R_pan_tilt, point_3D_inertial)

# Project the 3D point to the 2D image plane using the pinhole camera model
point_2D = np.dot(K, point_3D_body / point_3D_body[2])

# Image sensor bounds defined by vertices q1, q2, q3, q4
q1 = np.array([0, 0])
q2 = np.array([640, 0])
q3 = np.array([640, 480])
q4 = np.array([0, 480])

# Define FOV vertices (using a simple 60-degree FOV as an example)
fov_angle = np.radians(60)  # 60 degrees FOV
fov_depth = 50  # FOV depth in arbitrary units
fov_vertices = [
    [0, 0, 0],
    [fov_depth * np.tan(fov_angle/2), fov_depth * np.tan(fov_angle/2), fov_depth],
    [-fov_depth * np.tan(fov_angle/2), fov_depth * np.tan(fov_angle/2), fov_depth],
    [-fov_depth * np.tan(fov_angle/2), -fov_depth * np.tan(fov_angle/2), fov_depth],
    [fov_depth * np.tan(fov_angle/2), -fov_depth * np.tan(fov_angle/2), fov_depth]
]

# Plotting the simulation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3D point
ax.scatter(point_3D_inertial[0], point_3D_inertial[1], point_3D_inertial[2], c='r', label='3D Target')

# Plot the FOV vertices
fov_vertices = np.array(fov_vertices)
ax.plot_trisurf(fov_vertices[:,0], fov_vertices[:,1], fov_vertices[:,2], alpha=0.2, color='g')

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.set_zlim(0, 50)

# Show plot
plt.legend()
plt.title('Pinhole Camera Model - 3D Point Projection & FOV')
plt.show()

print(f"Projected 2D point: {point_2D[:2]}")
