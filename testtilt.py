import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cube_on_camera_simulation(ax, position=[0, 0, 0], axes=[5, 5, 5], alpha=0.9): 
    # Create Data
    data = np.ones(axes, dtype=np.bool)

    # Control colour
    colors = np.empty(axes + [4], dtype=np.float32)

    # Assigning colors for each voxel
    colors[:,:,:] = [1, 1, 1, alpha]  # Default white color
    colors[0] = [1, 0, 0, alpha]  # Red for the first slice
    colors[1] = [0, 1, 0, alpha]  # Green for the second slice
    colors[2] = [0, 0, 1, alpha]  # Blue for the third slice
    colors[3] = [1, 1, 0, alpha]  # Yellow for the fourth slice
    colors[4] = [1, 1, 1, alpha]  # White for the fifth slice

    # Define voxel positions
    x, y, z = np.indices(np.array(axes) + 1)

    # Shift the cube by adding the position offset
    x = x + position[0]
    y = y + position[1]
    z = z + position[2]

    # Plot voxels (cube) in the same figure as the FOV
    ax.voxels(x, y, z, data, facecolors=colors, edgecolors='grey')


# Define the camera parameters and other elements (kept unchanged)
f_x = 800   
f_y = 800   
c_x = 320   
c_y = 240   

K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])

point_3D_inertial = np.array([10, 5, 30])

def get_pan_tilt_matrix(pan, tilt):
    pan_matrix = np.array([[np.cos(pan), -np.sin(pan), 0],
                           [np.sin(pan), np.cos(pan), 0],
                           [0, 0, 1]])

    tilt_matrix = np.array([[1, 0, 0],
                            [0, np.cos(tilt), -np.sin(tilt)],
                            [0, np.sin(tilt), np.cos(tilt)]])
    
    return np.dot(tilt_matrix, pan_matrix)

pan_angle = np.radians(30)
tilt_angle = np.radians(15)

R_pan_tilt = get_pan_tilt_matrix(pan_angle, tilt_angle)

point_3D_body = np.dot(R_pan_tilt, point_3D_inertial)

point_2D = np.dot(K, point_3D_body / point_3D_body[2])

q1 = np.array([0, 0])
q2 = np.array([640, 0])
q3 = np.array([640, 480])
q4 = np.array([0, 480])

fov_angle = np.radians(60)
fov_depth = 50
fov_vertices = np.array([
    [0, 0, 0],  # Apex of the pyramid (camera origin)
    [fov_depth * np.tan(fov_angle/2), fov_depth * np.tan(fov_angle/2), fov_depth],  # Front-right vertex
    [-fov_depth * np.tan(fov_angle/2), fov_depth * np.tan(fov_angle/2), fov_depth],  # Front-left vertex
    [-fov_depth * np.tan(fov_angle/2), -fov_depth * np.tan(fov_angle/2), fov_depth],  # Back-left vertex
    [fov_depth * np.tan(fov_angle/2), -fov_depth * np.tan(fov_angle/2), fov_depth]   # Back-right vertex
])

# Rotation matrix for 45 degrees around the X-axis
theta = np.radians(45)
rotation_matrix_x = np.array([[1, 0, 0],
                              [0, np.cos(theta), -np.sin(theta)],
                              [0, np.sin(theta), np.cos(theta)]])

# Apply rotation to FOV vertices
fov_vertices = np.dot(fov_vertices, rotation_matrix_x.T)

# Plotting the simulation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Call the modified cube plotting function and shift the cube
plot_cube_on_camera_simulation(ax, position=[0,0,30])  # CHANGE THE CUBE LOCATION

# Define the triangular faces of the pyramid (FOV)
faces = [[0, 1, 2],  # Front face
         [0, 2, 3],  # Left face
         [0, 3, 4],  # Back face
         [0, 4, 1]]  # Right face

# Plot each face of the pyramid
for face in faces:
    ax.plot_trisurf(fov_vertices[face, 0], fov_vertices[face, 1], fov_vertices[face, 2], alpha=0.5, color='g')

# Set labels and limits
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim(-20, 30)
ax.set_ylim(-20, 30)
ax.set_zlim(0, 60)

# Show plot
plt.legend()
plt.title('Pinhole Camera Model - Moved Cube & Rotated FOV with Full Pyramid')
plt.show()

print(f"Projected 2D point: {point_2D[:2]}")
