import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Function to plot the cube in 3D space
def plot_cube_on_camera_simulation(ax, position=[0, 0, 0], axes=[5, 5, 5], alpha=0.9): 
    # Create Data
    data = np.ones(axes, dtype=bool)

    # Control color
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

# Camera parameters
f_x = 800   
f_y = 800   
c_x = 320   
c_y = 240   

K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])

# Cube position
cube_position = np.array([0, 0, 30])  # 3D position of the cube
cube_size = [5, 5, 5]  # Size of the cube

# Field of view (FOV) pyramid vertices
fov_angle = np.radians(60)  # Field of view angle
fov_depth = 50  # Depth of the field of view
fov_vertices = np.array([
    [0, 0, 0],  # Camera position (the tip of the pyramid)
    [fov_depth * np.tan(fov_angle / 2), fov_depth * np.tan(fov_angle / 2), fov_depth],   # Top-right
    [-fov_depth * np.tan(fov_angle / 2), fov_depth * np.tan(fov_angle / 2), fov_depth],  # Top-left
    [-fov_depth * np.tan(fov_angle / 2), -fov_depth * np.tan(fov_angle / 2), fov_depth], # Bottom-left
    [fov_depth * np.tan(fov_angle / 2), -fov_depth * np.tan(fov_angle / 2), fov_depth]   # Bottom-right
])

# Rotation matrix for rotating about the Y axis by 45 degrees (right tilt)
theta_y = np.radians(45)
rotation_matrix_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                              [0, 1, 0],
                              [-np.sin(theta_y), 0, np.cos(theta_y)]])

# Apply the rotation to the FOV vertices
fov_vertices_rotated = np.dot(fov_vertices, rotation_matrix_y.T)

# Project cube vertices onto the image plane
def project_points(K, points_3D):
    """ Projects 3D points onto 2D image plane using intrinsic matrix K. """
    points_3D = points_3D.astype(np.float32)  # Ensure the points are of float type
    points_2D = np.dot(K, points_3D.T).T  # Apply camera matrix
    points_2D /= points_2D[:, 2][:, np.newaxis]  # Normalize by depth (z)
    return points_2D[:, :2]  # Return only (x, y) 2D coordinates

# Cube vertices in 3D
cube_vertices = np.array([[0, 0, 30],   # Bottom-front-left
                          [5, 0, 30],   # Bottom-front-right
                          [0, 5, 30],   # Bottom-back-left
                          [5, 5, 30],   # Bottom-back-right
                          [0, 0, 35],   # Top-front-left
                          [5, 0, 35],   # Top-front-right
                          [0, 5, 35],   # Top-back-left
                          [5, 5, 35]])  # Top-back-right

# Function to check if a point is inside the FOV pyramid (by checking the bounding box)
def is_in_fov(point, fov_vertices):
    # This assumes that the FOV forms a bounding box
    min_x = np.min(fov_vertices[:, 0])
    max_x = np.max(fov_vertices[:, 0])
    min_y = np.min(fov_vertices[:, 1])
    max_y = np.max(fov_vertices[:, 1])
    min_z = np.min(fov_vertices[:, 2])
    max_z = np.max(fov_vertices[:, 2])

    return (min_x <= point[0] <= max_x) and (min_y <= point[1] <= max_y) and (min_z <= point[2] <= max_z)

# Check if any cube vertices are within the FOV
cube_in_fov = any(is_in_fov(vertex, fov_vertices_rotated) for vertex in cube_vertices)

# Plotting
fig = plt.figure(figsize=(10, 5))

# 3D plot to visualize the setup
ax3d = fig.add_subplot(121, projection='3d')

# Plot the cube in 3D
plot_cube_on_camera_simulation(ax3d, position=cube_position)

# Plot the rotated FOV pyramid in 3D (with all sides)
vertices_faces = [[fov_vertices_rotated[0], fov_vertices_rotated[1], fov_vertices_rotated[2]],
                  [fov_vertices_rotated[0], fov_vertices_rotated[2], fov_vertices_rotated[3]],
                  [fov_vertices_rotated[0], fov_vertices_rotated[3], fov_vertices_rotated[4]],
                  [fov_vertices_rotated[0], fov_vertices_rotated[4], fov_vertices_rotated[1]],
                  [fov_vertices_rotated[1], fov_vertices_rotated[2], fov_vertices_rotated[3], fov_vertices_rotated[4]]]

# Plot all four sides of the pyramid
ax3d.add_collection3d(Poly3DCollection(vertices_faces, color='green', alpha=0.3, linewidths=1, edgecolors='r'))

# Set labels and limits for the 3D view
ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')
ax3d.set_xlim(-20, 30)
ax3d.set_ylim(-20, 30)
ax3d.set_zlim(0, 60)

# 2D Camera View (Simulated image)
ax2d = fig.add_subplot(122)

# Plot the outer FOV rectangle (camera image size)
ax2d.plot([0, 640, 640, 0, 0], [0, 0, 480, 480, 0], 'g-', label='FOV')

if cube_in_fov:
    # If the cube is in the FOV, plot the projection
    cube_vertices_2D = project_points(K, cube_vertices)
    
    # Plot the projected rectangle (the cube as seen from the camera)
    ax2d.fill([cube_vertices_2D[0, 0], cube_vertices_2D[1, 0], cube_vertices_2D[3, 0], cube_vertices_2D[2, 0], cube_vertices_2D[0, 0]], 
              [cube_vertices_2D[0, 1], cube_vertices_2D[1, 1], cube_vertices_2D[3, 1], cube_vertices_2D[2, 1], cube_vertices_2D[0, 1]],
              'r-', alpha=0.5, label='Cube Projection')

else:
    # If the cube is outside the FOV, show no cube projection
    ax2d.text(320, 240, 'Cube is outside the FOV', color='red', fontsize=12, ha='center')

# Labels and settings for the 2D plot
ax2d.set_xlabel('Image X')
ax2d.set_ylabel('Image Y')
ax2d.set_xlim(0, 640)
ax2d.set_ylim(480, 0)  # In image coordinates, (0,0) is top-left
ax2d.set_title('2D Camera View (Projection)')
plt.legend()

# Show plot
plt.show()