import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Cube plotting function
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

# Cube position (manually update this position)
cube_position = np.array([10, 0, 30])  # Example position of the cube
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

# Function to check if the cube is within the FOV
def is_within_fov(cube_position):
    """ Check if the cube is within the field of view (FOV) pyramid. """
    x, y, z = cube_position
    if z <= 0 or z >= fov_depth:
        return False
    
    # Calculate the horizontal distance at the cube's z position
    horizontal_limit = (fov_depth - z) * np.tan(fov_angle / 2)

    # Check if the cube's x and y positions are within the limits
    return -horizontal_limit <= x <= horizontal_limit and -horizontal_limit <= y <= horizontal_limit

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

# Function to update the plot
def update_plot(cube_position):
    # Update cube vertices based on the new position
    cube_vertices[:, 0] += cube_position[0]  # Update x position
    cube_vertices[:, 1] += cube_position[1]  # Update y position
    cube_vertices[:, 2] += cube_position[2]  # Update z position

    # Check if the cube is within the FOV
    cube_in_fov = is_within_fov(cube_position)

    # 2D projection of the cube vertices (only if it's in FOV)
    if cube_in_fov:
        cube_vertices_2D = project_points(K, cube_vertices)
    else:
        cube_vertices_2D = None  # Set to None if not in FOV

    # Plotting
    fig = plt.figure(figsize=(10, 5))

    # 3D plot to visualize the setup
    ax3d = fig.add_subplot(121, projection='3d')

    # Plot the cube in 3D
    plot_cube_on_camera_simulation(ax3d, position=cube_position)

    # Plot the FOV pyramid in 3D
    ax3d.plot_trisurf(fov_vertices[:, 0], fov_vertices[:, 1], fov_vertices[:, 2], color='green', alpha=0.3)

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

    if cube_in_fov and cube_vertices_2D is not None:
        # Plot the projected rectangle (the cube as seen from the camera)
        ax2d.fill([cube_vertices_2D[0, 0], cube_vertices_2D[1, 0], cube_vertices_2D[3, 0], cube_vertices_2D[2, 0], cube_vertices_2D[0, 0]], 
                  [cube_vertices_2D[0, 1], cube_vertices_2D[1, 1], cube_vertices_2D[3, 1], cube_vertices_2D[2, 1], cube_vertices_2D[0, 1]], 
                  'r-', alpha=0.5, label='Cube Projection')
    else:
        # If the cube is not in the FOV, show blank plot with a message
        ax2d.set_xlim(0, 640)
        ax2d.set_ylim(480, 0)  # In image coordinates, (0,0) is top-left
        ax2d.set_title('2D Camera View (No Projection)')
        ax2d.text(320, 240, 'No Projection', horizontalalignment='center', verticalalignment='center', fontsize=20, color='red')

    # Labels and settings for the 2D plot
    ax2d.set_xlabel('Image X')
    ax2d.set_ylabel('Image Y')
    ax2d.set_title('2D Camera View')
    plt.legend()
    
    # Show plot
    plt.show()

# Run the initial plot with the cube at the specified position
update_plot(cube_position)
