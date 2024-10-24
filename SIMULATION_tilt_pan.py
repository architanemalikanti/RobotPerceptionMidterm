import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#RESOURCES:
#For the cube: https://www.geeksforgeeks.org/how-to-draw-3d-cube-using-matplotlib-in-python/
#For the tilt matrix: Pinhole Camera Model and Field-of-view (FOV) paper
#For the template to rotate the pyramid, and map the 2D projection from the 3D model: ChatGPT

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

#creates camera frame
K = np.array([[f_x, 0, c_x],
              [0, f_y, c_y],
              [0, 0, 1]])

# Cube position (manually update this position)
cube_position = np.array([10, 0, 30])  # CHANGE THIS TO EDIT CUBE POSITION
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

# Rotation matrices for tilt (x-axis) and pan (y-axis)
#
def rotation_matrix_x(angle_deg):
    """ Rotation matrix for tilt (rotation around x-axis). """
    angle_rad = np.radians(angle_deg)
    return np.array([[1, 0, 0],
                     [0, np.cos(angle_rad), np.sin(angle_rad)],
                     [0, -np.sin(angle_rad), np.cos(angle_rad)]])

def rotation_matrix_y(angle_deg):
    """ Rotation matrix for pan (rotation around y-axis). """
    angle_rad = np.radians(angle_deg)
    return np.array([[np.cos(angle_rad), 0, np.sin(angle_rad)],
                     [0, 1, 0],
                     [-np.sin(angle_rad), 0, np.cos(angle_rad)]])

    #angle_rad = np.radians(angle_deg)
    #return np.array([[np.cos(angle_rad), np.sin(angle_rad), 0],
                   #  [-np.sin(angle_rad), np.cos(angle_rad), 0],
                    # [0, 0, 1]])

# Function to rotate FOV and cube
def rotate_points(points_3D, tilt, pan):
    """ Rotate the 3D points by tilt (x-axis) and pan (y-axis). """
    # Apply tilt (rotation around x-axis)
    points_rotated = np.dot(points_3D, rotation_matrix_x(tilt).T)
    
    # Apply pan (rotation around y-axis)
    points_rotated = np.dot(points_rotated, rotation_matrix_y(pan).T)
    
    return points_rotated

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

# Function to add the pinhole obstacle
def add_pinhole_obstacle(ax, frame_width=4, frame_height=3, hole_radius=0.5):
    """ Add a 3D rectangular box with a circular pinhole to the plot. """
    # Define the box vertices (8 corners)
    box_x = [0, box_size[0], box_size[0], 0, 0, box_size[0], box_size[0], 0]
    box_y = [0, 0, box_size[1], box_size[1], 0, 0, box_size[1], box_size[1]]
    box_z = [0, 0, 0, 0, box_size[2], box_size[2], box_size[2], box_size[2]]

    # Define the edges connecting the vertices
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]

    # Draw the rectangular box
    for edge in edges:
        ax.plot([box_x[edge[0]], box_x[edge[1]]],
                [box_y[edge[0]], box_y[edge[1]]],
                [box_z[edge[0]], box_z[edge[1]]], color='black')

    # Draw the circular pinhole on the front face of the box
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = hole_radius * np.cos(theta) + box_size[0] / 2
    circle_y = hole_radius * np.sin(theta) + box_size[1] / 2
    circle_z = np.zeros_like(theta)

    ax.plot(circle_x, circle_y, circle_z, color='red', linewidth=2)


# Assume isObstaclePresent is a global or passed variable
isObstaclePresent = True  # Set this flag to toggle the obstacle



# Function to update the plot
def update_plot(cube_position, tilt=0, pan=0):
    # Rotate the cube vertices based on the new position
    cube_vertices_rotated = rotate_points(cube_vertices + cube_position, tilt, pan)

    # Check if the cube is within the FOV
    cube_in_fov = is_within_fov(cube_position)

    # 2D projection of the cube vertices (only if it's in FOV)
    if cube_in_fov:
        cube_vertices_2D = project_points(K, cube_vertices_rotated)
    else:
        cube_vertices_2D = None  # Set to None if not in FOV

    # Rotate the FOV pyramid based on tilt and pan
    fov_vertices_rotated = rotate_points(fov_vertices, tilt, pan)

    # Plotting
    fig = plt.figure(figsize=(10, 5))

    # 3D plot to visualize the setup
    ax3d = fig.add_subplot(121, projection='3d')
    
    # Add the 3D pinhole obstacle if the flag is set
    if isObstaclePresent:
        add_3d_pinhole_obstacle(ax3d)

    # Plot the cube in 3D
    plot_cube_on_camera_simulation(ax3d, position=cube_position)

    # Plot the rotated FOV pyramid in 3D
    ax3d.plot_trisurf(fov_vertices_rotated[:, 0], fov_vertices_rotated[:, 1], fov_vertices_rotated[:, 2], color='green', alpha=0.3)

    # Set labels and limits for the 3D view
    ax3d.set_xlabel('X')
    ax3d.set_ylabel('Y')
    ax3d.set_zlabel('Z')
    ax3d.set_xlim(-10, 20)
    ax3d.set_ylim(-10, 20)
    ax3d.set_zlim(0, 10)

    # 2D Camera View (Simulated image)
    ax2d = fig.add_subplot(122)

    # Plot the outer FOV rectangle (fixed size)
    ax2d.plot([40, 600, 600, 40, 40], [40, 40, 440, 440, 40], 'g-', label='FOV')

    if cube_in_fov and cube_vertices_2D is not None:
        # Plot the projected rectangle (the cube as seen from the camera)
        ax2d.fill([cube_vertices_2D[0, 0], cube_vertices_2D[1, 0], cube_vertices_2D[3, 0], cube_vertices_2D[2, 0], cube_vertices_2D[0, 0]], 
                  [cube_vertices_2D[0, 1], cube_vertices_2D[1, 1], cube_vertices_2D[3, 1], cube_vertices_2D[2, 1], cube_vertices_2D[0, 1]], 
                  color='orange', alpha=0.5, label='Cube Projection')
    else:
        # Display "No Projection" message when the cube is out of FOV
        ax2d.text(320, 240, "No Projection", fontsize=12, ha='center', va='center', color='red')

    ax2d.set_title('2D Camera View')
    ax2d.set_xlabel('X-axis')
    ax2d.set_ylabel('Y-axis')
    ax2d.set_xlim(0, 640)
    ax2d.set_ylim(0, 480)
    ax2d.axhline(0, color='black', lw=0.5)
    ax2d.axvline(0, color='black', lw=0.5)
    ax2d.grid()

    plt.tight_layout()
    plt.show()


# Example of changing the cube position
cube_position = np.array([0, 0, 30])  # CHANGE THIS TO EDIT CUBE POSITION
update_plot(cube_position, tilt= 0, pan= 0)  # CHANGE TILT AND PAN ANGLES HERE
