import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Time parameters
t_start = 0
t_end = 10
dt = 0.05  # Time step
t = np.arange(t_start, t_end + dt, dt)

# Circular trajectory parameters
T = 5  # Period of the circle
omega = 2 * np.pi / T  # Angular frequency

# Compute the trajectory
x = 1 + np.cos(omega * t)  # Centered at x=1
y = 1 + np.sin(omega * t)  # Centered at y=1
z = np.zeros_like(t)       # z = 0 for motion in the xy-plane

# Set up the figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set fixed limits for all axes
ax.set_xlim([-1.5, 2.5])
ax.set_ylim([-1.5, 2.5])
ax.set_zlim([-1.5, 2.5])

# Set labels for each axis
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Initialize the 3D OBJECT and the trajectory line
point, = ax.plot([], [], [], 'ro', label='Moving Point')  # Red dot EDIT WITH 3D FIGURE 
trajectory_line, = ax.plot([], [], [], 'b-', alpha=0.5, label='Trajectory')  # Blue line

# Initialization function for the animation
def init():
    point.set_data([], [])
    point.set_3d_properties([])
    trajectory_line.set_data([], [])
    trajectory_line.set_3d_properties([])
    return point, trajectory_line

# Update function for each frame of the animation
def update(frame):
    # Update the point position
    point.set_data(x[frame], y[frame])
    point.set_3d_properties(z[frame])

    # Update the trajectory line
    trajectory_line.set_data(x[:frame+1], y[:frame+1])
    trajectory_line.set_3d_properties(z[:frame+1])

    return point, trajectory_line

def savemp4(ani):
    Writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save("./animation.mp4", writer=Writer)
# Create the animation
ani = FuncAnimation(fig, update, frames=len(t), init_func=init,
                    interval=50, blit=True)

# Add a legend
ax.legend()

# Display the plot
plt.show()

savemp4(ani)
