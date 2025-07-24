import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import mplot3d for 3D plotting

# Step 1: Load the data from CSV
data = np.loadtxt("/home/kiyong/BEACLS/beacls/sources/samples/plane4D3_cradle/all_loop.csv", delimiter=",")  # Ensure the correct delimiter is used

# Grid parameters
Nx = 15  # Number of points in x-direction

# Ensure data size matches the expected grid size
assert data.size == Nx * Nx * Nx * Nx* Nx * Nx, f"Data size {data.size} does not match expected size {Nx * Nx * Nx * Nx* Nx * Nx}."

# Reshape the data into a 6D grid (Nx x Nx x Nx x Nx x Nx x Nx)
data_3d = data.reshape((Nx, Nx, Nx, Nx, Nx, Nx)).transpose(5, 4, 3, 2, 1, 0)

# Step 2: Map grid indices to physical coordinates
# mins = [-10, -5, -np.pi]
# maxs = [2, 5, np.pi]
mins = [-5, -5, -np.pi]
maxs = [2, 5, np.pi]

# Generate the physical coordinates for each dimension
x_vals = np.linspace(mins[0], maxs[0], Nx)
y_vals = np.linspace(mins[1], maxs[1], Nx)
z_vals = np.linspace(mins[2], maxs[2], Nx)

# Step 3: Find indices with negative values in the grid
neg_indices = np.argwhere(data_3d < 0.0)

# Map indices to physical coordinates
x_coords = x_vals[neg_indices[:, 0]]
y_coords = y_vals[neg_indices[:, 1]]
z_coords = z_vals[neg_indices[:, 4]]  # Corrected to use z_vals instead of y_vals for the Z-axis

# Step 4: Visualize the negative value points (3D scatter plot)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')  # Ensure 3D plot

# Scatter plot for points with negative values
ax.scatter(x_coords, y_coords, z_coords, c='red', alpha=0.6, s=5)

# Labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('PSI')  # Corrected to set z-axis label to 'PSI' for the Z-coordinate
ax.set_title('Points with Negative Values in all_loop.csv')

# Compute ranges
x_range = x_vals.max() - x_vals.min()
y_range = y_vals.max() - y_vals.min()
z_range = z_vals.max() - z_vals.min()
xyz_max_range = max(x_range, y_range, z_range)

# Compute midpoints
x_mid = (x_vals.max() + x_vals.min()) / 2
y_mid = (y_vals.max() + y_vals.min()) / 2
z_mid = (z_vals.max() + z_vals.min()) / 2

# Set equal limits for X, Y, and Z axes
ax.set_xlim(x_mid - xyz_max_range / 2, x_mid + xyz_max_range / 2)
ax.set_ylim(y_mid - xyz_max_range / 2, y_mid + xyz_max_range / 2)
ax.set_zlim(-5.5, 5.5)

# Show the plot
plt.tight_layout()
plt.show()
