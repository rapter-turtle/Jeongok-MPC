import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import mplot3d for 3D plotting

# Step 1: Load the data from CSV
data = np.loadtxt("/home/kiyong/BEACLS/beacls/sources/samples/plane4D3_cradle/all_loop.csv", delimiter=",")  # Ensure the correct delimiter is used

# Grid parameters
Nx = 11  # Number of points in x-direction

# Ensure data size matches the expected grid size
assert data.size == Nx * Nx * Nx * Nx* Nx * Nx, f"Data size {data.size} does not match expected size {Nx * Nx * Nx * Nx* Nx * Nx}."

# Reshape the data into a 6D grid (Nx x Nx x Nx x Nx x Nx x Nx)
data_3d = data.reshape((Nx, Nx, Nx, Nx, Nx, Nx)).transpose(5, 4, 3, 2, 1, 0)

# Step 2: Map grid indices to physical coordinates
# mins = [-10, -5, -np.pi]
# maxs = [2, 5, np.pi]

# mins = [-5, -5, -np.pi]
# maxs = [3, 5, np.pi]

mins = [-5, -5, -2.5]
maxs = [2, 5, 2.5]

# beacls::FloatVec mins{ (FLOAT_TYPE)-5, (FLOAT_TYPE)-5, (FLOAT_TYPE)-2.5, (FLOAT_TYPE)-2.5, (FLOAT_TYPE)-M_PI, (FLOAT_TYPE)-2};
# beacls::FloatVec maxs{ (FLOAT_TYPE)+3,(FLOAT_TYPE)+5,(FLOAT_TYPE)+2.5,(FLOAT_TYPE)+2.5, (FLOAT_TYPE)M_PI, (FLOAT_TYPE)+2 };


# Generate the physical coordinates for each dimension
x_vals = np.linspace(-5, 2, Nx)
y_vals = np.linspace(-5, 5, Nx)
u_vals = np.linspace(-2.5, 2.5, Nx)
v_vals = np.linspace(-2.5, 2.5, Nx)
psi_vals = np.linspace(-np.pi, np.pi, Nx)
r_vals = np.linspace(-2, 2, Nx)


# Step 3: Find indices with data_3d < 0
neg_indices = np.argwhere(data_3d < 0)

# Convert the 3rd index (u dimension) to physical coordinates
u_coords = u_vals[neg_indices[:, 2]]
v_coords = v_vals[neg_indices[:, 3]]
psi_coords = psi_vals[neg_indices[:, 4]]
r_coords = r_vals[neg_indices[:, 5]]

# Create mask for -1 < u < 1
mask = (u_coords > -2) & (u_coords < 2) & (v_coords > -1) & (v_coords < 1) & (psi_coords > -0.3*np.pi) & (psi_coords < 0.3*np.pi) & (r_coords > -0.5) & (r_coords < 0.5)

# Apply mask to filter valid indices
# filtered_indices = neg_indices[mask]
filtered_indices = neg_indices

# Map filtered indices to physical coordinates
x_coords = x_vals[filtered_indices[:, 0]]
y_coords = y_vals[filtered_indices[:, 1]]
# z_coords = psi_vals[filtered_indices[:, 4]]  # psi
z_coords = psi_vals[filtered_indices[:, 4]]  # u

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

# Compute range
# Set equal limits for X, Y, and Z axes
# ax.set_xlim(x_mid - xyz_max_range / 2, x_mid + xyz_max_range / 2)
# ax.set_ylim(y_mid - xyz_max_range / 2, y_mid + xyz_max_range / 2)
ax.set_xlim(-5, 2.0)
ax.set_ylim(-5, 5)
ax.set_zlim(-4.0, 4.0)

# Show the plot
plt.tight_layout()
plt.show()