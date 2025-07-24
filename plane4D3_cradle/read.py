import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the data from CSV
data = np.loadtxt("/home/kiyong/BEACLS/beacls/sources/samples/plane4D3_cradle/all_loop.csv", delimiter=",")  # Ensure the correct delimiter is used

# Grid parameters
Nx = 11  # Number of points in x-direction

# Ensure data size matches the expected grid size
assert data.size == Nx * Nx * Nx * Nx * Nx * Nx, f"Data size {data.size} does not match expected size {Nx * Nx * Nx * Nx * Nx * Nx}."

# Reshape the data into a 6D grid
data_4d = data.reshape((Nx, Nx, Nx, Nx, Nx, Nx)).transpose(5, 4, 3, 2, 1, 0)


	# beacls::FloatVec mins{ (FLOAT_TYPE)-10, (FLOAT_TYPE)-5, (FLOAT_TYPE)-5, (FLOAT_TYPE)-5, (FLOAT_TYPE)-M_PI, (FLOAT_TYPE)-5};
	# beacls::FloatVec maxs{ (FLOAT_TYPE)+2,(FLOAT_TYPE)+5,(FLOAT_TYPE)+5,(FLOAT_TYPE)+5, (FLOAT_TYPE)M_PI, (FLOAT_TYPE)+5 };


	# beacls::FloatVec mins{ (FLOAT_TYPE)-5, (FLOAT_TYPE)-3, (FLOAT_TYPE)-5, (FLOAT_TYPE)-5};
	# beacls::FloatVec maxs{ (FLOAT_TYPE)+1,(FLOAT_TYPE)+3,(FLOAT_TYPE)+5,(FLOAT_TYPE)+5 };
# Step 2: Map grid indices to physical coordinates
mins = [-5, -5, -5, -5, -np.pi]
maxs = [2, 5, 5, 5, np.pi]

# Generate the physical coordinates for each dimension
x_vals = np.linspace(mins[0], maxs[0], Nx)
# y_vals = np.linspace(mins[1], maxs[1], Nx)
y_vals = np.linspace(mins[4], maxs[4], Nx)

# Step 3: Find indices with negative values in the grid
neg_indices = np.argwhere(data_4d < 0.0)
print(neg_indices)
# Map indices to physical coordinates
x_coords = x_vals[neg_indices[:, 0]]
y_coords = y_vals[neg_indices[:, 1]]
y_coords = y_vals[neg_indices[:, 4]]

# Step 4: Visualize the negative value points (2D scatter plot)
plt.figure(figsize=(10, 7))

# Scatter plot for points with negative values
plt.scatter(x_coords, y_coords, c='red', s=5, alpha=0.6)

# Labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Points with Negative Values in all_loop.csv')

# Adjust limits to make the plot look better
x_range = x_vals.max() - x_vals.min()
y_range = y_vals.max() - y_vals.min()
xy_max_range = max(x_range, y_range)

# Compute midpoints for better visualization
x_mid = (x_vals.max() + x_vals.min()) / 2
y_mid = (y_vals.max() + y_vals.min()) / 2

# Set equal limits for X and Y
plt.xlim(x_mid - xy_max_range / 2, x_mid + xy_max_range / 2)
plt.ylim(y_mid - xy_max_range / 2, y_mid + xy_max_range / 2)

# Show the plot
plt.tight_layout()
plt.show()
