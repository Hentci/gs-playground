import numpy as np

# Load the .npy file to analyze its contents
file_path = "/project/hentci/mip-nerf-360/bicycle/poses_bounds.npy"
poses_bounds = np.load(file_path)

# Display the shape and a preview of the data to understand its structure
print(poses_bounds.shape, poses_bounds[:5])  # Previewing first 5 entries