import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing()
data = housing.data
feature_names = housing.feature_names
target = housing.target

# 1. Z-score normalization with PCA
# ---------------------------------
# Apply Z-score normalization
z_scaler = StandardScaler()
data_z_normalized = z_scaler.fit_transform(data)

# Apply PCA for dimensionality reduction (to 3 components)
pca_z_normalized = PCA(n_components=3)  # Changed to 3 components
data_pca_z_normalized = pca_z_normalized.fit_transform(data_z_normalized)

# Print explained variance ratio
print("WITH Z-SCORE NORMALIZATION:")
print(f"Explained variance ratio: {pca_z_normalized.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_z_normalized.explained_variance_ratio_):.4f}")

# Create a 3D visualization for Z-score normalized data
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')  # Create 3D subplot

# Since housing price is continuous, use a colormap for visualization
scatter = ax.scatter(data_pca_z_normalized[:, 0], 
                     data_pca_z_normalized[:, 1],
                     data_pca_z_normalized[:, 2],  # Added third component
                     c=target, cmap='viridis', alpha=0.6, s=10)

plt.colorbar(scatter, label='Median House Value')
ax.set_xlabel(f'First Principal Component ({pca_z_normalized.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'Second Principal Component ({pca_z_normalized.explained_variance_ratio_[1]:.2%} variance)')
ax.set_zlabel(f'Third Principal Component ({pca_z_normalized.explained_variance_ratio_[2]:.2%} variance)')  # Added z-label
ax.set_title('3D PCA with Z-score Normalization - California Housing Dataset')

# Save the 3D visualization
output_path_zscore = "/workspaces/10944-seminar/images/7-4-california_housing_zscore_pca_3d.png"
plt.savefig(output_path_zscore)
plt.close()

# 2. MinMax normalization with PCA
# --------------------------------
# Apply MinMax normalization
mm_scaler = MinMaxScaler()
data_mm_normalized = mm_scaler.fit_transform(data)

# Apply PCA for dimensionality reduction (to 3 components)
pca_mm_normalized = PCA(n_components=3)  # Changed to 3 components
data_pca_mm_normalized = pca_mm_normalized.fit_transform(data_mm_normalized)

# Print explained variance ratio
print("\nWITH MINMAX NORMALIZATION:")
print(f"Explained variance ratio: {pca_mm_normalized.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_mm_normalized.explained_variance_ratio_):.4f}")

# Create a 3D visualization for MinMax normalized data
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')  # Create 3D subplot

scatter = ax.scatter(data_pca_mm_normalized[:, 0], 
                     data_pca_mm_normalized[:, 1],
                     data_pca_mm_normalized[:, 2],  # Added third component
                     c=target, cmap='viridis', alpha=0.6, s=10)

plt.colorbar(scatter, label='Median House Value')
ax.set_xlabel(f'First Principal Component ({pca_mm_normalized.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'Second Principal Component ({pca_mm_normalized.explained_variance_ratio_[1]:.2%} variance)')
ax.set_zlabel(f'Third Principal Component ({pca_mm_normalized.explained_variance_ratio_[2]:.2%} variance)')  # Added z-label
ax.set_title('3D PCA with MinMax Normalization - California Housing Dataset')

# Save the 3D visualization
output_path_minmax = "/workspaces/10944-seminar/images/7-4-california_housing_minmax_pca_3d.png"
plt.savefig(output_path_minmax)
plt.close()

print("3D visualizations saved to images folder.")