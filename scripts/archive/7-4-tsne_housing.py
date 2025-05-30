import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
from sklearn.manifold import TSNE  # Changed from PCA to TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.datasets import fetch_california_housing

# Load California Housing dataset
housing = fetch_california_housing()
data = housing.data
feature_names = housing.feature_names
target = housing.target

# 1. Z-score normalization with t-SNE
# ----------------------------------
# Apply Z-score normalization
z_scaler = StandardScaler()
data_z_normalized = z_scaler.fit_transform(data)

# Apply t-SNE for dimensionality reduction (to 3 components)
print("Computing t-SNE with Z-score normalization...")
tsne_z_normalized = TSNE(n_components=3, random_state=42, perplexity=30)
data_tsne_z_normalized = tsne_z_normalized.fit_transform(data_z_normalized)

# Create a 3D visualization for Z-score normalized data
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')  # Create 3D subplot

# Since housing price is continuous, use a colormap for visualization
scatter = ax.scatter(data_tsne_z_normalized[:, 0], 
                     data_tsne_z_normalized[:, 1],
                     data_tsne_z_normalized[:, 2],
                     c=target, cmap='viridis', alpha=0.6, s=10)

plt.colorbar(scatter, label='Median House Value')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.set_title('3D t-SNE with Z-score Normalization - California Housing Dataset')

# Save the 3D visualization
output_path_zscore = "/workspaces/10944-seminar/images/7-4-california_housing_zscore_tsne_3d.png"
plt.savefig(output_path_zscore)
plt.close()

# 2. MinMax normalization with t-SNE
# ---------------------------------
# Apply MinMax normalization
mm_scaler = MinMaxScaler()
data_mm_normalized = mm_scaler.fit_transform(data)

# Apply t-SNE for dimensionality reduction (to 3 components)
print("\nComputing t-SNE with MinMax normalization...")
tsne_mm_normalized = TSNE(n_components=3, random_state=42, perplexity=30)
data_tsne_mm_normalized = tsne_mm_normalized.fit_transform(data_mm_normalized)

# Create a 3D visualization for MinMax normalized data
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')  # Create 3D subplot

scatter = ax.scatter(data_tsne_mm_normalized[:, 0], 
                     data_tsne_mm_normalized[:, 1],
                     data_tsne_mm_normalized[:, 2],
                     c=target, cmap='viridis', alpha=0.6, s=10)

plt.colorbar(scatter, label='Median House Value')
ax.set_xlabel('t-SNE Component 1')
ax.set_ylabel('t-SNE Component 2')
ax.set_zlabel('t-SNE Component 3')
ax.set_title('3D t-SNE with MinMax Normalization - California Housing Dataset')

# Save the 3D visualization
output_path_minmax = "/workspaces/10944-seminar/images/7-4-california_housing_minmax_tsne_3d.png"
plt.savefig(output_path_minmax)
plt.close()

print("3D t-SNE visualizations saved to images folder.")