import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
import umap.umap_ as umap
from utils.common import visualize_3d_scatter

# Load wine dataset
wine = load_wine()
data = wine.data
feature_names = wine.feature_names
target = wine.target
target_names = wine.target_names

# With Z-score normalization
# --------------------------
# Step 1: Apply Z-score normalization
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Step 2: Apply PCA for dimensionality reduction (to 2 components)
pca_normalized = PCA(n_components=2)
data_pca_normalized = pca_normalized.fit_transform(data_normalized)

# Print explained variance ratio (normalized data)
print("WITH Z-SCORE NORMALIZATION:")
print(f"Explained variance ratio: {pca_normalized.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_normalized.explained_variance_ratio_):.4f}")

# Create a 2D visualization for normalized data
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_pca_normalized[target == i, 0], data_pca_normalized[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel(f'First Principal Component ({pca_normalized.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca_normalized.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA with Z-score Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/3-3-zscore_pca_wine_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Step 2: Apply UMAP for dimensionality reduction (to 2 components)
reducer_normalized = umap.UMAP(n_components=2, random_state=42)
data_umap_normalized = reducer_normalized.fit_transform(data_normalized)

# Print information about the reduction
print("UMAP dimensionality reduction completed")

# Create a 2D visualization for normalized data
plt.figure(figsize=(10, 8))

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_umap_normalized[target == i, 0], data_umap_normalized[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP with Z-score Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/3-3-zscore_umap_wine_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Without Z-score normalization
# ----------------------------
# Apply PCA directly to raw data
pca_raw = PCA(n_components=2)
data_pca_raw = pca_raw.fit_transform(data)

# Print explained variance ratio (raw data)
print("\nWITHOUT Z-SCORE NORMALIZATION:")
print(f"Explained variance ratio: {pca_raw.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_raw.explained_variance_ratio_):.4f}")

# Create a 2D visualization for raw data
plt.figure(figsize=(10, 8))

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_pca_raw[target == i, 0], data_pca_raw[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel(f'First Principal Component ({pca_raw.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca_raw.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA without Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d_raw = "/workspaces/10944-seminar/images/3-3-raw_pca_wine_2d.png"
plt.savefig(output_path_2d_raw)
plt.close()

# Apply UMAP directly to raw data
reducer_raw = umap.UMAP(n_components=2, random_state=42)
data_umap_raw = reducer_raw.fit_transform(data)

# Print information about the reduction
print("UMAP dimensionality reduction completed")

# Create a 2D visualization for raw data
plt.figure(figsize=(10, 8))

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_umap_raw[target == i, 0], data_umap_raw[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP without Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d_raw = "/workspaces/10944-seminar/images/3-3-raw_umap_wine_2d.png"
plt.savefig(output_path_2d_raw)
plt.close()

# Visualize both in 3D (optional)
# For normalized data
data_pca_normalized_3d = np.column_stack((data_pca_normalized, np.zeros(len(data_pca_normalized))))
output_path = "/workspaces/10944-seminar/images/3-3-zscore_pca_wine_3d.png"
visualize_3d_scatter(
    data=data_pca_normalized_3d, 
    target=target,
    title="PCA with Z-score Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)

data_umap_normalized_3d = np.column_stack((data_umap_normalized, np.zeros(len(data_umap_normalized))))
output_path = "/workspaces/10944-seminar/images/3-3-zscore_umap_wine_3d.png"
visualize_3d_scatter(
    data=data_umap_normalized_3d, 
    target=target,
    title="UMAP with Z-score Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)

# For raw data
data_pca_raw_3d = np.column_stack((data_pca_raw, np.zeros(len(data_pca_raw))))
output_path_raw = "/workspaces/10944-seminar/images/3-3-raw_pca_wine_3d.png"
visualize_3d_scatter(
    data=data_pca_raw_3d, 
    target=target,
    title="PCA without Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path_raw,
    features_to_use=[0, 1, 2]
)

data_umap_raw_3d = np.column_stack((data_umap_raw, np.zeros(len(data_umap_raw))))
output_path_raw = "/workspaces/10944-seminar/images/3-3-raw_umap_wine_3d.png"
visualize_3d_scatter(
    data=data_umap_raw_3d, 
    target=target,
    title="UMAP without Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path_raw,
    features_to_use=[0, 1, 2]
)