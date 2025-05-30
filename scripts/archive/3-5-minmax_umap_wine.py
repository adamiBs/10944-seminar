import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
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
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Print explained variance ratio (normalized data)
print("WITH Z-SCORE NORMALIZATION:")

# Step 2: Apply UMAP for dimensionality reduction (to 2 components)
reducer_normalized = umap.UMAP(n_components=2, random_state=42)
data_umap_normalized = reducer_normalized.fit_transform(data_normalized)

# Print information about the reduction
print("UMAP dimensionality reduction completed")

# Create a 2D visualization for normalized data
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_umap_normalized[target == i, 0], data_umap_normalized[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('UMAP with MinMax Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/3-5-minmax_umap_wine_2d.png"
plt.savefig(output_path_2d)
plt.close()
