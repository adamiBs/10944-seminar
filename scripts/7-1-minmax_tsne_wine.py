import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
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
plt.title('PCA with MinMax Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/7-1-minmax_pca_wine_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Step 2: Apply t-SNE for dimensionality reduction (to 2 components)
tsne_normalized = TSNE(n_components=2, random_state=42)
data_tsne_normalized = tsne_normalized.fit_transform(data_normalized)

print("WITH Z-SCORE NORMALIZATION:")
print("t-SNE applied to normalized data")

# Create a 2D visualization for normalized data
plt.figure(figsize=(10, 8))

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_tsne_normalized[target == i, 0], data_tsne_normalized[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('First t-SNE Component')
plt.ylabel('Second t-SNE Component')
plt.title('t-SNE with MinMax Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d_tsne = "/workspaces/10944-seminar/images/7-1-minmax_tsne_wine_2d.png"
plt.savefig(output_path_2d_tsne)
plt.close()
