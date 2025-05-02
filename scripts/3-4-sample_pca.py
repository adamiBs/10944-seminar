import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Per-sample normalization (each sample vector scaled to unit norm)
data_normalized = normalize(data, norm='l2', axis=1)

# Apply PCA to reduce dimensions
pca = PCA(n_components=3)
pca_result = pca.fit_transform(data_normalized)

# Visualize the PCA result in 3D
output_path_3d = "/workspaces/10944-seminar/images/3-4-smaple_pca_3d.png"
visualize_3d_scatter(
    data=pca_result,
    target=target,
    title="Per-sample Normalization + PCA - Iris Dataset (3D Visualization)",
    save_path=output_path_3d
)

# Create a 2D visualization (using only the first two PCA components)
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(pca_result[target == i, 0], pca_result[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('Per-sample Normalization + PCA - Iris Dataset (2D Visualization)')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/3-4-smaple_pca_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Print explained variance ratios
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {np.sum(pca.explained_variance_ratio_):.2%}")