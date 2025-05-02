import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Per-feature normalization (each feature scaled to unit norm)
data_normalized = normalize(data, norm='l2', axis=0)

# Apply PCA to the normalized data
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data_normalized)

# Create 3D scatter plot of PCA-transformed data
output_path_3d = "/workspaces/10944-seminar/images/3-3-feature_pca_3d.png"
visualize_3d_scatter(
    data=data_pca, 
    target=target,
    title="Per-feature Normalized + PCA - Iris Dataset (3D Visualization)",
    save_path=output_path_3d
)

# Create 2D scatter plot using first two PCA components
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        data_pca[target == i, 0],
        data_pca[target == i, 1],
        color=color,
        lw=2,
        label=target_name
    )

plt.title("Per-feature Normalized + PCA - Iris Dataset (2D Visualization)")
plt.xlabel(f"First Principal Component ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"Second Principal Component ({pca.explained_variance_ratio_[1]:.2%})")
plt.legend(loc="best")
plt.grid(True)

# Save 2D plot
output_path_2d = "/workspaces/10944-seminar/images/3-3-feature_pca_2d.png"
plt.savefig(output_path_2d)
plt.tight_layout()
plt.show()

# Print explained variance ratio
print("Explained variance ratio by principal components:")
for i, ratio in enumerate(pca.explained_variance_ratio_):
    print(f"PC{i+1}: {ratio:.4f} ({ratio:.2%})")
print(f"Total variance explained by first 2 components: {sum(pca.explained_variance_ratio_[:2]):.2%}")