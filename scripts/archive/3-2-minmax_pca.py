import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Step 1: Apply Z-score normalization
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Step 2: Apply PCA for dimensionality reduction (to 2 components)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_normalized)

# Print explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Create a 2D visualization
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_pca[target == i, 0], data_pca[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('MinMax Normalized PCA - Iris Dataset (2D Visualization)')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/3-2-minmax_pca_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Add zero column for 3D visualization (since we only have 2D from PCA)
data_pca_3d = np.column_stack((data_pca, np.zeros(len(data_pca))))

# Visualize PCA reduced data in 3D
output_path = "/workspaces/10944-seminar/images/3-2-minmax_pca_3d.png"
visualize_3d_scatter(
    data=data_pca_3d, 
    target=target,
    title="MinMax Normalized + PCA - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use all 3 dimensions for plotting
)