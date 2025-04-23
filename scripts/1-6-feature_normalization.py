import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Apply PCA for dimensionality reduction (to 3 components)
pca = PCA(n_components=3)
data_pca = pca.fit_transform(data)

# Print explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Visualize PCA reduced data in 3D
output_path = "/workspaces/10944-seminar/images/2-1-pca_reduction.png"
visualize_3d_scatter(
    data=data_pca, 
    target=target,
    title="PCA - Iris Dataset (3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use all 3 PCA components
)