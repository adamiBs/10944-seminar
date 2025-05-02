# filepath: /workspaces/10944-seminar/scripts/3-4-feature_pca_reduction.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Per-feature normalization (each feature scaled to unit norm)
data_norm_feature = normalize(data, norm='l2', axis=0)

# Apply PCA for dimensionality reduction (to 2 components)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_norm_feature)

# Print explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Add zero column for 3D visualization
data_pca_3d = np.column_stack((data_pca, np.zeros(len(data_pca))))

# Visualize PCA reduced data in 3D
output_path = "/workspaces/10944-seminar/images/3-4-feature_pca_reduction.png"
visualize_3d_scatter(
    data=data_pca_3d, 
    target=target,
    title="Feature Normalization + PCA - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1]  # Use the first 2 PCA components for plotting
)