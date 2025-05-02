import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Apply PCA whitening (decorrelate features)
pca = PCA(whiten=True)
data_white = pca.fit_transform(data)

# Print explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Add zero column for 3D visualization
data_white_3d = np.column_stack((data_white, np.zeros(len(data_white))))

# Visualize PCA whitened data in 3D
output_path = "/workspaces/10944-seminar/images/3-3-pca_whitening_pca_reduction.png"
visualize_3d_scatter(
    data=data_white_3d, 
    target=target,
    title="PCA Whitening - Iris Dataset (3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use all 3 dimensions for plotting
)