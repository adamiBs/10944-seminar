import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Whitening using PCA (decorrelate features)
pca = PCA(whiten=True)
data_white = pca.fit_transform(data)

# Visualize PCA whitened data in 3D
output_path = "/workspaces/10944-seminar/images/4-pca_whitening.png"
visualize_3d_scatter(
    data=data_white,
    target=target,
    title="PCA Whitened Iris Dataset (3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use the first 3 principal components
)