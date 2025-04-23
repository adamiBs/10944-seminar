import numpy as np
import matplotlib.pyplot as plt
import umap
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Apply UMAP for dimensionality reduction (to 3 components)
reducer = umap.UMAP(n_components=3, random_state=42)
data_umap = reducer.fit_transform(data)

# Visualize UMAP reduced data in 3D
output_path = "/workspaces/10944-seminar/images/2-3-umap_reduction.png"
visualize_3d_scatter(
    data=data_umap, 
    target=target,
    title="UMAP - Iris Dataset (3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use all 3 UMAP components
)