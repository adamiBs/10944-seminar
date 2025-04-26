import numpy as np
import matplotlib.pyplot as plt
import umap
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Apply UMAP for dimensionality reduction (to 2 components)
reducer = umap.UMAP(n_components=2, random_state=42)
data_umap = reducer.fit_transform(data)

# Add zero column for 3D visualization
data_umap_3d = np.column_stack((data_umap, np.zeros(len(data_umap))))

# Visualize UMAP reduced data in 3D
output_path = "/workspaces/10944-seminar/images/2-3-umap_reduction.png"
visualize_3d_scatter(
    data=data_umap_3d, 
    target=target,
    title="UMAP - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)