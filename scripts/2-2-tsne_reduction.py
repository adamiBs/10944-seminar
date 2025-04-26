import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Apply t-SNE for dimensionality reduction (to 2 components)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
data_tsne = tsne.fit_transform(data)

# Add zero column for 3D visualization
data_tsne_3d = np.column_stack((data_tsne, np.zeros(len(data_tsne))))

# Visualize t-SNE reduced data in 3D
output_path = "/workspaces/10944-seminar/images/2-2-tsne_reduction.png"
visualize_3d_scatter(
    data=data_tsne_3d, 
    target=target,
    title="t-SNE - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)