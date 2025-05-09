import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import normalize
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Per-feature normalization (each feature scaled to unit norm)
data_normalized = normalize(data, norm='l2', axis=0)

# Apply UMAP to the normalized data
reducer = umap.UMAP(n_components=2, random_state=42)
data_umap = reducer.fit_transform(data_normalized)

# Create a 2D visualization
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        data_umap[target == i, 0],
        data_umap[target == i, 1],
        color=color,
        lw=2,
        label=target_name
    )

plt.title("Per-feature Normalized + UMAP - Iris Dataset (2D Visualization)")
plt.xlabel("UMAP Component 1")
plt.ylabel("UMAP Component 2")
plt.legend(loc="best")
plt.grid(True)

# Save 2D plot
output_path_2d = "/workspaces/10944-seminar/images/5-3-feature_umap_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Add zero column for 3D visualization (since we only have 2D from UMAP)
data_umap_3d = np.column_stack((data_umap, np.zeros(len(data_umap))))

# Create 3D scatter plot of UMAP-transformed data
output_path_3d = "/workspaces/10944-seminar/images/5-3-feature_umap_3d.png"
visualize_3d_scatter(
    data=data_umap_3d, 
    target=target,
    title="Per-feature Normalized + UMAP - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path_3d,
    features_to_use=[0, 1, 2]  # Use all 3 dimensions for plotting
)