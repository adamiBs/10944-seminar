import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import MinMaxScaler
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Step 1: Apply MinMax normalization
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Step 2: Apply UMAP for dimensionality reduction (to 2 components)
reducer = umap.UMAP(n_components=2, random_state=42)
data_umap = reducer.fit_transform(data_normalized)

# Create a 2D visualization
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_umap[target == i, 0], data_umap[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.title('MinMax Normalized UMAP - Iris Dataset (2D Visualization)')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/5-2-minmax_umap_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Add zero column for 3D visualization (since we only have 2D from UMAP)
data_umap_3d = np.column_stack((data_umap, np.zeros(len(data_umap))))

# Visualize UMAP reduced data in 3D
output_path = "/workspaces/10944-seminar/images/5-2-minmax_umap_3d.png"
visualize_3d_scatter(
    data=data_umap_3d, 
    target=target,
    title="MinMax Normalized + UMAP - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use all 3 dimensions for plotting
)