import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Step 1: Apply MinMax normalization
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Step 2: Apply t-SNE for dimensionality reduction (to 2 components)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
data_tsne = tsne.fit_transform(data_normalized)

# Create a 2D visualization
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_tsne[target == i, 0], data_tsne[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('MinMax Normalized t-SNE - Iris Dataset (2D Visualization)')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/4-2-minmax_tsne_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Add zero column for 3D visualization (since we only have 2D from t-SNE)
data_tsne_3d = np.column_stack((data_tsne, np.zeros(len(data_tsne))))

# Visualize t-SNE reduced data in 3D
output_path = "/workspaces/10944-seminar/images/4-2-minmax_tsne_3d.png"
visualize_3d_scatter(
    data=data_tsne_3d, 
    target=target,
    title="MinMax Normalized + t-SNE - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use all 3 dimensions for plotting
)