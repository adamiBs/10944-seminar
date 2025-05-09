import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Per-feature normalization (each feature scaled to unit norm)
data_normalized = normalize(data, norm='l2', axis=0)

# Apply t-SNE to the normalized data
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
data_tsne = tsne.fit_transform(data_normalized)

# Create a 2D visualization
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(
        data_tsne[target == i, 0],
        data_tsne[target == i, 1],
        color=color,
        lw=2,
        label=target_name
    )

plt.title("Per-feature Normalized + t-SNE - Iris Dataset (2D Visualization)")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend(loc="best")
plt.grid(True)

# Save 2D plot
output_path_2d = "/workspaces/10944-seminar/images/4-3-feature_tsne_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Add zero column for 3D visualization (since we only have 2D from t-SNE)
data_tsne_3d = np.column_stack((data_tsne, np.zeros(len(data_tsne))))

# Create 3D scatter plot of t-SNE-transformed data
output_path_3d = "/workspaces/10944-seminar/images/4-3-feature_tsne_3d.png"
visualize_3d_scatter(
    data=data_tsne_3d, 
    target=target,
    title="Per-feature Normalized + t-SNE - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path_3d,
    features_to_use=[0, 1, 2]  # Use all 3 dimensions for plotting
)