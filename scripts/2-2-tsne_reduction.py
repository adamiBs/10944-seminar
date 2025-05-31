import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.common import load_iris_dataset, visualize_3d_scatter
# Import our evaluation utilities
from utils.evaluation import evaluate_dimensionality_reduction

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Apply t-SNE for dimensionality reduction (to 2 components)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
data_tsne = tsne.fit_transform(data)

# Evaluate the dimensionality reduction using our metrics
metrics = evaluate_dimensionality_reduction(
    X=data,
    X_reduced=data_tsne,
    y=target,
    n_neighbors=10
)

# Print the evaluation metrics
print("\nDimensionality Reduction Quality Metrics:")
print("-----------------------------------------")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Save metrics to a text file
metrics_path = '/workspaces/10944-seminar/images/2-2-tsne_metrics.txt'
with open(metrics_path, 'w') as f:
    f.write("Dimensionality Reduction Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a 2D visualization
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(data_tsne[target == i, 0], data_tsne[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE - Iris Dataset (2D Visualization)')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/2-2-tsne_reduction_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Create a bar plot of the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('t-SNE Dimensionality Reduction Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/2-2-tsne_metrics.png', dpi=300)
plt.close()

# Add zero column for 3D visualization
data_tsne_3d = np.column_stack((data_tsne, np.zeros(len(data_tsne))))

# Visualize t-SNE reduced data in 3D
output_path = "/workspaces/10944-seminar/images/2-2-tsne_reduction_3d.png"
visualize_3d_scatter(
    data=data_tsne_3d, 
    target=target,
    title="t-SNE - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)

print(f"Script execution completed. Output images saved in the /workspaces/10944-seminar/images/ directory.")
print(f"Evaluation metrics saved to {metrics_path}")