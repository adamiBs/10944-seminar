import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils.common import load_iris_dataset, visualize_3d_scatter
# Import our evaluation utilities
from utils.evaluation import evaluate_dimensionality_reduction

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Apply PCA for dimensionality reduction (to 2 components)
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data)

# Print explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Evaluate the dimensionality reduction using our metrics
metrics = evaluate_dimensionality_reduction(
    X=data,
    X_reduced=data_pca,
    y=target,
    n_neighbors=10
)

# Print the evaluation metrics
print("\nDimensionality Reduction Quality Metrics:")
print("-----------------------------------------")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Save metrics to a text file
metrics_path = '/workspaces/10944-seminar/images/2-1-pca_metrics.txt'
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
    plt.scatter(data_pca[target == i, 0], data_pca[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA - Iris Dataset (2D Visualization)')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/2-1-pca_reduction_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Create a bar plot of the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('PCA Dimensionality Reduction Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/2-1-pca_metrics.png', dpi=300)
plt.close()

# Add zero column for 3D visualization
data_pca_3d = np.column_stack((data_pca, np.zeros(len(data_pca))))

# Visualize PCA reduced data in 3D
output_path = "/workspaces/10944-seminar/images/2-1-pca_reduction_3d.png"
visualize_3d_scatter(
    data=data_pca_3d, 
    target=target,
    title="PCA - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use all 3 dimensions for plotting
)

print(f"Script execution completed. Output images saved in the /workspaces/10944-seminar/images/ directory.")
print(f"Evaluation metrics saved to {metrics_path}")