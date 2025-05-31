import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
import umap
from utils.common import visualize_3d_scatter
# Import our evaluation utilities
from utils.evaluation import evaluate_dimensionality_reduction

# Load wine dataset
wine = load_wine()
data = wine.data
feature_names = wine.feature_names
target = wine.target
target_names = wine.target_names

# With MinMax normalization
# --------------------------
# Step 1: Apply MinMax normalization
scaler = MinMaxScaler()
data_normalized = scaler.fit_transform(data)

# Step 2: Apply PCA for dimensionality reduction (to 2 components) - for comparison
pca_normalized = PCA(n_components=2)
data_pca_normalized = pca_normalized.fit_transform(data_normalized)

# Evaluate PCA with MinMax normalization
metrics_pca_normalized = evaluate_dimensionality_reduction(
    X=data_normalized,
    X_reduced=data_pca_normalized,
    y=target,
    n_neighbors=10
)

# Save PCA MinMax metrics to a text file
metrics_path_pca_norm = '/workspaces/10944-seminar/images/3-5-minmax_pca_wine_metrics.txt'
with open(metrics_path_pca_norm, 'w') as f:
    f.write("PCA with MinMax Normalization - Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_pca_normalized.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Print explained variance ratio (normalized data)
print("WITH MINMAX NORMALIZATION:")
print(f"Explained variance ratio: {pca_normalized.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_normalized.explained_variance_ratio_):.4f}")
print("\nPCA with MinMax normalization quality metrics:")
for metric_name, metric_value in metrics_pca_normalized.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Step 3: Apply UMAP for dimensionality reduction (to 2 components)
umap_reducer = umap.UMAP(n_components=2, random_state=42)
data_umap_normalized = umap_reducer.fit_transform(data_normalized)

# Evaluate UMAP with MinMax normalization
metrics_umap_normalized = evaluate_dimensionality_reduction(
    X=data_normalized,
    X_reduced=data_umap_normalized,
    y=target,
    n_neighbors=10
)

# Save UMAP MinMax metrics to a text file
metrics_path_umap_norm = '/workspaces/10944-seminar/images/3-5-minmax_umap_wine_metrics.txt'
with open(metrics_path_umap_norm, 'w') as f:
    f.write("UMAP with MinMax Normalization - Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_umap_normalized.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a bar plot of the UMAP MinMax metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics_umap_normalized.keys(), metrics_umap_normalized.values())
plt.title('UMAP with MinMax Normalization - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-5-minmax_umap_wine_metrics.png', dpi=300)
plt.close()

print("WITH MINMAX NORMALIZATION:")
print("UMAP applied to normalized data")
print("\nUMAP with MinMax normalization quality metrics:")
for metric_name, metric_value in metrics_umap_normalized.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Create a 2D visualization for normalized UMAP data
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_umap_normalized[target == i, 0], data_umap_normalized[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('First UMAP Component')
plt.ylabel('Second UMAP Component')
plt.title('UMAP with MinMax Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d_umap = "/workspaces/10944-seminar/images/3-5-minmax_umap_wine_2d.png"
plt.savefig(output_path_2d_umap)
plt.close()

# Try 3D UMAP projection as well
umap_3d = umap.UMAP(n_components=3, random_state=42)
data_umap_3d = umap_3d.fit_transform(data_normalized)

# Evaluate 3D UMAP
metrics_umap_3d = evaluate_dimensionality_reduction(
    X=data_normalized,
    X_reduced=data_umap_3d,
    y=target,
    n_neighbors=10
)

# Save metrics for 3D UMAP
metrics_path_umap_3d = '/workspaces/10944-seminar/images/3-5-minmax_umap_wine_3d_metrics.txt'
with open(metrics_path_umap_3d, 'w') as f:
    f.write("UMAP 3D with MinMax Normalization - Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_umap_3d.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create 3D visualization using the helper function
visualize_3d_scatter(
    data_umap_3d, 
    target,
    target_names,
    title="3D UMAP with MinMax Normalization - Wine Dataset",
    output_path="/workspaces/10944-seminar/images/3-5-minmax_umap_wine_3d.png"
)

# Create a comparative bar chart for all methods
plt.figure(figsize=(12, 7))

methods = ['PCA+MinMax', 'UMAP+MinMax']
metrics_dicts = [metrics_pca_normalized, metrics_umap_normalized]
metric_names = list(metrics_pca_normalized.keys())

# Create a grouped bar chart for each metric
n_groups = len(methods)
n_metrics = len(metric_names)
bar_width = 0.2
index = np.arange(n_groups)

for i, metric in enumerate(metric_names):
    values = [metrics_dict[metric] for metrics_dict in metrics_dicts]
    plt.bar(index + i*bar_width, values, bar_width, label=metric)

plt.xlabel('Method')
plt.ylabel('Score')
plt.title('Comparison of Quality Metrics With MinMax Normalization')
plt.xticks(index + bar_width * (n_metrics/2 - 0.5), methods)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-5-minmax_umap_pca_comparison.png', dpi=300)
plt.close()

print(f"Script execution completed. Output images and metrics saved in the /workspaces/10944-seminar/images/ directory.")
print(f"Metrics saved to:")
print(f"- UMAP with MinMax: {metrics_path_umap_norm}")
print(f"- UMAP 3D with MinMax: {metrics_path_umap_3d}")