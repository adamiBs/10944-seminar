import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from utils.common import visualize_3d_scatter
# Import our evaluation utilities
from utils.evaluation import evaluate_dimensionality_reduction

# Load wine dataset
wine = load_wine()
data = wine.data
feature_names = wine.feature_names
target = wine.target
target_names = wine.target_names

# With Z-score normalization
# --------------------------
# Step 1: Apply Z-score normalization
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# Step 2: Apply PCA for dimensionality reduction (to 2 components)
pca_normalized = PCA(n_components=2)
data_pca_normalized = pca_normalized.fit_transform(data_normalized)

# Evaluate PCA with Z-score normalization
metrics_pca_normalized = evaluate_dimensionality_reduction(
    X=data_normalized,
    X_reduced=data_pca_normalized,
    y=target,
    n_neighbors=10
)

# Save PCA Z-score metrics to a text file
metrics_path_pca_norm = '/workspaces/10944-seminar/images/3-2-zscore_pca_wine_metrics.txt'
with open(metrics_path_pca_norm, 'w') as f:
    f.write("PCA with Z-score Normalization - Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_pca_normalized.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a bar plot of the PCA Z-score metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics_pca_normalized.keys(), metrics_pca_normalized.values())
plt.title('PCA with Z-score Normalization - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-2-zscore_pca_wine_metrics.png', dpi=300)
plt.close()

# Print explained variance ratio (normalized data)
print("WITH Z-SCORE NORMALIZATION:")
print(f"Explained variance ratio: {pca_normalized.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_normalized.explained_variance_ratio_):.4f}")
print("\nPCA with Z-score normalization quality metrics:")
for metric_name, metric_value in metrics_pca_normalized.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Create a 2D visualization for normalized data
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_pca_normalized[target == i, 0], data_pca_normalized[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel(f'First Principal Component ({pca_normalized.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca_normalized.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA with Z-score Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/3-2-zscore_pca_wine_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Step 2: Apply t-SNE for dimensionality reduction (to 2 components)
tsne_normalized = TSNE(n_components=2, random_state=42)
data_tsne_normalized = tsne_normalized.fit_transform(data_normalized)

# Evaluate t-SNE with Z-score normalization
metrics_tsne_normalized = evaluate_dimensionality_reduction(
    X=data_normalized,
    X_reduced=data_tsne_normalized,
    y=target,
    n_neighbors=10
)

# Save t-SNE Z-score metrics to a text file
metrics_path_tsne_norm = '/workspaces/10944-seminar/images/3-2-zscore_tsne_wine_metrics.txt'
with open(metrics_path_tsne_norm, 'w') as f:
    f.write("t-SNE with Z-score Normalization - Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_tsne_normalized.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a bar plot of the t-SNE Z-score metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics_tsne_normalized.keys(), metrics_tsne_normalized.values())
plt.title('t-SNE with Z-score Normalization - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-2-zscore_tsne_wine_metrics.png', dpi=300)
plt.close()

print("WITH Z-SCORE NORMALIZATION:")
print("t-SNE applied to normalized data")
print("\nt-SNE with Z-score normalization quality metrics:")
for metric_name, metric_value in metrics_tsne_normalized.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Create a 2D visualization for normalized data
plt.figure(figsize=(10, 8))

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_tsne_normalized[target == i, 0], data_tsne_normalized[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('First t-SNE Component')
plt.ylabel('Second t-SNE Component')
plt.title('t-SNE with Z-score Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d_tsne = "/workspaces/10944-seminar/images/3-2-zscore_tsne_wine_2d.png"
plt.savefig(output_path_2d_tsne)
plt.close()

# Without Z-score normalization
# ----------------------------
# Apply PCA directly to raw data
pca_raw = PCA(n_components=2)
data_pca_raw = pca_raw.fit_transform(data)

# Evaluate PCA with raw data
metrics_pca_raw = evaluate_dimensionality_reduction(
    X=data,
    X_reduced=data_pca_raw,
    y=target,
    n_neighbors=10
)

# Save PCA raw metrics to a text file
metrics_path_pca_raw = '/workspaces/10944-seminar/images/3-2-raw_pca_wine_metrics.txt'
with open(metrics_path_pca_raw, 'w') as f:
    f.write("PCA without Normalization - Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_pca_raw.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a bar plot of the PCA raw metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics_pca_raw.keys(), metrics_pca_raw.values())
plt.title('PCA without Normalization - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-2-raw_pca_wine_metrics.png', dpi=300)
plt.close()

# Print explained variance ratio (raw data)
print("\nWITHOUT Z-SCORE NORMALIZATION:")
print(f"Explained variance ratio: {pca_raw.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_raw.explained_variance_ratio_):.4f}")
print("\nPCA with raw data quality metrics:")
for metric_name, metric_value in metrics_pca_raw.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Create a 2D visualization for raw data
plt.figure(figsize=(10, 8))

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_pca_raw[target == i, 0], data_pca_raw[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel(f'First Principal Component ({pca_raw.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'Second Principal Component ({pca_raw.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA without Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d_raw = "/workspaces/10944-seminar/images/3-2-raw_pca_wine_2d.png"
plt.savefig(output_path_2d_raw)
plt.close()

# Apply t-SNE directly to raw data
tsne_raw = TSNE(n_components=2, random_state=42)
data_tsne_raw = tsne_raw.fit_transform(data)

# Evaluate t-SNE with raw data
metrics_tsne_raw = evaluate_dimensionality_reduction(
    X=data,
    X_reduced=data_tsne_raw,
    y=target,
    n_neighbors=10
)

# Save t-SNE raw metrics to a text file
metrics_path_tsne_raw = '/workspaces/10944-seminar/images/3-2-raw_tsne_wine_metrics.txt'
with open(metrics_path_tsne_raw, 'w') as f:
    f.write("t-SNE without Normalization - Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_tsne_raw.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a bar plot of the t-SNE raw metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics_tsne_raw.keys(), metrics_tsne_raw.values())
plt.title('t-SNE without Normalization - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-2-raw_tsne_wine_metrics.png', dpi=300)
plt.close()

print("\nWITHOUT Z-SCORE NORMALIZATION:")
print("t-SNE applied to raw data")
print("\nt-SNE with raw data quality metrics:")
for metric_name, metric_value in metrics_tsne_raw.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Create a 2D visualization for raw data
plt.figure(figsize=(10, 8))

for color, i, target_name in zip(colors, range(len(target_names)), target_names):
    plt.scatter(data_tsne_raw[target == i, 0], data_tsne_raw[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('First t-SNE Component')
plt.ylabel('Second t-SNE Component')
plt.title('t-SNE without Normalization - Wine Dataset')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d_raw_tsne = "/workspaces/10944-seminar/images/3-2-raw_tsne_wine_2d.png"
plt.savefig(output_path_2d_raw_tsne)
plt.close()

# Visualize both in 3D (optional)
# For normalized data (PCA)
data_pca_normalized_3d = np.column_stack((data_pca_normalized, np.zeros(len(data_pca_normalized))))
output_path_pca = "/workspaces/10944-seminar/images/3-2-zscore_pca_wine_3d.png"
visualize_3d_scatter(
    data=data_pca_normalized_3d, 
    target=target,
    title="PCA with Z-score Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path_pca,
    features_to_use=[0, 1, 2]
)

# For normalized data (t-SNE)
data_tsne_normalized_3d = np.column_stack((data_tsne_normalized, np.zeros(len(data_tsne_normalized))))
output_path_tsne = "/workspaces/10944-seminar/images/3-2-zscore_tsne_wine_3d.png"
visualize_3d_scatter(
    data=data_tsne_normalized_3d, 
    target=target,
    title="t-SNE with Z-score Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path_tsne,
    features_to_use=[0, 1, 2]
)

# For raw data (PCA)
data_pca_raw_3d = np.column_stack((data_pca_raw, np.zeros(len(data_pca_raw))))
output_path_raw_pca = "/workspaces/10944-seminar/images/3-2-raw_pca_wine_3d.png"
visualize_3d_scatter(
    data=data_pca_raw_3d, 
    target=target,
    title="PCA without Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path_raw_pca,
    features_to_use=[0, 1, 2]
)

# For raw data (t-SNE)
data_tsne_raw_3d = np.column_stack((data_tsne_raw, np.zeros(len(data_tsne_raw))))
output_path_raw_tsne = "/workspaces/10944-seminar/images/3-2-raw_tsne_wine_3d.png"
visualize_3d_scatter(
    data=data_tsne_raw_3d, 
    target=target,
    title="t-SNE without Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path_raw_tsne,
    features_to_use=[0, 1, 2]
)

# Create a comparative bar chart for all four methods
plt.figure(figsize=(15, 8))

methods = ['PCA+Z-score', 't-SNE+Z-score', 'PCA+Raw', 't-SNE+Raw']
metrics_dicts = [metrics_pca_normalized, metrics_tsne_normalized, metrics_pca_raw, metrics_tsne_raw]
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
plt.title('Comparison of Quality Metrics Across Methods')
plt.xticks(index + bar_width * (n_metrics/2 - 0.5), methods)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-2-wine_methods_comparison.png', dpi=300)
plt.close()

print(f"Script execution completed. Output images and metrics saved in the /workspaces/10944-seminar/images/ directory.")