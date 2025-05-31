import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

# Print explained variance ratio (normalized data)
print("WITH Z-SCORE NORMALIZATION:")
print(f"Explained variance ratio: {pca_normalized.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_normalized.explained_variance_ratio_):.4f}")

# Evaluate the normalized PCA dimensionality reduction
metrics_normalized = evaluate_dimensionality_reduction(
    X=data_normalized,
    X_reduced=data_pca_normalized,
    y=target,
    n_neighbors=10
)

# Print the evaluation metrics for normalized data
print("\nDimensionality Reduction Quality Metrics (Z-score Normalized):")
print("-----------------------------------------")
for metric_name, metric_value in metrics_normalized.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Save normalized PCA metrics to a text file
metrics_path_normalized = '/workspaces/10944-seminar/images/3-1-zscore_pca_wine_metrics.txt'
with open(metrics_path_normalized, 'w') as f:
    f.write("Dimensionality Reduction Quality Metrics (Z-score Normalized):\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_normalized.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a bar plot of the metrics for normalized data
plt.figure(figsize=(10, 6))
plt.bar(metrics_normalized.keys(), metrics_normalized.values())
plt.title('PCA with Z-score Normalization - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-1-zscore_pca_wine_metrics.png', dpi=300)
plt.close()

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
output_path_2d = "/workspaces/10944-seminar/images/3-1-zscore_pca_wine_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Without Z-score normalization
# ----------------------------
# Apply PCA directly to raw data
pca_raw = PCA(n_components=2)
data_pca_raw = pca_raw.fit_transform(data)

# Print explained variance ratio (raw data)
print("\nWITHOUT Z-SCORE NORMALIZATION:")
print(f"Explained variance ratio: {pca_raw.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca_raw.explained_variance_ratio_):.4f}")

# Evaluate the raw PCA dimensionality reduction
metrics_raw = evaluate_dimensionality_reduction(
    X=data,
    X_reduced=data_pca_raw,
    y=target,
    n_neighbors=10
)

# Print the evaluation metrics for raw data
print("\nDimensionality Reduction Quality Metrics (Raw Data):")
print("-----------------------------------------")
for metric_name, metric_value in metrics_raw.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Save raw PCA metrics to a text file
metrics_path_raw = '/workspaces/10944-seminar/images/3-1-raw_pca_wine_metrics.txt'
with open(metrics_path_raw, 'w') as f:
    f.write("Dimensionality Reduction Quality Metrics (Raw Data):\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_raw.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a bar plot of the metrics for raw data
plt.figure(figsize=(10, 6))
plt.bar(metrics_raw.keys(), metrics_raw.values())
plt.title('PCA without Normalization - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/3-1-raw_pca_wine_metrics.png', dpi=300)
plt.close()

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
output_path_2d_raw = "/workspaces/10944-seminar/images/3-1-raw_pca_wine_2d.png"
plt.savefig(output_path_2d_raw)
plt.close()

# Visualize both in 3D (optional)
# For normalized data
data_pca_normalized_3d = np.column_stack((data_pca_normalized, np.zeros(len(data_pca_normalized))))
output_path = "/workspaces/10944-seminar/images/3-1-zscore_pca_wine_3d.png"
visualize_3d_scatter(
    data=data_pca_normalized_3d, 
    target=target,
    title="PCA with Z-score Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)

# For raw data
data_pca_raw_3d = np.column_stack((data_pca_raw, np.zeros(len(data_pca_raw))))
output_path_raw = "/workspaces/10944-seminar/images/3-1-raw_pca_wine_3d.png"
visualize_3d_scatter(
    data=data_pca_raw_3d, 
    target=target,
    title="PCA without Normalization - Wine Dataset (2D → 3D)",
    save_path=output_path_raw,
    features_to_use=[0, 1, 2]
)

print(f"Script execution completed. Output images and metrics saved in the /workspaces/10944-seminar/images/ directory.")
print(f"Z-score normalized PCA metrics saved to {metrics_path_normalized}")
print(f"Raw PCA metrics saved to {metrics_path_raw}")