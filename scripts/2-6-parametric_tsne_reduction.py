import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from utils.common import load_iris_dataset, visualize_3d_scatter
# Import our evaluation utilities
from utils.evaluation import evaluate_dimensionality_reduction

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Use data as is without normalization
data_scaled = data

# Generate t-SNE embedding as targets
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embedding = tsne.fit_transform(data_scaled)

# Create parametric t-SNE model
class ParametricTSNE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(2)  # Output 2D embedding
        ])
        
    def call(self, inputs):
        return self.encoder(inputs)

# Create and compile model
model = ParametricTSNE()
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mse'
)

# Train to approximate t-SNE embedding
X_train = tf.convert_to_tensor(data_scaled, dtype=tf.float32)
y_train = tf.convert_to_tensor(tsne_embedding, dtype=tf.float32)

history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=16,
    verbose=1
)

# Generate embeddings with trained model
parametric_tsne_embedding = model.predict(X_train)

# Evaluate the dimensionality reduction using our metrics
metrics_parametric_tsne = evaluate_dimensionality_reduction(
    X=data_scaled,
    X_reduced=parametric_tsne_embedding,
    y=target,
    n_neighbors=10
)

# Print the evaluation metrics for parametric t-SNE
print("\nParametric t-SNE Dimensionality Reduction Quality Metrics:")
print("-----------------------------------------")
for metric_name, metric_value in metrics_parametric_tsne.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Evaluate the original t-SNE as well for comparison
metrics_original_tsne = evaluate_dimensionality_reduction(
    X=data_scaled,
    X_reduced=tsne_embedding,
    y=target,
    n_neighbors=10
)

# Print the evaluation metrics for original t-SNE
print("\nOriginal t-SNE Dimensionality Reduction Quality Metrics:")
print("-----------------------------------------")
for metric_name, metric_value in metrics_original_tsne.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Save metrics to text files
metrics_path_parametric = '/workspaces/10944-seminar/images/2-6-parametric_tsne_metrics.txt'
with open(metrics_path_parametric, 'w') as f:
    f.write("Parametric t-SNE Dimensionality Reduction Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_parametric_tsne.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

metrics_path_original = '/workspaces/10944-seminar/images/2-6-original_tsne_metrics.txt'
with open(metrics_path_original, 'w') as f:
    f.write("Original t-SNE Dimensionality Reduction Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics_original_tsne.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create bar plots of the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics_parametric_tsne.keys(), metrics_parametric_tsne.values())
plt.title('Parametric t-SNE - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/2-6-parametric_tsne_metrics.png', dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.bar(metrics_original_tsne.keys(), metrics_original_tsne.values())
plt.title('Original t-SNE - Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/2-6-original_tsne_metrics.png', dpi=300)
plt.close()

# Create a comparative bar chart for both methods
plt.figure(figsize=(12, 8))

methods = ['Parametric t-SNE', 'Original t-SNE']
metrics_dicts = [metrics_parametric_tsne, metrics_original_tsne]
metric_names = list(metrics_parametric_tsne.keys())

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
plt.title('Comparison of Quality Metrics: Parametric t-SNE vs Original t-SNE')
plt.xticks(index + bar_width * (n_metrics/2 - 0.5), methods)
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/2-6-tsne_comparison.png', dpi=300)
plt.close()

# Create a 2D visualization of parametric t-SNE
plt.figure(figsize=(10, 8))
colors = ['navy', 'turquoise', 'darkorange']
target_names = ['Setosa', 'Versicolor', 'Virginica']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(parametric_tsne_embedding[target == i, 0], parametric_tsne_embedding[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('Parametric t-SNE Dimension 1')
plt.ylabel('Parametric t-SNE Dimension 2')
plt.title('Parametric t-SNE - Iris Dataset (2D Visualization)')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/2-6-parametric_tsne_reduction_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Add zero column for 3D visualization
parametric_tsne_embedding_3d = np.column_stack((parametric_tsne_embedding, np.zeros(len(parametric_tsne_embedding))))

# Visualize parametric t-SNE embeddings in 3D
output_path = "/workspaces/10944-seminar/images/2-6-parametric_tsne_reduction.png"
visualize_3d_scatter(
    data=parametric_tsne_embedding_3d, 
    target=target,
    title="Parametric t-SNE - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)

print(f"Script execution completed. Output images and metrics saved in the /workspaces/10944-seminar/images/ directory.")
print(f"Metrics saved to:")
print(f"- Parametric t-SNE: {metrics_path_parametric}")
print(f"- Original t-SNE: {metrics_path_original}")