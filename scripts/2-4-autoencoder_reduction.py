import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from utils.common import load_iris_dataset, visualize_3d_scatter
# Import our evaluation utilities
from utils.evaluation import evaluate_dimensionality_reduction

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Use data as is without normalization
data_scaled = data

# Create autoencoder model
input_dim = data_scaled.shape[1]  # Number of features
encoding_dim = 2  # Dimension of encoded representation

# Encoder
input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(8, activation='relu')(input_layer)
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = tf.keras.layers.Dense(8, activation='relu')(encoded)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)

# Autoencoder model
autoencoder = tf.keras.models.Model(input_layer, decoded)
encoder = tf.keras.models.Model(input_layer, encoded)

# Compile model
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(data_scaled, data_scaled, 
                epochs=100, 
                batch_size=16, 
                shuffle=True, 
                verbose=1)

# Get encoded representation (2D)
data_encoded = encoder.predict(data_scaled)

# Get reconstructed data
data_reconstructed = autoencoder.predict(data_scaled)

# Evaluate the dimensionality reduction using our metrics
metrics = evaluate_dimensionality_reduction(
    X=data_scaled,
    X_reduced=data_encoded,
    y=target,
    X_reconstructed=data_reconstructed,
    n_neighbors=10
)

# Print the evaluation metrics
print("\nDimensionality Reduction Quality Metrics:")
print("-----------------------------------------")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Save metrics to a text file
metrics_path = '/workspaces/10944-seminar/images/2-4-autoencoder_metrics.txt'
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
    plt.scatter(data_encoded[target == i, 0], data_encoded[target == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel('Autoencoder Component 1')
plt.ylabel('Autoencoder Component 2')
plt.title('Autoencoder - Iris Dataset (2D Visualization)')
plt.legend(loc='best')
plt.grid(True)

# Save the 2D visualization
output_path_2d = "/workspaces/10944-seminar/images/2-4-autoencoder_reduction_2d.png"
plt.savefig(output_path_2d)
plt.close()

# Create a bar plot of the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('Autoencoder Dimensionality Reduction Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/2-4-autoencoder_metrics.png', dpi=300)
plt.close()

# Add zero column for 3D visualization
data_encoded_3d = np.column_stack((data_encoded, np.zeros(len(data_encoded))))

# Visualize autoencoder encoded data in 3D
output_path = "/workspaces/10944-seminar/images/2-4-autoencoder_reduction.png"
visualize_3d_scatter(
    data=data_encoded_3d, 
    target=target,
    title="Autoencoder - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)

print(f"Script execution completed. Output images saved in the /workspaces/10944-seminar/images/ directory.")
print(f"Evaluation metrics saved to {metrics_path}")