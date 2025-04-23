import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Normalize data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Generate t-SNE embedding as targets
tsne = TSNE(n_components=3, perplexity=30, random_state=42)
tsne_embedding = tsne.fit_transform(data_scaled)

# Create parametric t-SNE model
class ParametricTSNE(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(12, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(3)  # Output 3D embedding
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

# Visualize parametric t-SNE embeddings in 3D
output_path = "/workspaces/10944-seminar/images/2-6-parametric_tsne_reduction.png"
visualize_3d_scatter(
    data=parametric_tsne_embedding, 
    target=target,
    title="Parametric t-SNE - Iris Dataset (3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)