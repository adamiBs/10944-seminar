import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Normalize data for neural network
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create autoencoder model
input_dim = data_scaled.shape[1]  # Number of features
encoding_dim = 3  # Dimension of encoded representation

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

# Get encoded representation (3D)
data_encoded = encoder.predict(data_scaled)

# Visualize autoencoder encoded data in 3D
output_path = "/workspaces/10944-seminar/images/2-4-autoencoder_reduction.png"
visualize_3d_scatter(
    data=data_encoded, 
    target=target,
    title="Autoencoder - Iris Dataset (3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]  # Use all 3 encoded dimensions
)