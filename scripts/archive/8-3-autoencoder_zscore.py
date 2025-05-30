import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load the digits dataset
digits = load_digits()
X = digits.data
y = digits.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create normalized version with Z-score scaling
scaler = StandardScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define the autoencoder architecture
def create_autoencoder(input_dim, encoding_dim):
    """Create an autoencoder model for dimensionality reduction"""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(input_layer)
    encoded = Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(encoded)
    encoded = Dense(encoding_dim, activation='linear', name='encoder_output')(encoded)
    
    # Decoder
    decoded = Dense(16, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(encoded)
    decoded = Dense(32, activation=tf.keras.layers.LeakyReLU(alpha=0.3))(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Autoencoder model (full)
    autoencoder = Model(input_layer, decoded)
    
    # Encoder model (for extracting the reduced dimensions)
    encoder = Model(input_layer, encoded)
    
    # Compile the model
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder, encoder

# Create and train the autoencoder model on normalized data
input_dim = X_train.shape[1]  # Number of features in the digits dataset (64)
encoding_dim = 2  # Reduce to 2 dimensions for visualization

# Create the models
autoencoder, encoder = create_autoencoder(input_dim, encoding_dim)

# Display model summary
print("Autoencoder Model Summary:")
autoencoder.summary()

# Train the autoencoder on normalized data
history = autoencoder.fit(
    X_train_normalized, X_train_normalized,
    epochs=200,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Ensure the images directory exists
os.makedirs('/workspaces/10944-seminar/images', exist_ok=True)

# Generate encoded representations
encoded_normalized = encoder.predict(X_test_normalized)  # Z-score normalized test data

# Create a figure for normalized data
plt.figure(figsize=(10, 8))
plt.scatter(encoded_normalized[:, 0], encoded_normalized[:, 1], c=y_test, 
            cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
cbar = plt.colorbar()
cbar.set_label('Digit Class')
plt.title('2D Autoencoder Representation (Z-score Normalized Input)', fontsize=14)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-3-autoencoder_zscore.png', dpi=300)
plt.close()

print("Script execution completed. Output image saved in the /workspaces/10944-seminar/images/ directory.")

