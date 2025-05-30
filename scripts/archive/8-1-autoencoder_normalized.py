import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
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

# Create normalized version with MinMax scaling
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Define the autoencoder architecture
def create_autoencoder(input_dim, encoding_dim):
    """Create an autoencoder model for dimensionality reduction with improved architecture"""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(64, activation='relu')(input_layer)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = Dense(encoding_dim, activation='linear', name='encoder_output')(encoded)
    
    # Decoder
    decoded = Dense(16, activation='relu')(encoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(input_dim, activation='sigmoid')(decoded)
    
    # Autoencoder model (full)
    autoencoder = Model(input_layer, decoded)
    
    # Encoder model (for extracting the reduced dimensions)
    encoder = Model(input_layer, encoded)
    
    # Compile the model with a lower learning rate for better convergence
    autoencoder.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    
    return autoencoder, encoder

# Create and train the autoencoder model on normalized data
input_dim = X_train.shape[1]  # Number of features in the digits dataset
encoding_dim = 2  # Reduce to 2 dimensions for visualization

# Create the models
autoencoder, encoder = create_autoencoder(input_dim, encoding_dim)

# Display model summary
print("Autoencoder Model Summary:")
autoencoder.summary()

# Train the autoencoder on normalized data
history = autoencoder.fit(
    X_train_normalized, X_train_normalized,  # Train with normalized data
    epochs=400,
    batch_size=64,
    shuffle=True,
    validation_split=0.1,
    verbose=1
)

# Plot the training and validation loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Ensure the images directory exists
os.makedirs('/workspaces/10944-seminar/images', exist_ok=True)

# Save the loss plot
plt.savefig('/workspaces/10944-seminar/images/8-1-autoencoder_loss.png')
plt.close()

# Generate encoded representations
encoded_normalized = encoder.predict(X_test_normalized)  # Normalized test data
encoded_raw = encoder.predict(X_test)  # Raw (unnormalized) test data

# Create a figure for normalized data
plt.figure(figsize=(10, 8))
plt.scatter(encoded_normalized[:, 0], encoded_normalized[:, 1], c=y_test, 
            cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
cbar = plt.colorbar()
cbar.set_label('Digit Class')
plt.title('2D Autoencoder Representation (MinMax Normalized Input)', fontsize=14)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-1-autoencoder_normalized.png', dpi=300)
plt.close()

# Create a figure for unnormalized data
plt.figure(figsize=(10, 8))
plt.scatter(encoded_raw[:, 0], encoded_raw[:, 1], c=y_test, 
            cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
cbar = plt.colorbar()
cbar.set_label('Digit Class')
plt.title('2D Autoencoder Representation (Unnormalized Input)', fontsize=14)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-1-autoencoder_unnormalized.png', dpi=300)
plt.close()

# For comparison - side by side visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot normalized data
scatter1 = axes[0].scatter(encoded_normalized[:, 0], encoded_normalized[:, 1], c=y_test, 
                          cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
axes[0].set_title('MinMax Normalized Input', fontsize=14)
axes[0].set_xlabel('Dimension 1', fontsize=12)
axes[0].set_ylabel('Dimension 2', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Plot unnormalized data
scatter2 = axes[1].scatter(encoded_raw[:, 0], encoded_raw[:, 1], c=y_test, 
                          cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
axes[1].set_title('Unnormalized Input', fontsize=14)
axes[1].set_xlabel('Dimension 1', fontsize=12)
axes[1].set_ylabel('Dimension 2', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

# Add a colorbar
cbar = fig.colorbar(scatter1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_label('Digit Class', fontsize=12)

plt.suptitle('Autoencoder 2D Embedding Comparison: With and Without Normalization', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('/workspaces/10944-seminar/images/8-1-autoencoder_normalized_comparison.png', dpi=300)
plt.close()

# Evaluate the reconstruction error
norm_reconstruction = autoencoder.predict(X_test_normalized)
raw_reconstruction = autoencoder.predict(X_test)

norm_mse = np.mean(np.square(X_test_normalized - norm_reconstruction))
raw_mse = np.mean(np.square(X_test - raw_reconstruction))

print(f"Reconstruction MSE with normalized data: {norm_mse:.6f}")
print(f"Reconstruction MSE with raw data: {raw_mse:.6f}")

# Visualization of original vs reconstructed digits
n = 10  # Number of digits to display
plt.figure(figsize=(20, 6))

for i in range(n):
    # Original Normalized Image
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test_normalized[i].reshape(8, 8), cmap='gray')
    plt.title(f"Original {y_test[i]}")
    plt.axis('off')
    
    # Reconstructed Normalized Image
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(norm_reconstruction[i].reshape(8, 8), cmap='gray')
    plt.title(f"Normalized Recon")
    plt.axis('off')
    
    # Reconstructed Raw Image
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(raw_reconstruction[i].reshape(8, 8), cmap='gray')
    plt.title(f"Raw Recon")
    plt.axis('off')

plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-1-autoencoder_reconstruction.png', dpi=300)
plt.close()

print("Script execution completed. Output images saved in the /workspaces/10944-seminar/images/ directory.")

