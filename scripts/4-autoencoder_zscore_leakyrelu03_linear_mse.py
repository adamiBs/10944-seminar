import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
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

# Architecture parameters
input_dim = X_train.shape[1]  # Number of features in the digits dataset (64)
latent_dim = 2  # 2D latent space for direct visualization
intermediate_dim = 32  # Size of intermediate layers

# Define the autoencoder model with LeakyReLU(0.3) activation and linear output
def create_autoencoder(input_dim, latent_dim, intermediate_dim):
    # Encoder
    inputs = Input(shape=(input_dim,))
    
    x = Dense(intermediate_dim * 2)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(intermediate_dim)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    # Latent space representation - linear activation for latent space
    latent = Dense(latent_dim, name='latent_space')(x)
    
    # Decoder
    x = Dense(intermediate_dim)(latent)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    x = Dense(intermediate_dim * 2)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    # Output layer with linear activation
    outputs = Dense(input_dim, activation='linear')(x)
    
    # Create models
    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, latent)
    
    # Compile the model with MSE loss
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder, encoder

# Create model
autoencoder, encoder = create_autoencoder(input_dim, latent_dim, intermediate_dim)

# Display model summary
print("Autoencoder Model Summary:")
autoencoder.summary()

# Define callbacks for better training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                              min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, 
                               restore_best_weights=True, verbose=1)

# Train the autoencoder
history = autoencoder.fit(
    X_train_normalized, X_train_normalized,  # Input equals target for autoencoders
    epochs=200,
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    verbose=1,
    callbacks=[reduce_lr, early_stopping]
)

# Ensure the images directory exists
os.makedirs('/workspaces/10944-seminar/images', exist_ok=True)

# Plot the loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Training Loss (Z-score, LeakyReLU(0.3), Linear Output, MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/4-autoencoder_zscore_leakyrelu03_linear_mse_loss.png', dpi=300)
plt.close()

# Generate latent space embeddings for test data
latent_embeddings = encoder.predict(X_test_normalized)

# Create a figure for latent space visualization
plt.figure(figsize=(10, 8))
scatter = plt.scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], 
                     c=y_test, cmap='tab10', alpha=0.8, edgecolors='w', 
                     linewidths=0.5, s=60)
cbar = plt.colorbar()
cbar.set_label('Digit Class')
plt.title('Latent Space: Z-score + LeakyReLU(0.3) + Linear + MSE', fontsize=14)
plt.xlabel('Latent Dimension 1', fontsize=12)
plt.ylabel('Latent Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/4-autoencoder_zscore_leakyrelu03_linear_mse.png', dpi=300)
plt.close()

# Compute reconstruction loss on test set
reconstructed = autoencoder.predict(X_test_normalized)
mse = np.mean(np.square(X_test_normalized - reconstructed))
print(f"Test reconstruction MSE: {mse:.6f}")

# Visualize some reconstructed digits
n_samples = 10
sample_indices = np.random.choice(X_test_normalized.shape[0], n_samples, replace=False)
test_samples = X_test_normalized[sample_indices]
reconstructed_samples = autoencoder.predict(test_samples)

plt.figure(figsize=(15, 4))
for i in range(n_samples):
    # Original image
    plt.subplot(2, n_samples, i + 1)
    plt.imshow(test_samples[i].reshape(8, 8), cmap='gray')
    if i == 0:
        plt.title("Original", fontsize=16)
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(2, n_samples, i + 1 + n_samples)
    plt.imshow(reconstructed_samples[i].reshape(8, 8), cmap='gray')
    if i == 0:
        plt.title("Reconstructed", fontsize=16)
    plt.axis('off')
    
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/4-autoencoder_zscore_leakyrelu03_linear_mse_samples.png', dpi=300)
plt.close()

print("Script execution completed. Output images saved in the /workspaces/10944-seminar/images/ directory.")

"""
Comments on performance implications:

1. Z-score normalization with LeakyReLU(0.3) creates a good synergy
   - Z-score data is centered around 0 with approximately 50% negative values
   - LeakyReLU with higher alpha (0.3) preserves more information from negative inputs
   - This combination allows better feature extraction compared to standard ReLU
   - The higher alpha value (0.3) allows 30% of the negative activation to pass through
   
2. Linear output activation perfectly matches Z-score's distribution
   - Z-score data has unbounded range (typically within [-3,3] but potentially wider)
   - Linear output allows reconstructions across the full range of the input data
   - No artificial constraints are placed on the output values
   - This creates a natural alignment between input and output distributions
   
3. MSE loss with linear output provides direct optimization for Z-score data
   - MSE directly minimizes the squared Euclidean distance between inputs and reconstructions
   - With matching distributions, the loss function can directly optimize for exact values
   - The gradients flow naturally without any distortion from activation constraints
   
4. This configuration is theoretically optimal for Z-score data
   - The higher LeakyReLU alpha value (0.3) allows better information flow
   - Linear output matches the Z-score data distribution
   - MSE loss directly optimizes for reconstruction accuracy
   - Should produce accurate reconstructions and well-organized latent space
   - Likely to be among the best performing configurations for Z-score normalized data
"""