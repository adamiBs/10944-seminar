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

# Define the autoencoder model with LeakyReLU(0.3) activation and sigmoid output
def create_autoencoder(input_dim, latent_dim, intermediate_dim, loss_type='mse'):
    # Encoder
    inputs = Input(shape=(input_dim,))
    
    x = Dense(intermediate_dim * 2)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(intermediate_dim)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    # Latent space representation
    latent = Dense(latent_dim, name='latent_space')(x)
    
    # Decoder
    x = Dense(intermediate_dim)(latent)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    x = Dense(intermediate_dim * 2)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    # Output layer - using sigmoid activation for both loss functions to ensure fair comparison
    outputs = Dense(input_dim, activation='sigmoid')(x)
    
    # Create models
    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, latent)
    
    # Compile the model with specified loss
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss=loss_type)
    
    return autoencoder, encoder

# Function to train the model and return history and models
def train_model(loss_type):
    print(f"\nTraining model with {loss_type} loss...")
    autoencoder, encoder = create_autoencoder(input_dim, latent_dim, intermediate_dim, loss_type)
    
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
    
    return history, autoencoder, encoder

# Ensure the images directory exists
os.makedirs('/workspaces/10944-seminar/images', exist_ok=True)

# Train models with different loss functions
mse_history, mse_autoencoder, mse_encoder = train_model('mse')
bce_history, bce_autoencoder, bce_encoder = train_model('binary_crossentropy')

# Plot the training histories together
plt.figure(figsize=(12, 5))

# Plot MSE loss
plt.subplot(1, 2, 1)
plt.plot(mse_history.history['loss'], label='Training Loss')
plt.plot(mse_history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder with MSE Loss\n(Z-score, LeakyReLU, Sigmoid Out)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)

# Plot BCE loss
plt.subplot(1, 2, 2)
plt.plot(bce_history.history['loss'], label='Training Loss')
plt.plot(bce_history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder with Binary Cross-Entropy Loss\n(Z-score, LeakyReLU, Sigmoid Out)')
plt.xlabel('Epochs')
plt.ylabel('Loss (Binary Cross-Entropy)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-8-autoencoder_loss_comparison_zscore_training.png', dpi=300)
plt.close()

# Function to generate latent space visualization
def visualize_latent_space(encoder, loss_type):
    latent_embeddings = encoder.predict(X_test_normalized)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_embeddings[:, 0], latent_embeddings[:, 1], 
                        c=y_test, cmap='tab10', alpha=0.8, edgecolors='w', 
                        linewidths=0.5, s=60)
    cbar = plt.colorbar()
    cbar.set_label('Digit Class')
    plt.title(f'Autoencoder Latent Space\nZ-score + LeakyReLU + {loss_type.upper()} Loss', fontsize=14)
    plt.xlabel('Latent Dimension 1', fontsize=12)
    plt.ylabel('Latent Dimension 2', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'/workspaces/10944-seminar/images/8-8-autoencoder_zscore_latent_{loss_type}.png', dpi=300)
    plt.close()

# Visualize latent spaces
visualize_latent_space(mse_encoder, 'mse')
visualize_latent_space(bce_encoder, 'bce')

# Compare reconstructions from both models
def compare_reconstructions():
    # Get some test samples
    n_samples = 10
    sample_indices = np.random.choice(X_test_normalized.shape[0], n_samples, replace=False)
    test_samples = X_test_normalized[sample_indices]
    
    # Get reconstructions
    mse_reconstructed = mse_autoencoder.predict(test_samples)
    bce_reconstructed = bce_autoencoder.predict(test_samples)
    
    # Calculate reconstruction errors
    mse_error = np.mean(np.square(test_samples - mse_reconstructed))
    bce_error = np.mean(np.square(test_samples - bce_reconstructed))
    
    print(f"MSE model - Reconstruction MSE: {mse_error:.6f}")
    print(f"BCE model - Reconstruction MSE: {bce_error:.6f}")
    
    # Visualization of original vs reconstructed digits
    plt.figure(figsize=(15, 8))
    
    for i in range(n_samples):
        # Original image
        plt.subplot(3, n_samples, i + 1)
        plt.imshow(test_samples[i].reshape(8, 8), cmap='gray')
        if i == 0:
            plt.title("Original", fontsize=16)
        plt.axis('off')
        
        # MSE reconstruction
        plt.subplot(3, n_samples, i + 1 + n_samples)
        plt.imshow(mse_reconstructed[i].reshape(8, 8), cmap='gray')
        if i == 0:
            plt.title("MSE Reconstruction", fontsize=16)
        plt.axis('off')
        
        # BCE reconstruction
        plt.subplot(3, n_samples, i + 1 + 2*n_samples)
        plt.imshow(bce_reconstructed[i].reshape(8, 8), cmap='gray')
        if i == 0:
            plt.title("BCE Reconstruction", fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/workspaces/10944-seminar/images/8-8-autoencoder_zscore_reconstruction_comparison.png', dpi=300)
    plt.close()

# Compare reconstructions
compare_reconstructions()

print("Script execution completed. All comparison images saved in the /workspaces/10944-seminar/images/ directory.")

"""
Comments on performance implications:

1. MSE Loss vs Binary Cross-Entropy Loss:

   - MSE Loss: 
     * Measures the squared difference between pixel values
     * More sensitive to outliers due to the squared term
     * Generally works well when the distribution of values is Gaussian
     * Typically favors reconstructions that are smoother and may lose some details

   - Binary Cross-Entropy Loss:
     * Treats each pixel as an independent binary decision problem
     * Better suited for data with values between 0 and 1 (like normalized pixels)
     * Often preserves sharper edges and more details in reconstructions
     * May emphasize contrast over smooth gradients
   
2. Using Z-score normalization with binary cross-entropy creates an interesting situation:
   
   - Z-score data is centered around zero with potentially large positive and negative values
   - But sigmoid output compresses this to [0,1] range
   - This compression may cause information loss, but BCE loss is optimized for [0,1] outputs
   - The network must adapt by learning a more effective compression/decompression strategy

3. The comparison allows us to see how the choice of loss function affects:
   - Convergence speed and stability
   - Clustering properties in the latent space
   - Reconstruction quality and detail preservation
"""