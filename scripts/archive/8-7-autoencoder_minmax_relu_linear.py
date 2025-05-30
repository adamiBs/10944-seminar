import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
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

# Create normalized version with MinMax scaling
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Architecture parameters
input_dim = X_train.shape[1]  # Number of features in the digits dataset (64)
latent_dim = 2  # 2D latent space for direct visualization
intermediate_dim = 32  # Size of intermediate layers

# Define the autoencoder model with ReLU activation and linear output
def create_autoencoder(input_dim, latent_dim, intermediate_dim):
    # Encoder
    inputs = Input(shape=(input_dim,))
    
    x = Dense(intermediate_dim * 2)(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(intermediate_dim)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    # Latent space representation - linear activation for latent space
    latent = Dense(latent_dim, name='latent_space')(x)
    
    # Decoder
    x = Dense(intermediate_dim)(latent)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
    x = Dense(intermediate_dim * 2)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    
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
plt.title('Autoencoder Training Loss (MinMax, ReLU, Linear Output)')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-7-autoencoder_minmax_relu_linear_loss.png', dpi=300)
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
plt.title('Autoencoder Latent Space\nMinMax + ReLU + Linear Output', fontsize=14)
plt.xlabel('Latent Dimension 1', fontsize=12)
plt.ylabel('Latent Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-7-autoencoder_minmax_relu_linear.png', dpi=300)
plt.close()

# Compute reconstruction loss on test set
reconstructed = autoencoder.predict(X_test_normalized)
mse = np.mean(np.square(X_test_normalized - reconstructed))
print(f"Test reconstruction MSE: {mse:.6f}")

print("Script execution completed. Output images saved in the /workspaces/10944-seminar/images/ directory.")

"""
Comments on performance implications:

1. MinMax normalization scales the data to the range [0,1], but using a linear output
   activation allows the network to potentially generate values outside this range.
   
2. This mismatch between input range (constrained to [0,1]) and output range (unconstrained)
   may lead to higher reconstruction error if the model learns to produce values outside
   the valid input range.
   
3. Standard ReLU activation provides stable training with MinMax-scaled input since all values
   are positive, avoiding the "dying ReLU" problem commonly associated with negative inputs.
   
4. The linear output layer allows for unconstrained predictions which might be beneficial
   for capturing subtle variations in the data, but the autoencoder would need to implicitly
   learn that valid output values should stay within the [0,1] range.
   
5. This configuration provides a balance of stability (from ReLU with strictly positive inputs)
   and flexibility (from the linear output), but may lead to less precise reconstructions
   compared to a sigmoid output that naturally enforces the [0,1] range.
"""