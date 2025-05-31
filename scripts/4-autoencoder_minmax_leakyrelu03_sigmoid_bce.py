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

# Import our new evaluation utilities
from utils.evaluation import evaluate_dimensionality_reduction

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

# Define the autoencoder model with LeakyReLU(0.3) activation and sigmoid output
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
    
    # Latent space representation
    latent = Dense(latent_dim, name='latent_space')(x)
    
    # Decoder
    x = Dense(intermediate_dim)(latent)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    x = Dense(intermediate_dim * 2)(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.3)(x)
    
    # Output layer with sigmoid activation
    outputs = Dense(input_dim, activation='sigmoid')(x)
    
    # Create models
    autoencoder = Model(inputs, outputs)
    encoder = Model(inputs, latent)
    
    # Compile the model with Binary Crossentropy loss
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    
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
plt.title('Autoencoder Training Loss (MinMax, LeakyReLU(0.3), Sigmoid Output, BCE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/4-autoencoder_minmax_leakyrelu03_sigmoid_bce_loss.png', dpi=300)
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
plt.title('Latent Space: MinMax + LeakyReLU(0.3) + Sigmoid + BCE', fontsize=14)
plt.xlabel('Latent Dimension 1', fontsize=12)
plt.ylabel('Latent Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/4-autoencoder_minmax_leakyrelu03_sigmoid_bce.png', dpi=300)
plt.close()

# Compute reconstructions
reconstructed = autoencoder.predict(X_test_normalized)
mse = np.mean(np.square(X_test_normalized - reconstructed))
print(f"Test reconstruction MSE: {mse:.6f}")

# Evaluate the dimensionality reduction using our new metrics
metrics = evaluate_dimensionality_reduction(
    X=X_test_normalized,
    X_reduced=latent_embeddings,
    y=y_test,
    X_reconstructed=reconstructed,
    n_neighbors=10
)

# Print the evaluation metrics
print("\nDimensionality Reduction Quality Metrics:")
print("-----------------------------------------")
for metric_name, metric_value in metrics.items():
    print(f"{metric_name}: {metric_value:.6f}")

# Save metrics to a text file
metrics_path = '/workspaces/10944-seminar/images/4-autoencoder_minmax_leakyrelu03_sigmoid_bce_metrics.txt'
with open(metrics_path, 'w') as f:
    f.write("Dimensionality Reduction Quality Metrics:\n")
    f.write("-----------------------------------------\n")
    for metric_name, metric_value in metrics.items():
        f.write(f"{metric_name}: {metric_value:.6f}\n")

# Create a bar plot of the metrics
plt.figure(figsize=(10, 6))
plt.bar(metrics.keys(), metrics.values())
plt.title('Dimensionality Reduction Quality Metrics')
plt.xlabel('Metric')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/4-autoencoder_minmax_leakyrelu03_sigmoid_bce_metrics.png', dpi=300)
plt.close()

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
plt.savefig('/workspaces/10944-seminar/images/4-autoencoder_minmax_leakyrelu03_sigmoid_bce_samples.png', dpi=300)
plt.close()

print("Script execution completed. Output images saved in the /workspaces/10944-seminar/images/ directory.")
print(f"Evaluation metrics saved to {metrics_path}")

"""
Comments on performance implications:

1. MinMax normalization with LeakyReLU(0.3) and BCE loss
   - MinMax normalization ensures all values are in [0,1] range
   - For positive-only data, LeakyReLU behaves similarly to standard ReLU
   - The alpha=0.3 parameter has minimal impact with MinMax normalized data
   
2. Sigmoid output with BCE loss creates a perfect statistical match
   - Both align with the [0,1] range of MinMax normalized data
   - BCE interprets outputs as probabilities, ideal for sigmoid activations
   - This combination is theoretically optimal for this type of data
   
3. The LeakyReLU may provide subtle differences versus standard ReLU
   - Different weight initialization dynamics
   - Different gradient flow in the network
   - May lead to slightly different latent space organization
   
4. This configuration is expected to produce:
   - Clean, sharp reconstructions due to BCE's focus on binary-like patterns
   - Well-structured latent space with good digit class separation
   - Faster convergence due to the statistical alignment of all components
   - Should be among the best performing configurations for MinMax data
   
5. Dimensionality Reduction Quality Metrics:
   - Trustworthiness: Measures how well local structures are preserved
   - Continuity: Measures how well original neighborhoods are reconstructed
   - KNN Preservation: Shows how many nearest neighbors are preserved 
   - Silhouette Score: Evaluates how well-separated the clusters are
   - Reconstruction Error: Measures fidelity of input reconstruction (MSE)
"""