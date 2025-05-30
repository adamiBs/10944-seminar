import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
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

# Define the parametric t-SNE model
def create_parametric_tsne(input_dim, embedding_dim):
    """Create a parametric t-SNE model for dimensionality reduction"""
    input_layer = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(input_layer)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    embedding = Dense(embedding_dim, activation='linear', name='embedding')(x)
    
    model = Model(input_layer, embedding)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# First generate standard t-SNE embeddings to serve as targets for the parametric model
print("Generating t-SNE embeddings for training targets...")
standard_tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_embedding_normalized = standard_tsne.fit_transform(X_train_normalized)

# Create parametric t-SNE model
input_dim = X_train.shape[1]  # Number of features in the digits dataset
embedding_dim = 2  # Reduce to 2 dimensions for visualization

# Create the model
parametric_tsne = create_parametric_tsne(input_dim, embedding_dim)

# Display model summary
print("Parametric t-SNE Model Summary:")
parametric_tsne.summary()

# Train the parametric t-SNE model on normalized data
print("Training parametric t-SNE model...")
X_train_tf = tf.convert_to_tensor(X_train_normalized, dtype=tf.float32)
y_train_tf = tf.convert_to_tensor(tsne_embedding_normalized, dtype=tf.float32)

history = parametric_tsne.fit(
    X_train_tf, y_train_tf,
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
plt.title('Parametric t-SNE Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# Ensure the images directory exists
os.makedirs('/workspaces/10944-seminar/images', exist_ok=True)

# Save the loss plot
plt.savefig('/workspaces/10944-seminar/images/8-2-parametric_tsne_loss.png')
plt.close()

# Generate embeddings for normalized and raw test data
embedded_normalized = parametric_tsne.predict(X_test_normalized)
embedded_raw = parametric_tsne.predict(X_test)

# Create a figure for normalized data
plt.figure(figsize=(10, 8))
plt.scatter(embedded_normalized[:, 0], embedded_normalized[:, 1], c=y_test, 
            cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
cbar = plt.colorbar()
cbar.set_label('Digit Class')
plt.title('2D Parametric t-SNE Representation (MinMax Normalized Input)', fontsize=14)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-2-parametric_tsne_normalized.png', dpi=300)
plt.close()

# Create a figure for unnormalized data
plt.figure(figsize=(10, 8))
plt.scatter(embedded_raw[:, 0], embedded_raw[:, 1], c=y_test, 
            cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
cbar = plt.colorbar()
cbar.set_label('Digit Class')
plt.title('2D Parametric t-SNE Representation (Unnormalized Input)', fontsize=14)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-2-parametric_tsne_unnormalized.png', dpi=300)
plt.close()

# For comparison - side by side visualization
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot normalized data
scatter1 = axes[0].scatter(embedded_normalized[:, 0], embedded_normalized[:, 1], c=y_test, 
                          cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
axes[0].set_title('MinMax Normalized Input', fontsize=14)
axes[0].set_xlabel('Dimension 1', fontsize=12)
axes[0].set_ylabel('Dimension 2', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Plot unnormalized data
scatter2 = axes[1].scatter(embedded_raw[:, 0], embedded_raw[:, 1], c=y_test, 
                          cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
axes[1].set_title('Unnormalized Input', fontsize=14)
axes[1].set_xlabel('Dimension 1', fontsize=12)
axes[1].set_ylabel('Dimension 2', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

# Add a colorbar
cbar = fig.colorbar(scatter1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_label('Digit Class', fontsize=12)

plt.suptitle('Parametric t-SNE 2D Embedding Comparison: With and Without Normalization', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('/workspaces/10944-seminar/images/8-2-parametric_tsne_normalized_comparison.png', dpi=300)
plt.close()

# Compare with standard t-SNE for normalized data
print("Generating standard t-SNE embedding for test data comparison...")
standard_tsne_test = TSNE(n_components=2, perplexity=30, random_state=42)
standard_tsne_embedding = standard_tsne_test.fit_transform(X_test_normalized)

# Comparison between standard t-SNE and parametric t-SNE
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot standard t-SNE 
scatter1 = axes[0].scatter(standard_tsne_embedding[:, 0], standard_tsne_embedding[:, 1], c=y_test, 
                          cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
axes[0].set_title('Standard t-SNE', fontsize=14)
axes[0].set_xlabel('Dimension 1', fontsize=12)
axes[0].set_ylabel('Dimension 2', fontsize=12)
axes[0].grid(True, linestyle='--', alpha=0.7)

# Plot parametric t-SNE
scatter2 = axes[1].scatter(embedded_normalized[:, 0], embedded_normalized[:, 1], c=y_test, 
                          cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
axes[1].set_title('Parametric t-SNE', fontsize=14)
axes[1].set_xlabel('Dimension 1', fontsize=12)
axes[1].set_ylabel('Dimension 2', fontsize=12)
axes[1].grid(True, linestyle='--', alpha=0.7)

# Add a colorbar
cbar = fig.colorbar(scatter1, ax=axes.ravel().tolist(), fraction=0.02, pad=0.04)
cbar.set_label('Digit Class', fontsize=12)

plt.suptitle('Standard vs Parametric t-SNE Comparison (Normalized Data)', fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig('/workspaces/10944-seminar/images/8-2-parametric_tsne_vs_standard.png', dpi=300)
plt.close()

# Calculate embedding quality metrics
param_tsne_distances = np.square(embedded_normalized[:, np.newaxis] - embedded_normalized).sum(axis=2)
standard_tsne_distances = np.square(standard_tsne_embedding[:, np.newaxis] - standard_tsne_embedding).sum(axis=2)

# Calculate Pearson correlation between distance matrices
flattened_param = param_tsne_distances[np.triu_indices(param_tsne_distances.shape[0], k=1)]
flattened_standard = standard_tsne_distances[np.triu_indices(standard_tsne_distances.shape[0], k=1)]
correlation = np.corrcoef(flattened_param, flattened_standard)[0, 1]

print(f"Correlation between standard t-SNE and parametric t-SNE distances: {correlation:.4f}")
print("Script execution completed. Output images saved in the /workspaces/10944-seminar/images/ directory.")

