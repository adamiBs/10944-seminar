import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2
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

# First, get t-SNE embeddings on the training data to use as target values
print("Computing t-SNE embeddings for training targets...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_train_tsne = tsne.fit_transform(X_train_normalized)

# Define the parametric t-SNE network
def create_parametric_tsne(input_dim, output_dim=2):
    """Create a neural network that approximates t-SNE mapping"""
    inputs = Input(shape=(input_dim,))
    
    # Network with more capacity for learning complex t-SNE mapping
    x = Dense(256, kernel_regularizer=l2(1e-4))(inputs)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(128, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    x = Dropout(0.2)(x)
    
    x = Dense(64, kernel_regularizer=l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = tf.keras.layers.LeakyReLU(0.2)(x)
    
    # Output layer with linear activation for t-SNE coordinates
    outputs = Dense(output_dim, activation='linear', name='tsne_embedding')(x)
    
    # Create model
    model = Model(inputs, outputs)
    
    # Compile the model with MSE loss to learn the t-SNE embedding
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return model

# Create model
input_dim = X_train.shape[1]  # Number of features in the digits dataset (64)
output_dim = 2  # 2D embeddings

# Create the parametric t-SNE model
parametric_tsne = create_parametric_tsne(input_dim, output_dim)

# Display model summary
print("Parametric t-SNE Model Summary:")
parametric_tsne.summary()

# Define callbacks for better training
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, 
                             min_lr=1e-6, verbose=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, 
                              restore_best_weights=True, verbose=1)

# Train the model to learn the mapping from data to t-SNE coordinates
history = parametric_tsne.fit(
    X_train_normalized, X_train_tsne,
    epochs=200,
    batch_size=64,
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
plt.title('Parametric t-SNE Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-5-parametric_tsne_loss.png', dpi=300)
plt.close()

# Generate embeddings for test data
test_embeddings = parametric_tsne.predict(X_test_normalized)

# Create a figure for embeddings
plt.figure(figsize=(10, 8))
plt.scatter(test_embeddings[:, 0], test_embeddings[:, 1], c=y_test, 
            cmap='tab10', alpha=0.8, edgecolors='w', linewidths=0.5, s=60)
cbar = plt.colorbar()
cbar.set_label('Digit Class')
plt.title('Parametric t-SNE (Z-score Normalized Input)', fontsize=14)
plt.xlabel('Dimension 1', fontsize=12)
plt.ylabel('Dimension 2', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/8-5-parametric_tsne_zscore.png', dpi=300)
plt.close()

print("Script execution completed. Output image saved in the /workspaces/10944-seminar/images/ directory.")

