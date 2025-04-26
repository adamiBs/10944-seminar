import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Use data as is without normalization
data_scaled = data


# Define augmentation function
def augment(x):
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=0.1)
    return x + noise

# Create contrastive learning model
class ContrastiveModel(tf.keras.Model):
    def __init__(self, embedding_size=2):
        super().__init__()
        self.embedding_size = embedding_size
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(embedding_size)
        ])
        self.projection = tf.keras.Sequential([
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(4)
        ])
        
    def call(self, inputs):
        embeddings = self.encoder(inputs)
        projections = self.projection(embeddings)
        return embeddings, projections

# Define contrastive loss function
def contrastive_loss(projections_1, projections_2, temperature=0.1):
    # Normalize projections
    projections_1 = tf.math.l2_normalize(projections_1, axis=1)
    projections_2 = tf.math.l2_normalize(projections_2, axis=1)
    
    # Similarity matrix
    batch_size = tf.shape(projections_1)[0]
    similarities = tf.matmul(projections_1, projections_2, transpose_b=True) / temperature
    
    # Labels: positives are on the diagonal
    labels = tf.range(batch_size)
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
        labels, similarities, from_logits=True))
    
    return loss

# Training setup
model = ContrastiveModel(embedding_size=2)
optimizer = tf.keras.optimizers.Adam(0.001)
batch_size = 32

# Training loop with progress reporting
X_train_tensor = tf.convert_to_tensor(data_scaled, dtype=tf.float32)
epochs = 100

for epoch in range(epochs):
    # Create augmented versions of the dataset
    aug1 = augment(X_train_tensor)
    aug2 = augment(X_train_tensor)
    
    with tf.GradientTape() as tape:
        # Forward pass
        _, proj1 = model(aug1)
        _, proj2 = model(aug2)
        
        # Calculate loss
        loss = contrastive_loss(proj1, proj2)
    
    # Backpropagation
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}")

# Get embeddings for visualization
embeddings, _ = model(tf.convert_to_tensor(data_scaled, dtype=tf.float32))
data_contrastive = embeddings.numpy()

# Add zero column for 3D visualization
data_contrastive_3d = np.column_stack((data_contrastive, np.zeros(len(data_contrastive))))

# Visualize contrastive learning embeddings in 3D
output_path = "/workspaces/10944-seminar/images/2-5-contrastive_reduction.png"
visualize_3d_scatter(
    data=data_contrastive_3d, 
    target=target,
    title="Contrastive Learning - Iris Dataset (2D → 3D Visualization)",
    save_path=output_path,
    features_to_use=[0, 1, 2]
)