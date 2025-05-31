import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

def trustworthiness(X, X_reduced, n_neighbors=10):
    """
    Calculate the trustworthiness metric.
    
    Trustworthiness measures how well local structures are preserved in the reduced space.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Original high-dimensional data
    X_reduced : array-like, shape (n_samples, n_components)
        Low-dimensional embedding of X
    n_neighbors : int, default=10
        Number of neighbors to consider when calculating trustworthiness
        
    Returns:
    --------
    float : trustworthiness score (between 0 and 1)
    """
    n_samples = X.shape[0]
    
    # Find nearest neighbors in the original space
    knn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn_orig.fit(X)
    _, orig_neighbors = knn_orig.kneighbors()
    orig_neighbors = orig_neighbors[:, 1:]  # Remove self
    
    # Find nearest neighbors in the embedded space
    knn_embedded = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn_embedded.fit(X_reduced)
    _, embedded_neighbors = knn_embedded.kneighbors()
    embedded_neighbors = embedded_neighbors[:, 1:]  # Remove self
    
    trustworthiness_sum = 0
    for i in range(n_samples):
        # Calculate rank of embedded neighbors in original space
        for j in embedded_neighbors[i]:
            # If j is not in the original neighborhood
            if j not in orig_neighbors[i]:
                # Find the rank of j in original space
                rank = np.where(knn_orig.kneighbors(X[j].reshape(1, -1), n_neighbors=n_samples)[1][0] == i)[0][0] + 1
                trustworthiness_sum += (rank - n_neighbors)
    
    # Normalize
    norm = 2 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
    return 1 - norm * trustworthiness_sum

def continuity(X, X_reduced, n_neighbors=10):
    """
    Calculate the continuity metric.
    
    Continuity complements trustworthiness by measuring how well original 
    neighborhoods are reconstructed in the reduced space.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Original high-dimensional data
    X_reduced : array-like, shape (n_samples, n_components)
        Low-dimensional embedding of X
    n_neighbors : int, default=10
        Number of neighbors to consider when calculating continuity
        
    Returns:
    --------
    float : continuity score (between 0 and 1)
    """
    n_samples = X.shape[0]
    
    # Find nearest neighbors in the original space
    knn_orig = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn_orig.fit(X)
    _, orig_neighbors = knn_orig.kneighbors()
    orig_neighbors = orig_neighbors[:, 1:]  # Remove self
    
    # Find nearest neighbors in the embedded space
    knn_embedded = NearestNeighbors(n_neighbors=n_neighbors + 1)
    knn_embedded.fit(X_reduced)
    _, embedded_neighbors = knn_embedded.kneighbors()
    embedded_neighbors = embedded_neighbors[:, 1:]  # Remove self
    
    continuity_sum = 0
    for i in range(n_samples):
        # Calculate rank of original neighbors in embedded space
        for j in orig_neighbors[i]:
            # If j is not in the embedded neighborhood
            if j not in embedded_neighbors[i]:
                # Find the rank of j in embedded space
                rank = np.where(knn_embedded.kneighbors(X_reduced[j].reshape(1, -1), n_neighbors=n_samples)[1][0] == i)[0][0] + 1
                continuity_sum += (rank - n_neighbors)
    
    # Normalize
    norm = 2 / (n_samples * n_neighbors * (2 * n_samples - 3 * n_neighbors - 1))
    return 1 - norm * continuity_sum

def silhouette(X_reduced, labels):
    """
    Calculate the silhouette score for the reduced data.
    
    Silhouette score evaluates how well-separated the clusters are in the reduced space.
    
    Parameters:
    -----------
    X_reduced : array-like, shape (n_samples, n_components)
        Low-dimensional embedding of data
    labels : array-like, shape (n_samples,)
        Cluster labels for each sample
        
    Returns:
    --------
    float : silhouette score (between -1 and 1)
    """
    if len(np.unique(labels)) <= 1:
        return 0  # Silhouette score undefined for single cluster
    
    return silhouette_score(X_reduced, labels)

def reconstruction_error(X, X_reconstructed):
    """
    Calculate the reconstruction error (MSE).
    
    Measures how well a model like an autoencoder can reconstruct the original data.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Original data
    X_reconstructed : array-like, shape (n_samples, n_features)
        Reconstructed data
        
    Returns:
    --------
    float : Mean squared error between original and reconstructed data
    """
    return np.mean(np.square(X - X_reconstructed))

def knn_preservation(X, X_reduced, k=10):
    """
    Measure how many k-nearest neighbors are preserved between original and reduced space.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Original high-dimensional data
    X_reduced : array-like, shape (n_samples, n_components)
        Low-dimensional embedding of X
    k : int, default=10
        Number of nearest neighbors to consider
        
    Returns:
    --------
    float : KNN preservation score (between 0 and 1)
    """
    n_samples = X.shape[0]
    
    # Find k-nearest neighbors in the original space
    knn_orig = NearestNeighbors(n_neighbors=k+1)
    knn_orig.fit(X)
    _, orig_neighbors = knn_orig.kneighbors()
    orig_neighbors = orig_neighbors[:, 1:]  # Remove self
    
    # Find k-nearest neighbors in the embedded space
    knn_embedded = NearestNeighbors(n_neighbors=k+1)
    knn_embedded.fit(X_reduced)
    _, embedded_neighbors = knn_embedded.kneighbors()
    embedded_neighbors = embedded_neighbors[:, 1:]  # Remove self
    
    # Count preserved neighbors
    preserved = 0
    for i in range(n_samples):
        preserved += len(set(orig_neighbors[i]).intersection(set(embedded_neighbors[i])))
    
    # Normalize by total possible preserved neighbors
    return preserved / (n_samples * k)

def evaluate_dimensionality_reduction(X, X_reduced, y=None, X_reconstructed=None, n_neighbors=10):
    """
    Evaluate dimensionality reduction using multiple metrics.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        Original high-dimensional data
    X_reduced : array-like, shape (n_samples, n_components)
        Low-dimensional embedding of X
    y : array-like, shape (n_samples,), optional
        Labels for the data (for silhouette score)
    X_reconstructed : array-like, shape (n_samples, n_features), optional
        Reconstructed data (for autoencoders)
    n_neighbors : int, default=10
        Number of neighbors to consider for neighborhood-based metrics
        
    Returns:
    --------
    dict : Dictionary containing all evaluation metrics
    """
    results = {}
    
    # Calculate trustworthiness and continuity
    results['trustworthiness'] = trustworthiness(X, X_reduced, n_neighbors)
    results['continuity'] = continuity(X, X_reduced, n_neighbors)
    results['knn_preservation'] = knn_preservation(X, X_reduced, n_neighbors)
    
    # Calculate silhouette score if labels are provided
    if y is not None:
        results['silhouette_score'] = silhouette(X_reduced, y)
    
    # Calculate reconstruction error if reconstructed data is provided
    if X_reconstructed is not None:
        results['reconstruction_error'] = reconstruction_error(X, X_reconstructed)
    
    return results