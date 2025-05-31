# This makes the utils directory a Python package

# Import evaluation functions for easy access
from .evaluation import (
    trustworthiness,
    continuity,
    silhouette,
    reconstruction_error,
    knn_preservation,
    evaluate_dimensionality_reduction
)

# Export all evaluation functions
__all__ = [
    'trustworthiness',
    'continuity',
    'silhouette',
    'reconstruction_error',
    'knn_preservation',
    'evaluate_dimensionality_reduction'
]