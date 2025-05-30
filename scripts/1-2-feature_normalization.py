import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Per-feature normalization (each feature scaled to unit norm)
data_norm_feature = normalize(data, norm='l2', axis=0)

# Visualize per-feature normalized data in 3D
output_path = "/workspaces/10944-seminar/images/1-5-per_feature_normalization.png"
visualize_3d_scatter(
    data=data_norm_feature,
    target=target,
    title="Per-Feature Normalized Iris Dataset (3D Visualization)",
    save_path=output_path
)