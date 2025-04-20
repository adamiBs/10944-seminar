import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Per-sample normalization (each sample vector scaled to unit norm)
data_norm_sample = normalize(data, norm='l2', axis=1)

# Visualize per-sample normalized data in 3D
output_path = "/workspaces/10944-seminar/images/5-2-per_sample_normalization.png"
visualize_3d_scatter(
    data=data_norm_sample,
    target=target,
    title="Per-Sample Normalized Iris Dataset (3D Visualization)",
    save_path=output_path
)