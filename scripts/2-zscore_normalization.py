import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Z-score normalization (per feature)
scaler_z = StandardScaler()
data_zscore = scaler_z.fit_transform(data)

# Visualize Z-score normalized data in 3D
output_path = "/workspaces/10944-seminar/images/2-zscore_normalization.png"
visualize_3d_scatter(
    data=data_zscore,
    target=target,
    title="Z-score Normalized Iris Dataset (3D Visualization)",
    save_path=output_path
)