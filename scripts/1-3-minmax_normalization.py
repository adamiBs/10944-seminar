import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Min-max normalization (per feature)
scaler_mm = MinMaxScaler()
data_minmax = scaler_mm.fit_transform(data)

# Visualize min-max normalized data in 3D
output_path = "/workspaces/10944-seminar/images/1-3-minmax_normalization.png"
visualize_3d_scatter(
    data=data_minmax,
    target=target,
    title="Min-Max Normalized Iris Dataset (3D Visualization)",
    save_path=output_path
)