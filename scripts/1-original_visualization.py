import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from utils.common import load_iris_dataset, visualize_3d_scatter

# Load iris dataset
data, feature_names, target = load_iris_dataset()

# Visualize original data in 3D
output_path = "/workspaces/10944-seminar/images/1-original_visualization.png"
visualize_3d_scatter(
    data=data,
    target=target,
    title="Original Iris Dataset (3D Visualization)",
    save_path=output_path
)