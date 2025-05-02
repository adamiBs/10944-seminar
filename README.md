# 10944 Seminar

## The effect of normilization on dimensionality reduction

## Setting Up the Development Environment
1. Open the project in VS Code.
2. Install the Remote - Containers extension if prompted.
3. Install Docker
4. Reopen the project in the container by selecting "Reopen in Container" from the command palette.

## Running Each Normalization Script
To run the original visualization script:
  ```
  uv run ./scripts/1-1-original_visualization.py
  ```

![Original](./images/1-1-original_visualization.png)

To run the Z-score normalization script:
  ```
  uv run ./scripts/1-2-zscore_normalization.py
  ```

![Z-Score](./images/1-2-zscore_normalization.png)


To run the Min-Max normalization script:
  ```
  uv run ./scripts/1-3-minmax_normalization.py
  ```

![Mini-max](./images/1-3-minmax_normalization.png)

To run the PCA whitening script:
  ```
  uv run ./scripts/1-4-pca_whitening.py
  ```

![Whitening](./images/1-4-pca_whitening.png)

To run the per-sample normalization script:
  ```
  uv run scripts/1-5-sample_normalization.py
  ```

![per-sample](./images/1-5-per_feature_normalization.png)

To run the per-feature normalization script:
  ```
  uv run scripts/1-6-feature_normalization.py
  ```

![per-feature](./images/1-6-per_sample_normalization.png)

------

## Dimensionality Reduction Techniques

To run the PCA dimensionality reduction script:
```
uv run ./scripts/2-1-pca_reduction.py
```

![PCA](./images/2-1-pca_reduction_3d.png)

To run the t-SNE dimensionality reduction script:
```
uv run ./scripts/2-2-tsne_reduction.py
```

![t-SNE](./images/2-2-tsne_reduction.png)

To run the UMAP dimensionality reduction script:
```
uv run ./scripts/2-3-umap_reduction.py
```

![UMAP](./images/2-3-umap_reduction.png)

To run the Autoencoder dimensionality reduction script:
```
uv run ./scripts/2-4-autoencoder_reduction.py
```

![Autoencoder](./images/2-4-autoencoder_reduction.png)

To run the Contrastive Learning dimensionality reduction script:
```
uv run ./scripts/2-5-contrastive_reduction.py
```

![Contrastive](./images/2-5-contrastive_reduction.png)

To run the Parametric t-SNE dimensionality reduction script:
```
uv run ./scripts/2-6-parametric_tsne_reduction.py
```

![Parametric t-SNE](./images/2-6-parametric_tsne_reduction.png)

# Normalization and Dimensionality Reduction Combinations

### Comparison of 3D and 2D Normalized PCA Visualizations

| 3D Visualization | 2D Visualization |
|-----------------|------------------|
| ![Z-Score PCA 3D](./images/3-1-zscore_pca.png) | ![Z-Score PCA 2D](./images/3-1-zscore_pca_2d.png) |
| ![Min-Max PCA 3D](./images/3-2-minmax_pca_3d.png) | ![Min-Max PCA 2D](./images/3-2-minmax_pca_2d.png) |
| ![Feature PCA 3D](./images/3-3-feature_pca_3d.png) | ![Feature PCA 2D](./images/3-3-feature_pca_2d.png) |
| ![Sample PCA 3D](./images/3-4-smaple_pca_3d.png) | ![Sample PCA 2D](./images/3-4-smaple_pca_2d.png) |

### Comparison of 3D and 2D Normalized t-SNE Visualizations

| 3D Visualization | 2D Visualization |
|-----------------|------------------|
| ![Z-Score t-SNE 3D](./images/4-1-zscore_tsne.png) | ![Z-Score t-SNE 2D](./images/4-1-zscore_tsne_2d.png) |
| ![Min-Max t-SNE 3D](./images/4-2-minmax_tsne_3d.png) | ![Min-Max t-SNE 2D](./images/4-2-minmax_tsne_2d.png) |
| ![Feature t-SNE 3D](./images/4-3-feature_tsne_3d.png) | ![Feature t-SNE 2D](./images/4-3-feature_tsne_2d.png) |
| ![Sample t-SNE 3D](./images/4-4-sample_tsne_3d.png) | ![Sample t-SNE 2D](./images/4-4-sample_tsne_2d.png) |

### Comparison of 3D and 2D Normalized UMAP Visualizations

| 3D Visualization | 2D Visualization |
|-----------------|------------------|
| ![Z-Score UMAP 3D](./images/5-1-zscore_umap.png) | ![Z-Score UMAP 2D](./images/5-1-zscore_umap_2d.png) |
| ![Min-Max UMAP 3D](./images/5-2-minmax_umap_3d.png) | ![Min-Max UMAP 2D](./images/5-2-minmax_umap_2d.png) |
| ![Feature UMAP 3D](./images/5-3-feature_umap_3d.png) | ![Feature UMAP 2D](./images/5-3-feature_umap_2d.png) |
| ![Sample UMAP 3D](./images/5-4-sample_umap_3d.png) | ![Sample UMAP 2D](./images/5-4-sample_umap_2d.png) |

### Comparison of 3D and 2D Normalized Autoencoder Visualizations

| 3D Visualization | 2D Visualization |
|-----------------|------------------|
| ![Z-Score Autoencoder 3D](./images/6-1-zscore_autoencoder.png) | ![Z-Score Autoencoder 2D](./images/6-1-zscore_autoencoder_2d.png) |
| ![Min-Max Autoencoder 3D](./images/6-2-minmax_autoencoder_3d.png) | ![Min-Max Autoencoder 2D](./images/6-2-minmax_autoencoder_2d.png) |
| ![Feature Autoencoder 3D](./images/6-3-feature_autoencoder_3d.png) | ![Feature Autoencoder 2D](./images/6-3-feature_autoencoder_2d.png) |
| ![Sample Autoencoder 3D](./images/6-4-sample_autoencoder_3d.png) | ![Sample Autoencoder 2D](./images/6-4-sample_autoencoder_2d.png) |

### Comparison of 3D and 2D Normalized Contrastive Learning Visualizations

| 3D Visualization | 2D Visualization |
|-----------------|------------------|
| ![Z-Score Contrastive Learning 3D](./images/7-1-zscore_contrastive.png) | ![Z-Score Contrastive Learning 2D](./images/7-1-zscore_contrastive_2d.png) |
| ![Min-Max Contrastive Learning 3D](./images/7-2-minmax_contrastive_3d.png) | ![Min-Max Contrastive Learning 2D](./images/7-2-minmax_contrastive_2d.png) |
| ![Feature Contrastive Learning 3D](./images/7-3-feature_contrastive_3d.png) | ![Feature Contrastive Learning 2D](./images/7-3-feature_contrastive_2d.png) |
| ![Sample Contrastive Learning 3D](./images/7-4-sample_contrastive_3d.png) | ![Sample Contrastive Learning 2D](./images/7-4-sample_contrastive_2d.png) |

### Comparison of 3D and 2D Parametric t-SNE Visualizations

| 3D Visualization | 2D Visualization |
|-----------------|------------------|
| ![Z-Score Parametric t-SNE 3D](./images/8-1-zscore_parametric_tsne.png) | ![Z-Score Parametric t-SNE 2D](./images/8-1-zscore_parametric_tsne_2d.png) |
| ![Min-Max Parametric t-SNE 3D](./images/8-2-minmax_parametric_tsne_3d.png) | ![Min-Max Parametric t-SNE 2D](./images/8-2-minmax_parametric_tsne_2d.png) |
| ![Feature Parametric t-SNE 3D](./images/8-3-feature_parametric_tsne_3d.png) | ![Feature Parametric t-SNE 2D](./images/8-3-feature_parametric_tsne_2d.png) |
| ![Sample Parametric t-SNE 3D](./images/8-4-sample_parametric_tsne_3d.png) | ![Sample Parametric t-SNE 2D](./images/8-4-sample_parametric_tsne_2d.png) |
