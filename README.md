# 10944 Seminar

## The effect of normalization on dimensionality reduction

## Setting Up the Development Environment
1. Open the project in VS Code.
2. Install the Remote - Containers extension if prompted.
3. Install Docker
4. Reopen the project in the container by selecting "Reopen in Container" from the command palette.

## Project Visualizations

### 1. Normalization Techniques

| Technique | Visualization |
|-----------|---------------|
| Original Data | ![Original](./images/1-1-original_visualization.png) |
| Z-Score Normalization | ![Z-Score](./images/1-4-zscore_normalization.png) |
| Min-Max Normalization | ![Min-Max](./images/1-5-minmax_normalization.png) |
| Per-Sample Normalization | ![Per-Sample](./images/1-3-per_sample_normalization.png) |
| Per-Feature Normalization | ![Per-Feature](./images/1-5-per_feature_normalization.png) |

### 2. Dimensionality Reduction Techniques

| Technique | 2D Visualization | 3D Visualization |
|-----------|------------------|------------------|
| PCA | ![PCA 2D](./images/2-1-pca_reduction_2d.png) | ![PCA 3D](./images/2-1-pca_reduction_3d.png) |
| t-SNE | ![t-SNE 2D](./images/2-2-tsne_reduction_2d.png) | ![t-SNE 3D](./images/2-2-tsne_reduction_3d.png) |
| UMAP | ![UMAP](./images/2-3-umap_reduction.png) | - |
| Autoencoder | ![Autoencoder](./images/2-4-autoencoder_reduction.png) | - |
| Contrastive Learning | ![Contrastive](./images/2-5-contrastive_reduction.png) | - |
| Parametric t-SNE | ![Parametric t-SNE](./images/2-6-parametric_tsne_reduction.png) | - |

### 3. Wine Dataset: Combined Normalization and Dimensionality Reduction 

#### 3.1 Raw vs Z-Score PCA
| Processing | 2D | 3D |
|------------|-----|-----|
| Raw PCA | ![Raw PCA 2D](./images/3-1-raw_pca_wine_2d.png) | ![Raw PCA 3D](./images/3-1-raw_pca_wine_3d.png) |
| Z-Score PCA | ![Z-Score PCA 2D](./images/3-1-zscore_pca_wine_2d.png) | ![Z-Score PCA 3D](./images/3-1-zscore_pca_wine_3d.png) |

#### 3.2 Raw vs Z-Score with PCA and t-SNE
| Processing | PCA 2D | PCA 3D | t-SNE 2D | t-SNE 3D |
|------------|--------|--------|----------|----------|
| Raw | ![Raw PCA 2D](./images/3-2-raw_pca_wine_2d.png) | ![Raw PCA 3D](./images/3-2-raw_pca_wine_3d.png) | ![Raw t-SNE 2D](./images/3-2-raw_tsne_wine_2d.png) | ![Raw t-SNE 3D](./images/3-2-raw_tsne_wine_3d.png) |
| Z-Score | ![Z-Score PCA 2D](./images/3-2-zscore_pca_wine_2d.png) | ![Z-Score PCA 3D](./images/3-2-zscore_pca_wine_3d.png) | ![Z-Score t-SNE 2D](./images/3-2-zscore_tsne_wine_2d.png) | ![Z-Score t-SNE 3D](./images/3-2-zscore_tsne_wine_3d.png) |

#### 3.3 Raw vs Z-Score with PCA and UMAP
| Processing | PCA 2D | PCA 3D | UMAP 2D | UMAP 3D |
|------------|--------|--------|---------|---------|
| Raw | ![Raw PCA 2D](./images/3-3-raw_pca_wine_2d.png) | ![Raw PCA 3D](./images/3-3-raw_pca_wine_3d.png) | ![Raw UMAP 2D](./images/3-3-raw_umap_wine_2d.png) | ![Raw UMAP 3D](./images/3-3-raw_umap_wine_3d.png) |
| Z-Score | ![Z-Score PCA 2D](./images/3-3-zscore_pca_wine_2d.png) | ![Z-Score PCA 3D](./images/3-3-zscore_pca_wine_3d.png) | ![Z-Score UMAP 2D](./images/3-3-zscore_umap_wine_2d.png) | ![Z-Score UMAP 3D](./images/3-3-zscore_umap_wine_3d.png) |

#### 3.4 Min-Max Normalization
| Technique | Visualization |
|-----------|---------------|
| Min-Max PCA | ![Min-Max PCA 2D](./images/3-4-minmax_pca_wine_2d.png) |
| Min-Max t-SNE | ![Min-Max t-SNE 2D](./images/3-4-minmax_tsne_wine_2d.png) |

### 4. Autoencoder Experiments

#### 4.1 MinMax Normalized Autoencoders with LeakyReLU(0.3) and Sigmoid

| Loss Function | Model Visualization | Loss Plot | Sample Reconstruction |
|---------------|---------------------|-----------|----------------------|
| BCE | ![BCE Model](./images/4-autoencoder_minmax_leakyrelu03_sigmoid_bce.png) | ![BCE Loss](./images/4-autoencoder_minmax_leakyrelu03_sigmoid_bce_loss.png) | ![BCE Samples](./images/4-autoencoder_minmax_leakyrelu03_sigmoid_bce_samples.png) |
| MSE | ![MSE Model](./images/4-autoencoder_minmax_leakyrelu03_sigmoid_mse.png) | ![MSE Loss](./images/4-autoencoder_minmax_leakyrelu03_sigmoid_mse_loss.png) | ![MSE Samples](./images/4-autoencoder_minmax_leakyrelu03_sigmoid_mse_samples.png) |

#### 4.2 MinMax Normalized Autoencoders with ReLU and Sigmoid

| Loss Function | Model Visualization | Loss Plot | Sample Reconstruction |
|---------------|---------------------|-----------|----------------------|
| BCE | ![BCE Model](./images/4-autoencoder_minmax_relu_sigmoid_bce.png) | ![BCE Loss](./images/4-autoencoder_minmax_relu_sigmoid_bce_loss.png) | ![BCE Samples](./images/4-autoencoder_minmax_relu_sigmoid_bce_samples.png) |
| MSE | ![MSE Model](./images/4-autoencoder_minmax_relu_sigmoid_mse.png) | ![MSE Loss](./images/4-autoencoder_minmax_relu_sigmoid_mse_loss.png) | ![MSE Samples](./images/4-autoencoder_minmax_relu_sigmoid_mse_samples.png) |

#### 4.3 Z-Score Normalized Autoencoders with LeakyReLU(0.3) and Linear Activation

| Loss Function | Model Visualization | Loss Plot | Sample Reconstruction |
|---------------|---------------------|-----------|----------------------|
| BCE | ![BCE Model](./images/4-autoencoder_zscore_leakyrelu03_linear_bce.png) | ![BCE Loss](./images/4-autoencoder_zscore_leakyrelu03_linear_bce_loss.png) | ![BCE Samples](./images/4-autoencoder_zscore_leakyrelu03_linear_bce_samples.png) |
| MSE | ![MSE Model](./images/4-autoencoder_zscore_leakyrelu03_linear_mse.png) | ![MSE Loss](./images/4-autoencoder_zscore_leakyrelu03_linear_mse_loss.png) | ![MSE Samples](./images/4-autoencoder_zscore_leakyrelu03_linear_mse_samples.png) |

#### 4.4 Z-Score Normalized Autoencoders with ReLU and Linear Activation

| Loss Function | Model Visualization | Loss Plot | Sample Reconstruction |
|---------------|---------------------|-----------|----------------------|
| BCE | ![BCE Model](./images/4-autoencoder_zscore_relu_linear_bce.png) | ![BCE Loss](./images/4-autoencoder_zscore_relu_linear_bce_loss.png) | ![BCE Samples](./images/4-autoencoder_zscore_relu_linear_bce_samples.png) |
| MSE | ![MSE Model](./images/4-autoencoder_zscore_relu_linear_mse.png) | ![MSE Loss](./images/4-autoencoder_zscore_relu_linear_mse_loss.png) | ![MSE Samples](./images/4-autoencoder_zscore_relu_linear_mse_samples.png) |
