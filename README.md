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

![PCA](./images/2-1-pca_reduction.png)

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
