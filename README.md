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
  uv run scripts/original_visualization.py
  ```

![Original](./images/1-original_visualization.png)

To run the Z-score normalization script:
  ```
  uv run scripts/zscore_normalization.py
  ```

![Z-Score](./images/2-zscore_normalization.png)


To run the Min-Max normalization script:
  ```
  uv run scripts/minmax_normalization.py
  ```

![Mini-max](./images/3-minmax_normalization.png)

To run the PCA whitening script:
  ```
  uv run scripts/pca_whitening.py
  ```

![Whitening](./images/4-pca_whitening.png)

To run the per-sample normalization script:
  ```
  uv run scripts/sample_normalization.py
  ```

![per-sample](./images/5-1-per_feature_normalization.png)

To run the per-feature normalization script:
  ```
  uv run scripts/feature_normalization.py
  ```

![per-feature](./images/5-2-per_sample_normalization.png)