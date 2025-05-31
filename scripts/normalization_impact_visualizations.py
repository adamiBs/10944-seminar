import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid", palette="muted")

# Read the data
df = pd.read_csv('/workspaces/10944-seminar/data.csv', sep='\t')

# Create the images directory if it doesn't exist
os.makedirs('/workspaces/10944-seminar/images', exist_ok=True)

# Clean up the data
# Rename 'Raw' to 'None' for consistency
df['normalization'] = df['normalization'].replace('Raw', 'None')

# Define custom colors for better differentiation between methods
method_colors = {
    'PCA': '#2ca02c',         # Green
    't-SNE': '#ff7f0e',       # Orange
    'UMAP': '#1f77b4',        # Blue
    'Autoencoder': '#d62728'  # Red
}

# Define metrics to analyze
metrics = ['trustworthiness', 'continuity', 'knn_preservation', 'silhouette_score', 'reconstruction_error']

# === 1. Grouped Bar Chart: Trustworthiness by Method and Normalization ===
plt.figure(figsize=(12, 8))
ax = sns.barplot(
    data=df, 
    x='normalization', 
    y='trustworthiness', 
    hue='method',
    palette=method_colors
)

# Add error bars to show variance (standard error)
ax = sns.barplot(
    data=df, 
    x='normalization', 
    y='trustworthiness',
    hue='method',
    palette=method_colors,
    errorbar='se',
    alpha=0.8
)

# Add individual data points for better understanding
sns.stripplot(
    data=df,
    x='normalization',
    y='trustworthiness',
    hue='method',
    dodge=True,
    alpha=0.7,
    size=8,
    palette=method_colors,
    legend=False
)

# Customize the plot
plt.title('Impact of Normalization on Trustworthiness by Method', fontsize=16)
plt.xlabel('Normalization Method', fontsize=14)
plt.ylabel('Trustworthiness', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add text explaining the metric
plt.figtext(0.5, -0.05,
         "Higher trustworthiness values indicate better performance.\n"
         "Note how Autoencoder trustworthiness fluctuates with normalization, unlike other methods.",
         ha='center', fontsize=12)

# Customize the legend
plt.legend(title="Method", fontsize=12)

plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/normalization_trustworthiness_by_method.png', dpi=300, bbox_inches='tight')
print("Visualization 1 saved: normalization_trustworthiness_by_method.png")

# === 2. Grouped Bar Chart: Reconstruction Error for Autoencoder by Normalization ===
# Filter for Autoencoder data only
autoencoder_df = df[df['method'] == 'Autoencoder'].copy()
autoencoder_df['config'] = autoencoder_df['activation_function'] + ' + ' + autoencoder_df['loss_function']

plt.figure(figsize=(12, 8))

# Create a barplot for reconstruction error
ax = sns.barplot(
    data=autoencoder_df,
    x='normalization',
    y='reconstruction_error',
    hue='config',
    errorbar='se'
)

# Add individual data points
sns.stripplot(
    data=autoencoder_df,
    x='normalization',
    y='reconstruction_error',
    hue='config',
    dodge=True,
    alpha=0.7,
    size=8,
    legend=False
)

# Customize the plot
plt.title('Impact of Normalization on Autoencoder Reconstruction Error', fontsize=16)
plt.xlabel('Normalization Method', fontsize=14)
plt.ylabel('Reconstruction Error', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.yscale('log')  # Use log scale due to large differences

# Add text explaining the metric
plt.figtext(0.5, -0.05,
         "Lower reconstruction error values indicate better performance.\n"
         "Note how Z-score normalization generally results in higher reconstruction error for Autoencoders.",
         ha='center', fontsize=12)

# Customize the legend
plt.legend(title="Activation + Loss", fontsize=12)

plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/normalization_reconstruction_error_autoencoder.png', dpi=300, bbox_inches='tight')
print("Visualization 2 saved: normalization_reconstruction_error_autoencoder.png")

# === 3. Boxplot: Trustworthiness Distribution by Normalization (Faceted by Method) ===
plt.figure(figsize=(16, 10))

# Create a FacetGrid for the boxplot
g = sns.catplot(
    data=df,
    kind='box',
    x='normalization',
    y='trustworthiness',
    col='method',
    palette='Set3',
    height=5,
    aspect=0.8,
    sharey=True,
    margin_titles=True
)

# Add individual data points
for ax, method in zip(g.axes.flat, df['method'].unique()):
    sns.stripplot(
        data=df[df['method'] == method],
        x='normalization',
        y='trustworthiness',
        ax=ax,
        color='black',
        alpha=0.7,
        size=8,
        jitter=True
    )
    ax.set_title(f"{method}")
    ax.set_xlabel('Normalization')

# Customize the plot
g.fig.suptitle('Trustworthiness Distribution by Method and Normalization', fontsize=16, y=1.05)
g.set_axis_labels("Normalization", "Trustworthiness")

# Add text explaining the visualization
plt.figtext(0.5, -0.05,
         "This visualization shows the distribution of trustworthiness scores for each method under different normalization schemes.\n"
         "Note the high variance for Autoencoders compared to the low variance for PCA, t-SNE, and UMAP.",
         ha='center', fontsize=12)

plt.tight_layout()
g.savefig('/workspaces/10944-seminar/images/trustworthiness_distribution_by_method.png', dpi=300, bbox_inches='tight')
print("Visualization 3 saved: trustworthiness_distribution_by_method.png")

# === 4. Heatmap: Silhouette Score per Method and Normalization ===
# Create a pivot table for the heatmap
silhouette_pivot = df.pivot_table(
    index='method',
    columns='normalization',
    values='silhouette_score',
    aggfunc='mean'
)

plt.figure(figsize=(10, 8))

# Create a heatmap
ax = sns.heatmap(
    silhouette_pivot,
    annot=True,
    cmap='YlGnBu',
    fmt='.3f',
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Silhouette Score'}
)

# Customize the plot
plt.title('Silhouette Score by Method and Normalization', fontsize=16)
plt.tight_layout()

# Add text explaining the metric
plt.figtext(0.5, -0.05,
         "Higher silhouette scores indicate better cluster separation.\n"
         "The heatmap reveals interaction patterns between dimensionality reduction methods and normalization techniques.",
         ha='center', fontsize=12)

plt.savefig('/workspaces/10944-seminar/images/silhouette_heatmap.png', dpi=300, bbox_inches='tight')
print("Visualization 4 saved: silhouette_heatmap.png")

# === 5. Scatter Plot: Trustworthiness vs. Continuity ===
plt.figure(figsize=(12, 10))

# Create a scatter plot
scatter = sns.scatterplot(
    data=df,
    x='trustworthiness',
    y='continuity',
    hue='method',
    style='normalization',
    palette=method_colors,
    s=150,
    alpha=0.8
)

# Add annotations to the points
for i, row in df.iterrows():
    plt.text(
        row['trustworthiness'] + 0.005,
        row['continuity'] - 0.005,
        f"{row['method'][:1]}{row['normalization'][:1]}",
        fontsize=9
    )

# Customize the plot
plt.title('Trustworthiness vs. Continuity by Method and Normalization', fontsize=16)
plt.xlabel('Trustworthiness', fontsize=14)
plt.ylabel('Continuity', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)

# Add a diagonal line for reference
lims = [
    np.min([plt.xlim()[0], plt.ylim()[0]]),
    np.max([plt.xlim()[1], plt.ylim()[1]])
]
plt.plot(lims, lims, '--', color='gray', alpha=0.8, zorder=0)

# Add text explaining the visualization
plt.figtext(0.5, -0.05,
         "This scatter plot captures the trade-offs between local structure preservation (trustworthiness) and global structure preservation (continuity).\n"
         "Note how PCA, t-SNE, and UMAP cluster tightly, while Autoencoders spread out depending on normalization.",
         ha='center', fontsize=12)

# Customize the legend
plt.legend(title="Method & Normalization", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('/workspaces/10944-seminar/images/trustworthiness_vs_continuity.png', dpi=300, bbox_inches='tight')
print("Visualization 5 saved: trustworthiness_vs_continuity.png")

# === 6. Matrix Plot: Autoencoder Configuration Grid ===
# Only for Autoencoders
if not autoencoder_df.empty:
    # Create a figure with subplots for each normalization
    norm_types = autoencoder_df['normalization'].unique()
    fig, axes = plt.subplots(1, len(norm_types), figsize=(16, 6), sharey=True)
    
    for i, norm_type in enumerate(norm_types):
        # Filter data for current normalization
        norm_data = autoencoder_df[autoencoder_df['normalization'] == norm_type]
        
        # Create a pivot table
        pivot_data = norm_data.pivot_table(
            index='activation_function',
            columns='loss_function',
            values=['reconstruction_error', 'trustworthiness'],
            aggfunc='mean'
        )
        
        # Get the reconstruction error data
        recon_data = pivot_data['reconstruction_error']
        
        # Generate a heatmap
        ax = axes[i] if len(norm_types) > 1 else axes
        sns.heatmap(
            recon_data,
            annot=True,
            cmap='coolwarm_r',  # Reversed coolwarm (blue is good, red is bad)
            fmt='.3f',
            cbar=i == len(norm_types)-1, 
            ax=ax
        )
        
        # Set title for each subplot
        ax.set_title(f'Normalization: {norm_type}', fontsize=14)
        
        # Adjust labels
        if i == 0:
            ax.set_ylabel('Activation Function', fontsize=12)
        ax.set_xlabel('Loss Function', fontsize=12)
    
    # Add main title
    fig.suptitle('Autoencoder Configuration Impact on Reconstruction Error', fontsize=16, y=1.05)
    
    # Add explanation
    fig.text(0.5, -0.05,
             "This matrix plot shows how different Autoencoder configurations perform with each normalization method.\n"
             "Lower values (blue) indicate better reconstruction performance.",
             ha='center', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('/workspaces/10944-seminar/images/autoencoder_configuration_matrix.png', dpi=300, bbox_inches='tight')
    print("Visualization 6 saved: autoencoder_configuration_matrix.png")

# Display completion message
print("\nAll visualizations have been saved to the /workspaces/10944-seminar/images/ directory.")