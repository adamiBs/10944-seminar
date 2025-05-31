import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# Set the style for the plots
plt.style.use('seaborn-v0_8-whitegrid')

# Read the data
df = pd.read_csv('/workspaces/10944-seminar/data.csv', sep='\t')

# Define metrics, including reconstruction_error
all_metrics = ['trustworthiness', 'continuity', 'knn_preservation', 'silhouette_score', 'reconstruction_error']

# Create two DataFrames:
# 1. For standard metrics that all methods have
std_metrics = ['trustworthiness', 'continuity', 'knn_preservation', 'silhouette_score']
df_std = df.dropna(subset=std_metrics)

# 2. For reconstruction_error (mainly for Autoencoder)
reconstruction_df = df.dropna(subset=['reconstruction_error'])

# Ensure the images directory exists
os.makedirs('/workspaces/10944-seminar/images', exist_ok=True)

# Define custom colors for better differentiation between methods
method_colors = {
    'PCA': '#2ca02c',         # Green
    't-SNE': '#ff7f0e',       # Orange
    'UMAP': '#1f77b4',        # Blue
    'Autoencoder': '#d62728'  # Red
}

# Plot 1: Standard metrics visualization
# Create a figure with subplots for each standard metric
fig1, axes1 = plt.subplots(1, 4, figsize=(20, 8), sharey=False)

# Plot each standard metric
for i, metric in enumerate(std_metrics):
    # Create a boxplot for the current metric
    ax = axes1[i]
    
    # Create a custom boxplot with improved appearance
    sns.boxplot(
        data=df_std, 
        x='normalization', 
        y=metric,
        hue='method',
        ax=ax,
        palette=method_colors,
        width=0.7,
        linewidth=1.5,
        fliersize=5
    )
    
    # Add individual data points for better understanding
    sns.stripplot(
        data=df_std,
        x='normalization',
        y=metric,
        hue='method',
        ax=ax,
        dodge=True,
        alpha=0.7,
        size=7,
        palette=method_colors,
        legend=False
    )
    
    # Customize the plot
    ax.set_title(metric.replace('_', ' ').title(), fontsize=14, pad=10)
    ax.set_xlabel('Normalization Method', fontsize=12)
    ax.set_ylabel('Performance Metric' if i == 0 else '', fontsize=12)
    
    # Improve grid appearance
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', rotation=0, labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    
    # Only show legend for the last plot to avoid repetition
    if i < 3:
        ax.get_legend().remove()
    else:
        # Customize the legend for better appearance
        legend = ax.get_legend()
        legend.set_title("Method")
        legend._legend_box.align = "left"

# Add a main title to the figure
fig1.suptitle('Impact of Normalization on Dimensionality Reduction Performance', fontsize=18, y=1.05)

# Add a text annotation explaining the visualization
fig1.text(0.5, -0.05, 
         "This visualization shows how different normalization methods (Raw, Z-score, MinMax) affect the performance of various\n"
         "dimensionality reduction techniques across four key metrics. Higher values are generally better for all metrics.",
         ha='center', fontsize=12)

# Adjust the layout
plt.tight_layout()

# Save the first figure
plt.savefig('/workspaces/10944-seminar/images/normalization_impact_on_metrics.png', dpi=300, bbox_inches='tight')
print("Standard metrics visualization saved to '/workspaces/10944-seminar/images/normalization_impact_on_metrics.png'")

# Plot 2: Reconstruction Error visualization (specific to methods that perform reconstruction, like Autoencoders)
if not reconstruction_df.empty:
    plt.figure(figsize=(10, 6))
    
    # Use barplot for reconstruction error to better visualize differences
    ax = sns.barplot(
        data=reconstruction_df,
        x='normalization',
        y='reconstruction_error',
        hue='method',
        palette=method_colors,
        alpha=0.8,
        errwidth=1.5
    )
    
    # Add individual data points
    sns.stripplot(
        data=reconstruction_df,
        x='normalization',
        y='reconstruction_error',
        hue='method',
        dodge=True,
        alpha=0.7,
        size=8,
        palette=method_colors,
        legend=False
    )
    
    # Customize the plot
    plt.title('Impact of Normalization on Reconstruction Error', fontsize=16)
    plt.xlabel('Normalization Method', fontsize=14)
    plt.ylabel('Reconstruction Error', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text explaining the metric
    plt.figtext(0.5, -0.05, 
             "Lower reconstruction error values indicate better performance.\n"
             "This metric is primarily applicable to methods like Autoencoders that perform reconstruction.",
             ha='center', fontsize=12)
    
    # Customize the legend
    legend = plt.legend(title="Method", loc="best")
    legend._legend_box.align = "left"
    
    plt.tight_layout()
    
    # Save the second figure
    plt.savefig('/workspaces/10944-seminar/images/normalization_impact_on_reconstruction.png', dpi=300, bbox_inches='tight')
    print("Reconstruction error visualization saved to '/workspaces/10944-seminar/images/normalization_impact_on_reconstruction.png'")

# Alternative Altair implementation for interactive visualization
try:
    import altair as alt
    
    # 1. Standard metrics visualization
    # Melt the dataframe to long format for easier plotting
    df_std_melted = df_std.melt(
        id_vars=['method', 'normalization'],
        value_vars=std_metrics,
        var_name='metric',
        value_name='value'
    )
    
    # Define domains and ranges for consistent coloring
    domain = ['PCA', 't-SNE', 'UMAP', 'Autoencoder']
    range_ = ['#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']
    
    # Create box plots for each standard metric
    chart1 = alt.Chart(df_std_melted).mark_boxplot(size=30, extent='min-max').encode(
        x=alt.X('normalization:N', 
                title='Normalization Method',
                axis=alt.Axis(labelAngle=0, titleFontSize=12, labelFontSize=11)),
        y=alt.Y('value:Q', 
                title='Performance Metric',
                axis=alt.Axis(titleFontSize=12, labelFontSize=11)),
        color=alt.Color('method:N', 
                      title='Method',
                      scale=alt.Scale(domain=domain, range=range_)),
        column=alt.Column('metric:N', 
                        title='Metric', 
                        header=alt.Header(
                            titleOrient="top", 
                            labelOrient="bottom", 
                            titleFontSize=14, 
                            labelFontSize=14))
    )
    
    # Add individual data points as circles
    points1 = alt.Chart(df_std_melted).mark_circle(size=60, opacity=0.7).encode(
        x=alt.X('normalization:N'),
        y=alt.Y('value:Q'),
        color=alt.Color('method:N', 
                      scale=alt.Scale(domain=domain, range=range_)),
        column=alt.Column('metric:N'),
        tooltip=['method', 'normalization', 'value']
    )
    
    # Combine the box plots and points for standard metrics
    final_chart1 = (chart1 + points1).properties(
        title=alt.TitleParams(
            'Impact of Normalization on Dimensionality Reduction Performance',
            fontSize=16
        )
    ).configure_view(
        stroke=None
    )
    
    # Save the first chart
    final_chart1.save('/workspaces/10944-seminar/images/normalization_impact_on_metrics.html')
    print("Interactive standard metrics visualization saved to '/workspaces/10944-seminar/images/normalization_impact_on_metrics.html'")
    
    # 2. Reconstruction Error visualization
    if not reconstruction_df.empty:
        # Create a chart for reconstruction error
        recon_chart = alt.Chart(reconstruction_df).mark_bar(opacity=0.8).encode(
            x=alt.X('normalization:N', 
                    title='Normalization Method',
                    axis=alt.Axis(labelAngle=0, titleFontSize=12, labelFontSize=11)),
            y=alt.Y('reconstruction_error:Q', 
                    title='Reconstruction Error',
                    axis=alt.Axis(titleFontSize=12, labelFontSize=11)),
            color=alt.Color('method:N', 
                          title='Method',
                          scale=alt.Scale(domain=domain, range=range_)),
            tooltip=['method', 'normalization', 'reconstruction_error', 
                    'activation_function', 'output_activation', 'loss_function']
        )
        
        # Add individual data points as circles
        recon_points = alt.Chart(reconstruction_df).mark_circle(size=80, opacity=0.9).encode(
            x=alt.X('normalization:N', 
                    title='Normalization Method'),
            y=alt.Y('reconstruction_error:Q'),
            color=alt.Color('method:N'),
            tooltip=['method', 'normalization', 'reconstruction_error', 
                    'activation_function', 'output_activation', 'loss_function']
        )
        
        # Combine the charts
        final_chart2 = (recon_chart + recon_points).properties(
            title=alt.TitleParams(
                'Impact of Normalization on Reconstruction Error',
                fontSize=16
            ),
            width=600,
            height=400
        ).configure_view(
            stroke=None
        )
        
        # Save the second chart
        final_chart2.save('/workspaces/10944-seminar/images/normalization_impact_on_reconstruction.html')
        print("Interactive reconstruction error visualization saved to '/workspaces/10944-seminar/images/normalization_impact_on_reconstruction.html'")
    
except ImportError:
    print("Altair not installed. Using matplotlib visualizations only.")