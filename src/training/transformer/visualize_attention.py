"""
Utility functions to visualize attention weights from the transformer model.

This module provides functions to visualize how the transformer model
attends to different parts of the input sequence.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import Optional, List, Tuple, Union

def plot_attention_weights(
    attention_weights: torch.Tensor, 
    sample_idx: int = 0, 
    window_size: int = None,
    timestamps: Optional[List[Union[str, int]]] = None,
    title: str = "Attention Weights",
    fig_size: Tuple[int, int] = (12, 6)
) -> Figure:
    """
    Plot attention weights for a single sample.
    
    Args:
        attention_weights: Attention weights tensor [batch_size, seq_len]
        sample_idx: Index of the sample in the batch to visualize
        window_size: Window size for visualization, if None uses the full sequence
        timestamps: Optional list of timestamps for x-axis labels
        title: Title for the plot
        fig_size: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Get attention weights for the selected sample
    weights = attention_weights[sample_idx].cpu().detach().numpy()
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Set window size if not provided
    if window_size is None:
        window_size = len(weights)
    else:
        window_size = min(window_size, len(weights))
    
    # Plot attention weights
    x = np.arange(window_size)
    ax.bar(x, weights[:window_size], width=0.8, alpha=0.7, color='steelblue')
    
    # Set x-axis labels if timestamps are provided
    if timestamps is not None:
        if len(timestamps) > 20:
            # If too many timestamps, show only a subset
            step = len(timestamps) // 10
            tick_positions = np.arange(0, window_size, step)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([timestamps[i] for i in tick_positions], rotation=45)
        else:
            ax.set_xticks(x)
            ax.set_xticklabels(timestamps[:window_size], rotation=45)
    
    # Set labels and title
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Attention Weight')
    ax.set_title(title)
    
    # Add grid
    ax.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line at the average attention
    avg_weight = 1.0 / len(weights)
    ax.axhline(y=avg_weight, color='r', linestyle='--', alpha=0.5, 
               label=f'Average Weight: {avg_weight:.4f}')
    
    # Highlight top-5 highest attention weights
    top_k = 5
    top_indices = np.argsort(weights)[-top_k:]
    for idx in top_indices:
        if idx < window_size:  # Only highlight if within visualization window
            ax.bar([idx], [weights[idx]], width=0.8, color='orange', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_attention_heatmap(
    attention_weights: torch.Tensor,
    window_size: int = None,
    num_samples: int = 10,
    title: str = "Attention Weights Heatmap",
    fig_size: Tuple[int, int] = (12, 8)
) -> Figure:
    """
    Plot a heatmap of attention weights for multiple samples.
    
    Args:
        attention_weights: Attention weights tensor [batch_size, seq_len]
        window_size: Window size for visualization, if None uses the full sequence
        num_samples: Number of samples to visualize
        title: Title for the plot
        fig_size: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Ensure we don't try to plot more samples than we have
    batch_size = attention_weights.shape[0]
    num_samples = min(num_samples, batch_size)
    
    # Get attention weights for the selected samples
    weights = attention_weights[:num_samples].cpu().detach().numpy()
    
    # Set window size if not provided
    if window_size is None:
        window_size = weights.shape[1]
    else:
        window_size = min(window_size, weights.shape[1])
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot heatmap
    im = ax.imshow(weights[:, :window_size], cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight')
    
    # Set labels and title
    ax.set_xlabel('Sequence Position')
    ax.set_ylabel('Sample Index')
    ax.set_title(title)
    
    # Set y-ticks
    ax.set_yticks(np.arange(num_samples))
    ax.set_yticklabels([f'Sample {i}' for i in range(num_samples)])
    
    # Set x-ticks
    if window_size > 20:
        # If too many positions, show only a subset
        step = window_size // 10
        tick_positions = np.arange(0, window_size, step)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_positions)
    else:
        ax.set_xticks(np.arange(window_size))
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def visualize_sample_with_attention(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    window_size: int = None,
    title: str = "Sample with Attention Weights",
    fig_size: Tuple[int, int] = (14, 10)
) -> Figure:
    """
    Visualize a sample with its attention weights.
    
    Args:
        model: Transformer model
        sample: Input tensor [1, seq_len, input_dim]
        device: Device to run the model on
        window_size: Window size for visualization, if None uses the full sequence
        title: Title for the plot
        fig_size: Figure size as (width, height)
        
    Returns:
        Matplotlib Figure object
    """
    # Ensure sample is on the correct device
    sample = sample.to(device)
    
    # Get model predictions and attention weights
    model.eval()
    with torch.no_grad():
        # Get attention weights
        attention_weights = model.get_attention_weights(sample)
    
    # Convert to numpy
    sample_np = sample[0].cpu().numpy()  # [seq_len, input_dim]
    attention_np = attention_weights[0].cpu().numpy()  # [seq_len]
    
    # Set window size if not provided
    seq_len = sample_np.shape[0]
    if window_size is None:
        window_size = seq_len
    else:
        window_size = min(window_size, seq_len)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot input features in the top subplot
    input_dim = sample_np.shape[1]
    for i in range(min(input_dim, 5)):  # Plot up to 5 features
        ax1.plot(sample_np[:window_size, i], label=f'Feature {i}')
    
    if input_dim > 5:
        ax1.set_title(f"{title} (showing first 5 of {input_dim} features)")
    else:
        ax1.set_title(title)
    
    ax1.set_xlabel('Sequence Position')
    ax1.set_ylabel('Feature Value')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Plot attention weights in the bottom subplot
    x = np.arange(window_size)
    ax2.bar(x, attention_np[:window_size], width=0.8, alpha=0.7, color='steelblue')
    
    # Set labels
    ax2.set_xlabel('Sequence Position')
    ax2.set_ylabel('Attention Weight')
    ax2.set_title('Attention Weights')
    
    # Add grid
    ax2.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line at the average attention
    avg_weight = 1.0 / len(attention_np)
    ax2.axhline(y=avg_weight, color='r', linestyle='--', alpha=0.5, 
                label=f'Average Weight: {avg_weight:.4f}')
    
    # Highlight top-5 highest attention weights
    top_k = 5
    top_indices = np.argsort(attention_np)[-top_k:]
    for idx in top_indices:
        if idx < window_size:  # Only highlight if within visualization window
            ax2.bar([idx], [attention_np[idx]], width=0.8, color='orange', alpha=0.7)
    
    # Add legend
    ax2.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def save_attention_visualizations(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: torch.device,
    output_dir: str,
    num_samples: int = 10,
    window_size: int = None
):
    """
    Save attention visualizations for a batch of samples.
    
    Args:
        model: Transformer model
        data_loader: DataLoader containing samples
        device: Device to run the model on
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        window_size: Window size for visualization, if None uses the full sequence
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a batch of samples
    batch_samples, batch_labels = next(iter(data_loader))
    
    # Ensure we don't try to visualize more samples than we have
    batch_size = batch_samples.shape[0]
    num_samples = min(num_samples, batch_size)
    
    # Set model to evaluation mode
    model.eval()
    
    # Process all samples in the batch
    with torch.no_grad():
        # Get attention weights for the entire batch
        attention_weights = model.get_attention_weights(batch_samples.to(device))
    
    # Plot heatmap for all samples
    fig_heatmap = plot_attention_heatmap(
        attention_weights=attention_weights,
        window_size=window_size,
        num_samples=num_samples,
        title="Attention Weights Heatmap"
    )
    
    # Save heatmap
    heatmap_path = os.path.join(output_dir, "attention_heatmap.png")
    fig_heatmap.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(fig_heatmap)
    
    # Plot individual samples
    for i in range(num_samples):
        # Get label for the sample
        label = batch_labels[i].item()
        label_str = "Anomaly" if label == 1 else "Normal"
        
        # Plot sample with attention
        fig_sample = visualize_sample_with_attention(
            model=model,
            sample=batch_samples[i:i+1],
            device=device,
            window_size=window_size,
            title=f"Sample {i} (Label: {label_str})"
        )
        
        # Save sample visualization
        sample_path = os.path.join(output_dir, f"sample_{i}_label_{label_str}.png")
        fig_sample.savefig(sample_path, dpi=300, bbox_inches='tight')
        plt.close(fig_sample)
    
    print(f"Saved {num_samples} sample visualizations to {output_dir}") 