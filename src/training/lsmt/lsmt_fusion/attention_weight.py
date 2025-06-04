
import os
import torch
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def visualize_attention_weights(attn_weights, experiment_dir, epoch, prefix='val'):
    """
    Visualize attention weights and save the plots.
    
    Args:
        attn_weights: List of attention weight tensors
        experiment_dir: Main experiment directory
        epoch: Current epoch
        prefix: Prefix for the plot filenames
    """
    # Create attention weights directory
    save_dir = os.path.join("experiments/lstm_late_fusion/attention_weight")
    os.makedirs(save_dir, exist_ok=True)
    
    # Also visualize the average attention pattern
    plt.figure(figsize=(10, 4))
    avg_weights = torch.cat(attn_weights, dim=0).mean(dim=0)
    plt.plot(avg_weights.numpy())
    plt.title(f'Average Attention Weights - Epoch {epoch}')
    plt.xlabel('Time Step')
    plt.ylabel('Weight')
    plt.grid(True)
    
    avg_save_path = os.path.join(save_dir, f'{prefix}_avg_attention_weights_epoch_{epoch}.png')
    plt.savefig(avg_save_path)
    plt.close()
    
    logger.info(f"Saved attention weight visualizations to {save_dir}")

