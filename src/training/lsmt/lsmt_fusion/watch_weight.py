import os
import torch
import matplotlib.pyplot as plt
import logging
import numpy as np

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


def visualize_lstm_gradients(model, epoch, prefix='train'):
    """
    Visualize gradients of the LSTM layers to check for vanishing gradients.
    
    Args:
        model: LSTM Late Fusion model
        epoch: Current epoch number
        prefix: Prefix for the plot filenames ('train' or 'val')
    """
    # Create directory for saving gradient visualizations
    save_dir = os.path.join("experiments/lstm_late_fusion/lsmt_weight")
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract gradients from LSTM layers
    lstm_gradients = {}
    
    # Get gradients for LSTM weight matrices
    for name, param in model.named_parameters():
        if 'lstm' in name and param.requires_grad and param.grad is not None:
            # Store the mean absolute value of gradients for each LSTM parameter
            lstm_gradients[name] = param.grad.abs().mean().item()
    
    # Log the gradient values
    for name, grad_value in lstm_gradients.items():    
        logger.info(f"Epoch {epoch} - Mean gradient for {name}: {grad_value:.6f}")
    
    # Plot the gradients
    plt.figure(figsize=(12, 6))
    
    # Sort parameters by layer (weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, etc.)
    layer_groups = {}
    for name, value in lstm_gradients.items():
        # Extract layer number
        if "_l" in name:
            layer_num = int(name.split("_l")[1].split("_")[0])
            if layer_num not in layer_groups:
                layer_groups[layer_num] = []
            layer_groups[layer_num].append((name, value))
    

    # Calculate average gradient magnitude per layer
    layer_avg_grads = {}
    for layer_num, params in layer_groups.items():
        layer_avg_grads[layer_num] = np.mean([val for _, val in params])
        '''

    # Plot average gradient magnitude per layer
    layers = sorted(layer_avg_grads.keys())
    avg_grads = [layer_avg_grads[layer] for layer in layers]
    plt.subplot(2, 1, 1)
    plt.bar(range(len(layers)), avg_grads, align='center')
    plt.xticks(range(len(layers)), [f"Layer {l}" for l in layers])
    plt.title(f'Average LSTM Gradient Magnitude per Layer - Epoch {epoch}')
    plt.ylabel('Mean Gradient Magnitude')
    plt.grid(True, axis='y')
   
    # Plot all parameter gradients
    plt.subplot(2, 1, 2)
    names = list(lstm_gradients.keys())
    values = list(lstm_gradients.values())
    
    plt.bar(range(len(names)), values, align='center')
    plt.xticks(range(len(names)), names, rotation=90)
    plt.title(f'LSTM Parameter Gradients - Epoch {epoch}')
    plt.ylabel('Mean Gradient Magnitude')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(save_dir, f'{prefix}_lstm_gradients_epoch_{epoch}.png')
    plt.savefig(save_path)
    plt.close()
     '''
    # If we have data from previous epochs, plot the trend
    if hasattr(visualize_lstm_gradients, 'history'):
        # Add current epoch data
        for layer, grad in layer_avg_grads.items():
            if layer not in visualize_lstm_gradients.history:
                visualize_lstm_gradients.history[layer] = []
            visualize_lstm_gradients.history[layer].append(grad)
    else:
        # Initialize history on first call
        visualize_lstm_gradients.history = {}
        for layer, grad in layer_avg_grads.items():
            visualize_lstm_gradients.history[layer] = [grad]
    
    # Plot gradient trend over epochs
    plt.figure(figsize=(10, 6))
    for layer in sorted(visualize_lstm_gradients.history.keys()):
        epochs = range(1, epoch + 1)[-len(visualize_lstm_gradients.history[layer]):]
        plt.plot(epochs, visualize_lstm_gradients.history[layer], marker='o', label=f'Layer {layer}')
    
    plt.title('LSTM Gradient Magnitude Trend Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Gradient Magnitude')
    plt.legend()
    plt.grid(True)
    
    # Save the trend plot
    trend_save_path = os.path.join(save_dir, f'{prefix}_lstm_gradient_trend.png')
    plt.savefig(trend_save_path)
    plt.close()
    
    logger.info(f"Saved LSTM gradient visualizations to {save_dir}")

