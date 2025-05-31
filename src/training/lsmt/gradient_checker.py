import torch
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import os
from datetime import datetime
from pathlib import Path

from utils.logger import logger

class GradientChecker:
    """
    Utility class to monitor and analyze gradients during training to detect issues
    like vanishing or exploding gradients, or other anomalies.
    """
    def __init__(self, model, log_dir="experiments/logs/gradient_stats"):
        """
        Initialize the gradient checker.
        
        Args:
            model: The PyTorch model to monitor
            log_dir: Directory to save gradient statistics and plots
        """
        self.model = model
        self.log_dir = log_dir
        self.grad_stats = defaultdict(list)
        self.hooks = []
        self.step = 0
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Timestamp for file naming
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Gradient checker initialized. Logs will be saved to {log_dir}")
    
    def register_hooks(self):
        """Register hooks for all parameters to capture gradients"""
        self.hooks = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(
                    lambda grad, name=name: self._grad_hook(grad, name)
                )
                self.hooks.append(hook)
        logger.info(f"Registered gradient hooks for {len(self.hooks)} parameters")
        return self
    
    def _grad_hook(self, grad, name):
        """Hook function to capture gradient statistics"""
        if grad is not None:
            # Calculate statistics
            grad_data = grad.detach().cpu()
            
            with torch.no_grad():
                norm = torch.norm(grad_data).item()
                mean = torch.mean(grad_data).item()
                std = torch.std(grad_data).item()
                max_val = torch.max(grad_data).item()
                min_val = torch.min(grad_data).item()
                has_nan = torch.isnan(grad_data).any().item()
                has_inf = torch.isinf(grad_data).any().item()
            
            # Store statistics
            self.grad_stats[name].append({
                'step': self.step,
                'norm': norm,
                'mean': mean,
                'std': std,
                'max': max_val,
                'min': min_val,
                'has_nan': has_nan,
                'has_inf': has_inf
            })
        
        return grad
    
    def check_gradients(self):
        """
        Check for gradient issues after backward pass
        
        Returns:
            dict: Dictionary with detected issues
        """
        issues = {
            'vanishing': False,
            'exploding': False,
            'dead': False,
            'nan_inf': False,
            'details': {}
        }
        
        for name, stats_list in self.grad_stats.items():
            if not stats_list:
                continue
                
            current_stats = stats_list[-1]
            
            # Check for NaN/Inf values
            if current_stats['has_nan'] or current_stats['has_inf']:
                issues['nan_inf'] = True
                issues['details'][name] = 'NaN/Inf values detected'
                logger.warning(f"NaN/Inf gradients detected in {name}")
            
            # Check for vanishing gradients (norm very close to zero)
            elif current_stats['norm'] < 1e-7:
                issues['vanishing'] = True
                issues['details'][name] = f"Vanishing gradient (norm: {current_stats['norm']:.2e})"
                logger.warning(f"Possible vanishing gradient in {name}: norm = {current_stats['norm']:.2e}")
            
            # Check for exploding gradients (very large norm)
            elif current_stats['norm'] > 1e3:
                issues['exploding'] = True
                issues['details'][name] = f"Exploding gradient (norm: {current_stats['norm']:.2e})"
                logger.warning(f"Possible exploding gradient in {name}: norm = {current_stats['norm']:.2e}")
            
            # Check for dead gradients (zero or near-zero variance)
            elif current_stats['std'] < 1e-7 and current_stats['norm'] > 0:
                issues['dead'] = True
                issues['details'][name] = f"Dead gradient (std: {current_stats['std']:.2e})"
                logger.warning(f"Possible dead gradient in {name}: std = {current_stats['std']:.2e}")
        
        self.step += 1
        return issues
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        logger.info("Removed all gradient hooks")
    
    def plot_gradient_stats(self, save=True):
        """
        Plot gradient statistics over training steps
        
        Args:
            save (bool): Whether to save plots to disk
        """
        if not self.grad_stats:
            logger.warning("No gradient statistics available for plotting")
            return
        
        # Create plots directory
        plots_dir = os.path.join(self.log_dir, f"gradient_plots_{self.timestamp}")
        if save:
            os.makedirs(plots_dir, exist_ok=True)
        
        # Group parameters by layer type for more organized visualization
        layer_groups = defaultdict(list)
        
        for name in self.grad_stats.keys():
            # Extract layer type from parameter name
            if 'lstm' in name:
                group = 'lstm'
            elif 'linear' in name:
                group = 'linear'
            elif 'embedding' in name:
                group = 'embedding'
            else:
                group = 'other'
            
            layer_groups[group].append(name)
        
        # Plot statistics for each group
        for group, param_names in layer_groups.items():
            self._plot_group_stats(group, param_names, plots_dir, save)
        
        logger.info(f"Gradient statistics plots {'saved to ' + plots_dir if save else 'generated'}")
    
    def _plot_group_stats(self, group_name, param_names, plots_dir, save):
        """Plot statistics for a group of parameters"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Gradient Statistics for {group_name.upper()} layers', fontsize=16)
        
        # Flatten axes for easier iteration
        axes = axes.flatten()
        
        # Statistics to plot
        stats_to_plot = ['norm', 'mean', 'std', 'max']
        
        for i, stat in enumerate(stats_to_plot):
            ax = axes[i]
            ax.set_title(f'{stat.capitalize()} of Gradients')
            ax.set_xlabel('Training Step')
            ax.set_ylabel(stat.capitalize())
            
            for name in param_names:
                if not self.grad_stats[name]:
                    continue
                    
                steps = [s['step'] for s in self.grad_stats[name]]
                values = [s[stat] for s in self.grad_stats[name]]
                
                # Plot on log scale for norm to better visualize vanishing/exploding
                if stat == 'norm':
                    ax.set_yscale('log')
                
                ax.plot(steps, values, label=name.split('.')[-1], alpha=0.7)
            
            # Add thresholds for norm plot
            if stat == 'norm':
                ax.axhline(y=1e-7, color='r', linestyle='--', alpha=0.5, label='Vanishing threshold')
                ax.axhline(y=1e3, color='orange', linestyle='--', alpha=0.5, label='Exploding threshold')
            
            # Add legend with smaller font
            if len(param_names) < 10:  # Only show legend if not too many parameters
                ax.legend(fontsize='small')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(plots_dir, f'{group_name}_gradients.png'), dpi=150)
            plt.close()
    
    def save_stats_to_file(self):
        """Save gradient statistics to a file"""
        if not self.grad_stats:
            logger.warning("No gradient statistics available to save")
            return
        
        stats_file = os.path.join(self.log_dir, f"gradient_stats_{self.timestamp}.txt")
        
        with open(stats_file, 'w') as f:
            f.write("Gradient Statistics Summary\n")
            f.write("==========================\n\n")
            
            for name, stats_list in self.grad_stats.items():
                f.write(f"Parameter: {name}\n")
                f.write("-" * (len(name) + 11) + "\n")
                
                # Calculate average statistics
                if stats_list:
                    avg_norm = np.mean([s['norm'] for s in stats_list])
                    avg_mean = np.mean([s['mean'] for s in stats_list])
                    avg_std = np.mean([s['std'] for s in stats_list])
                    max_norm = max([s['norm'] for s in stats_list])
                    min_norm = min([s['norm'] for s in stats_list])
                    nan_count = sum([1 for s in stats_list if s['has_nan']])
                    inf_count = sum([1 for s in stats_list if s['has_inf']])
                    
                    f.write(f"Average Norm: {avg_norm:.6e}\n")
                    f.write(f"Average Mean: {avg_mean:.6e}\n")
                    f.write(f"Average Std: {avg_std:.6e}\n")
                    f.write(f"Max Norm: {max_norm:.6e}\n")
                    f.write(f"Min Norm: {min_norm:.6e}\n")
                    f.write(f"NaN Count: {nan_count}\n")
                    f.write(f"Inf Count: {inf_count}\n")
                    
                    # Assessment
                    if nan_count > 0 or inf_count > 0:
                        f.write("Assessment: NaN/Inf values detected - CRITICAL ISSUE\n")
                    elif max_norm > 1e3:
                        f.write("Assessment: Possible exploding gradients detected\n")
                    elif min_norm < 1e-7:
                        f.write("Assessment: Possible vanishing gradients detected\n")
                    else:
                        f.write("Assessment: Gradients appear normal\n")
                
                f.write("\n")
        
        logger.info(f"Gradient statistics saved to {stats_file}")
        return stats_file


def check_gradients_during_training(model, dataloader, criterion, optimizer, device, 
                                   n_batches=None, clip_value=None):
    """
    Function to check gradients during training on a subset of data
    
    Args:
        model: PyTorch model to check
        dataloader: PyTorch DataLoader containing training data
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: Device to run on (cuda/cpu)
        n_batches: Number of batches to check (None = all batches)
        clip_value: Value to clip gradients at (None = no clipping)
        
    Returns:
        dict: Summary of gradient issues detected
    """
    # Initialize gradient checker
    checker = GradientChecker(model)
    checker.register_hooks()
    
    # Track overall issues
    all_issues = {
        'vanishing': False,
        'exploding': False,
        'dead': False,
        'nan_inf': False,
        'problematic_layers': set()
    }
    
    # Set model to training mode
    model.train()
    
    logger.info(f"Starting gradient check on {n_batches if n_batches else 'all'} batches")
    
    # Process batches
    for batch_idx, (data, targets) in enumerate(dataloader):
        if n_batches is not None and batch_idx >= n_batches:
            break
            
        # Move data to device
        data, targets = data.to(device), targets.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        
        # Calculate loss
        loss = criterion(outputs, targets.long())
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        issues = checker.check_gradients()
        
        # Update overall issues
        all_issues['vanishing'] |= issues['vanishing']
        all_issues['exploding'] |= issues['exploding']
        all_issues['dead'] |= issues['dead']
        all_issues['nan_inf'] |= issues['nan_inf']
        
        # Add problematic layers
        all_issues['problematic_layers'].update(issues['details'].keys())
        
        # Apply gradient clipping if specified
        if clip_value is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            
            # Re-check after clipping
            post_clip_issues = checker.check_gradients()
            if post_clip_issues['exploding']:
                logger.warning(f"Exploding gradients persist after clipping with value {clip_value}")
        
        # Optimizer step
        optimizer.step()
        
        # Log progress
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1} batches")
    
    # Generate plots and save statistics
    checker.plot_gradient_stats(save=True)
    stats_file = checker.save_stats_to_file()
    
    # Remove hooks to prevent memory leaks
    checker.remove_hooks()
    
    # Convert problematic_layers from set to list for better serialization
    all_issues['problematic_layers'] = list(all_issues['problematic_layers'])
    
    # Log summary
    logger.info(f"Gradient check completed. Statistics saved to {stats_file}")
    logger.info(f"Issues detected: vanishing={all_issues['vanishing']}, "
                f"exploding={all_issues['exploding']}, "
                f"dead={all_issues['dead']}, "
                f"nan_inf={all_issues['nan_inf']}")
    
    if all_issues['problematic_layers']:
        logger.info(f"Problematic layers: {', '.join(all_issues['problematic_layers'])}")
    
    return all_issues


def gradient_diagnosis(model, dataloader, criterion, optimizer, device, 
                      plot_dir="experiments/logs/gradient_diagnosis",
                      clip_values=[None, 1.0, 5.0]):
    """
    Comprehensive gradient diagnosis function that:
    1. Detects gradient issues
    2. Tests different gradient clipping values
    3. Makes recommendations for model improvement
    
    Args:
        model: PyTorch model to diagnose
        dataloader: PyTorch DataLoader containing training data
        criterion: Loss function
        optimizer: PyTorch optimizer
        device: Device to run on (cuda/cpu)
        plot_dir: Directory to save diagnostic plots
        clip_values: List of gradient clipping values to test (None = no clipping)
        
    Returns:
        dict: Diagnosis results with recommendations
    """
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info(f"Starting comprehensive gradient diagnosis")
    
    # Create deep copy of model for testing different configurations
    original_state_dict = model.state_dict()
    
    # Results dictionary
    results = {
        'timestamp': timestamp,
        'issues_detected': False,
        'clip_test_results': {},
        'recommendations': []
    }
    
    # Run initial gradient check with no clipping
    logger.info("Running initial gradient check without clipping")
    issues = check_gradients_during_training(
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        n_batches=5,  # Check first 5 batches for initial diagnosis
        clip_value=None
    )
    
    # Record if any issues were detected
    results['issues_detected'] = any([
        issues['vanishing'], 
        issues['exploding'], 
        issues['dead'], 
        issues['nan_inf']
    ])
    
    # Add initial recommendations based on issues
    if issues['vanishing']:
        results['recommendations'].append({
            'issue': 'vanishing_gradients',
            'suggestions': [
                'Try using ReLU or LeakyReLU activation functions instead of sigmoid/tanh',
                'Initialize weights properly (Xavier/Kaiming initialization)',
                'Use residual connections or skip connections',
                'Consider using batch normalization',
                'Reduce the depth of the network if possible'
            ]
        })
    
    if issues['exploding']:
        results['recommendations'].append({
            'issue': 'exploding_gradients',
            'suggestions': [
                'Implement gradient clipping (see clip test results below)',
                'Reduce learning rate',
                'Use weight regularization (L2)',
                'Check for proper weight initialization',
                'Consider using layer normalization'
            ]
        })
    
    if issues['dead']:
        results['recommendations'].append({
            'issue': 'dead_gradients',
            'suggestions': [
                'Check for ReLU units that might be consistently inactive',
                'Use LeakyReLU instead of ReLU',
                'Ensure proper weight initialization',
                'Consider adding batch normalization'
            ]
        })
    
    if issues['nan_inf']:
        results['recommendations'].append({
            'issue': 'nan_inf_gradients',
            'suggestions': [
                'Check for division by zero or log of zero in loss function',
                'Verify input data normalization',
                'Implement gradient clipping',
                'Reduce learning rate significantly',
                'Check for numerical stability issues in custom operations'
            ]
        })
    
    # Test different clipping values if exploding gradients were detected
    if issues['exploding'] or issues['nan_inf']:
        logger.info("Testing different gradient clipping values")
        
        for clip_value in clip_values:
            if clip_value is None:
                continue  # Already tested above
                
            logger.info(f"Testing gradient clipping with value {clip_value}")
            
            # Restore original model state
            model.load_state_dict(original_state_dict)
            
            # Test with this clipping value
            clip_issues = check_gradients_during_training(
                model=model,
                dataloader=dataloader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                n_batches=5,
                clip_value=clip_value
            )
            
            # Record results
            results['clip_test_results'][str(clip_value)] = {
                'vanishing': clip_issues['vanishing'],
                'exploding': clip_issues['exploding'],
                'dead': clip_issues['dead'],
                'nan_inf': clip_issues['nan_inf'],
                'improved': not (clip_issues['exploding'] or clip_issues['nan_inf'])
            }
    
    # Determine best clipping value
    if 'clip_test_results' in results and results['clip_test_results']:
        best_clip = None
        for clip_value, result in results['clip_test_results'].items():
            if result['improved'] and (best_clip is None or float(clip_value) < float(best_clip)):
                best_clip = clip_value
        
        if best_clip is not None:
            results['recommendations'].append({
                'issue': 'optimal_clipping',
                'suggestions': [
                    f'Recommended gradient clipping value: {best_clip}'
                ]
            })
    
    # Generate final diagnosis report
    report_path = os.path.join(plot_dir, f"gradient_diagnosis_report_{timestamp}.txt")
    
    with open(report_path, 'w') as f:
        f.write("LSTM Gradient Diagnosis Report\n")
        f.write("=============================\n\n")
        
        f.write("Issues Detected:\n")
        f.write(f"- Vanishing Gradients: {issues['vanishing']}\n")
        f.write(f"- Exploding Gradients: {issues['exploding']}\n")
        f.write(f"- Dead Gradients: {issues['dead']}\n")
        f.write(f"- NaN/Inf Gradients: {issues['nan_inf']}\n\n")
        
        if issues['problematic_layers']:
            f.write("Problematic Layers:\n")
            for layer in issues['problematic_layers']:
                f.write(f"- {layer}\n")
            f.write("\n")
        
        if 'clip_test_results' in results and results['clip_test_results']:
            f.write("Gradient Clipping Test Results:\n")
            for clip_value, result in results['clip_test_results'].items():
                f.write(f"- Clip value {clip_value}: {'IMPROVED' if result['improved'] else 'NO IMPROVEMENT'}\n")
            f.write("\n")
        
        f.write("Recommendations:\n")
        for rec in results['recommendations']:
            f.write(f"For {rec['issue'].replace('_', ' ')}:\n")
            for suggestion in rec['suggestions']:
                f.write(f"- {suggestion}\n")
            f.write("\n")
    
    logger.info(f"Gradient diagnosis completed. Report saved to {report_path}")
    
    # Restore original model state
    model.load_state_dict(original_state_dict)
    
    return results 