import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from src.utils.logger import logger

def plot_anomaly_scores_distribution(scores, true_labels, save_path):
    """Plot distribution of anomaly scores."""
    plt.figure(figsize=(10, 6))
    
    # Plot distributions for normal and anomaly samples
    sns.kdeplot(scores[true_labels == 0], label='Normal', color='blue')
    sns.kdeplot(scores[true_labels == 1], label='Anomaly', color='red')
    
    plt.title('Distribution of Anomaly Scores')
    plt.xlabel('Anomaly Score')
    plt.ylabel('Density')
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Anomaly scores distribution plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=['Normal', 'Anomaly'],
               yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")

def save_results(results, save_path):
    """Save validation results to markdown file."""
    # Sort results by F1 score
    results.sort(key=lambda x: x['metrics']['f1_score'], reverse=True)
    
    # Create markdown content
    md_content = "# Isolation Forest Model Validation Results\n\n"
    
    # Add timestamp
    from datetime import datetime
    md_content += f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Add best model's detailed metrics
    best_result = results[0]
    md_content += "## Best Model Metrics\n\n"
    md_content += "### Performance Metrics\n"
    md_content += "| Metric | Value |\n"
    md_content += "|--------|-------|\n"
    metrics = best_result['metrics']
    md_content += f"| Accuracy | {metrics['accuracy']:.4f} |\n"
    md_content += f"| Precision | {metrics['precision']:.4f} |\n"
    md_content += f"| Recall | {metrics['recall']:.4f} |\n"
    md_content += f"| F1 Score | {metrics['f1_score']:.4f} |\n\n"
    
    # Add confusion matrix
    md_content += "### Confusion Matrix\n"
    md_content += "```\n"
    cm = metrics['confusion_matrix']
    md_content += f"True Negative: {cm[0,0]}, False Positive: {cm[0,1]}\n"
    md_content += f"False Negative: {cm[1,0]}, True Positive: {cm[1,1]}\n"
    md_content += "```\n\n"
    
    # Add all results summary
    md_content += "## All Models Performance Summary\n\n"
    md_content += "| Rank | F1 Score | Accuracy | Precision | Recall | Parameters |\n"
    md_content += "|------|----------|----------|-----------|---------|------------|\n"
    
    for i, result in enumerate(results, 1):
        metrics = result['metrics']
        params_str = ", ".join(f"{k}={v}" for k, v in result['parameters'].items())
        md_content += (f"| {i} | {metrics['f1_score']:.4f} | {metrics['accuracy']:.4f} | "
                      f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {params_str} |\n")
    
    # Add visualizations
    md_content += "\n## Visualizations\n\n"
    md_content += "### Anomaly Scores Distribution\n"
    md_content += "![Anomaly Scores Distribution](./anomaly_scores_distribution.png)\n\n"
    md_content += "### Confusion Matrix\n"
    md_content += "![Confusion Matrix](./confusion_matrix.png)\n\n"
    
    # Save markdown file
    with open(save_path, 'w') as f:
        f.write(md_content)
    logger.info(f"Validation results saved to {save_path}")