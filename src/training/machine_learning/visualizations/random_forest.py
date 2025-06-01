import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from src.utils.logger import logger

def plot_roc_curve(model, X_val, y_val, save_path):
    """Plot and save ROC curve."""
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"ROC curve saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
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

def plot_precision_recall_curve(y_true, y_pred_proba, optimal_threshold, save_path):
    """Plot and save precision-recall curve with optimal threshold point."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    plt.figure(figsize=(10, 6))
    
    # 绘制PR曲线
    plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
    
    # 找到最佳阈值对应的点
    threshold_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    plt.plot(recall[threshold_idx], precision[threshold_idx], 'ro', 
             label=f'Optimal threshold: {optimal_threshold:.4f}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Optimal Threshold')
    plt.grid(True)
    plt.legend()
    
    # 添加阈值信息
    plt.text(0.02, 0.02, 
             f'Optimal Threshold: {optimal_threshold:.4f}\n'
             f'Precision: {precision[threshold_idx]:.4f}\n'
             f'Recall: {recall[threshold_idx]:.4f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8))
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Precision-Recall curve saved to {save_path}")

def calculate_metrics_from_confusion_matrix(cm):
    """Calculate metrics from confusion matrix."""
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score
    }

def plot_metrics_comparison(results, save_path):
    """Plot comparison of metrics across different models."""
    metrics = []
    for result in results:
        # Get predictions from the model
        y_true = result['y_true']
        y_pred = result['y_pred']
        cm = confusion_matrix(y_true, y_pred)
        model_metrics = calculate_metrics_from_confusion_matrix(cm)
        metrics.append(model_metrics)
    
    df_metrics = pd.DataFrame(metrics)
    plt.figure(figsize=(12, 6))
    df_metrics.plot(kind='bar', ax=plt.gca())
    plt.title('Model Performance Metrics Comparison')
    plt.xlabel('Model Index')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Metrics comparison plot saved to {save_path}")

def save_results(results, validation_report_path):
    """Save validation results to markdown file."""
    # Sort results by F1 Score
    results.sort(key=lambda x: x['metrics']['F1 Score'], reverse=True)
    
    md_content = "# Random Forest Model Validation Results\n\n"
    md_content += "## Performance Summary\n\n"
    md_content += "| Rank | Accuracy | Precision | Recall | F1 Score | Threshold | Parameters |\n"
    md_content += "|------|----------|-----------|---------|-----------|-----------|------------|\n"
    
    for i, result in enumerate(results, 1):
        metrics = result['metrics']
        params_str = ", ".join(f"{k}={v}" for k, v in result['parameters'].items())
        md_content += (f"| {i} | {metrics['Accuracy']:.4f} | {metrics['Precision']:.4f} | "
                      f"{metrics['Recall']:.4f} | {metrics['F1 Score']:.4f} | "
                      f"{metrics['threshold']:.4f} | {params_str} |\n")
    
    md_content += "\n## Detailed Classification Reports\n\n"
    
    for i, result in enumerate(results, 1):
        metrics = result['metrics']
        threshold_metrics = metrics['threshold_metrics']
        
        md_content += f"### Model {i}\n\n"
        md_content += f"Parameters: {result['parameters']}\n\n"
        md_content += "```\n"
        md_content += f"Optimal Threshold: {metrics['threshold']:.4f}\n"
        md_content += f"Threshold Metrics:\n"
        md_content += f"  Precision: {threshold_metrics['precision']:.4f}\n"
        md_content += f"  Recall: {threshold_metrics['recall']:.4f}\n"
        md_content += f"  F1 Score: {threshold_metrics['f1_score']:.4f}\n\n"
        md_content += f"Final Metrics:\n"
        md_content += f"  Accuracy:  {metrics['Accuracy']:.4f}\n"
        md_content += f"  Precision: {metrics['Precision']:.4f}\n"
        md_content += f"  Recall:    {metrics['Recall']:.4f}\n"
        md_content += f"  F1 Score:  {metrics['F1 Score']:.4f}\n"
        md_content += "```\n\n"
    
    os.makedirs(os.path.dirname(validation_report_path), exist_ok=True)
    with open(validation_report_path, 'w') as f:
        f.write(md_content)
    logger.info(f"Validation results saved to {validation_report_path}")
