"""
Training script for Isolation Forest-based time series anomaly detection.

This script trains an Isolation Forest model for energy anomaly detection
with comprehensive evaluation.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import confusion_matrix, classification_report

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.training.isolation_forest.isolation_forest_model import IsolationForestAnomalyDetector
from src.training.isolation_forest.dataloader import create_isolation_forest_data


def point_adjustment(gt, pred):
    """
    Point adjustment strategy for anomaly detection evaluation.
    
    Args:
        gt: Ground truth labels
        pred: Predicted labels
    
    Returns:
        Tuple of (adjusted_gt, adjusted_pred)
    """
    gt = gt.copy()
    pred = pred.copy()
    anomaly_state = False
    
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Adjust backward
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            # Adjust forward
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    
    return gt, pred


def evaluate_with_adjustment(preds, labels, model, X, threshold):
    """
    Evaluate predictions with point adjustment.
    
    Args:
        preds: Predictions
        labels: Ground truth labels
        model: Isolation Forest model
        X: Features
        threshold: Decision threshold
        
    Returns:
        Dictionary of adjusted metrics
    """
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    
    labels_adj, preds_adj = point_adjustment(labels, preds)
    
    # Calculate adjusted metrics
    adj_accuracy = accuracy_score(labels_adj, preds_adj)
    adj_precision, adj_recall, adj_f1, _ = precision_recall_fscore_support(
        labels_adj, preds_adj, average='binary', zero_division=0
    )
    
    logger.info(f"\n===== Point Adjustment Results =====")
    logger.info(f"Original predictions: {np.sum(preds)} anomalies")
    logger.info(f"Adjusted predictions: {np.sum(preds_adj)} anomalies")
    logger.info(f"Adjusted Accuracy: {adj_accuracy:.4f}")
    logger.info(f"Adjusted Precision: {adj_precision:.4f}")
    logger.info(f"Adjusted Recall: {adj_recall:.4f}")
    logger.info(f"Adjusted F1: {adj_f1:.4f}")
    
    return {
        'adj_accuracy': adj_accuracy,
        'adj_precision': adj_precision,
        'adj_recall': adj_recall,
        'adj_f1': adj_f1,
        'original_anomaly_count': int(np.sum(preds)),
        'adjusted_anomaly_count': int(np.sum(preds_adj))
    }


def plot_confusion_matrix(cm, save_path):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        save_path: Path to save plot
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Anomaly'],
                yticklabels=['Normal', 'Anomaly'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to: {save_path}")


def plot_feature_importance(importance_df, save_path, top_n=30):
    """
    Plot feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        save_path: Path to save plot
        top_n: Number of top features to plot
    """
    plt.figure(figsize=(10, max(8, top_n * 0.3)))
    
    top_features = importance_df.head(top_n)
    
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved feature importance plot to: {save_path}")


def plot_anomaly_scores(scores, labels, save_path, threshold=None):
    """
    Plot anomaly scores distribution.
    
    Args:
        scores: Anomaly scores
        labels: True labels
        save_path: Path to save plot
        threshold: Decision threshold
    """
    plt.figure(figsize=(10, 6))
    
    # Plot score distribution by class
    plt.hist(scores[labels == 0], bins=50, alpha=0.5, label='Normal', color='blue')
    plt.hist(scores[labels == 1], bins=50, alpha=0.5, label='Anomaly', color='red')
    
    # Add threshold line if provided
    if threshold is not None:
        plt.axvline(x=threshold, color='black', linestyle='--', 
                   label=f'Threshold = {threshold:.4f}')
    
    plt.xlabel('Anomaly Score')
    plt.ylabel('Count')
    plt.title('Distribution of Anomaly Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved anomaly score plot to: {save_path}")


def save_results(results, save_dir):
    """
    Save evaluation results to files.
    
    Args:
        results: Dictionary of results
        save_dir: Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON
    import json
    metrics_path = os.path.join(save_dir, 'metrics.json')
    
    def convert_numpy_types(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy_types(i) for i in obj]
        else:
            return obj
    
    # 递归转换numpy类型为Python原生类型
    metrics_serializable = convert_numpy_types(results)
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    
    logger.info(f"Saved metrics to: {metrics_path}")
    
    # Save summary as text
    summary_path = os.path.join(save_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Isolation Forest Time Series Anomaly Detection Results\n")
        f.write("=" * 60 + "\n\n")
        
        for key, value in results.items():
            if isinstance(value, float):
                f.write(f"{key}: {value:.4f}\n")
            elif isinstance(value, (int, str)):
                f.write(f"{key}: {value}\n")
            elif isinstance(value, np.ndarray) and value.ndim == 2:
                f.write(f"\n{key}:\n")
                f.write(str(value) + "\n")
    
    logger.info(f"Saved summary to: {summary_path}")


def load_config(config_path):
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config


def main(args):
    """
    Main training function.
    
    Args:
        args: Command line arguments
    """
    # Load configuration
    config = load_config(args.config)
    
    # Apply variant if specified
    if args.variant and args.variant in config.get('variants', {}):
        logger.info(f"Applying variant: {args.variant}")
        variant_config = config['variants'][args.variant]
        
        # Update config with variant settings
        for key in ['model', 'data', 'training']:
            if key in variant_config:
                config[key].update(variant_config[key])
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"isolation_forest_{timestamp}"
    
    output_dir = config['paths']['output_dir']
    experiment_dir = os.path.join(output_dir, 'experiments', experiment_name)
    model_dir = os.path.join(experiment_dir, 'model')
    results_dir = os.path.join(experiment_dir, 'results')
    plots_dir = os.path.join(experiment_dir, 'plots')
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Load data
    data_path = config['data']['data_path']
    logger.info(f"Loading data from: {data_path}")
    
    data_loader, data_dict = create_isolation_forest_data(data_path, config)
    
    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    data_loader.save_scaler(scaler_path)
    
    # Create model
    logger.info("Creating Isolation Forest model...")
    model = IsolationForestAnomalyDetector(config)
    
    # Train model
    model.train(
        X_train=data_dict['X_train'],
        y_train=data_dict['y_train'],
        X_val=data_dict['X_val'],
        y_val=data_dict['y_val'],
        feature_names=data_dict['feature_names'],
        contamination=data_dict['contamination']
    )
    
    # Optimize threshold if enabled
    if config['training'].get('optimize_threshold', True):
        threshold_metric = config['training'].get('threshold_metric', 'f1')
        model.optimize_threshold(
            X_val=data_dict['X_val'],
            y_val=data_dict['y_val'],
            metric=threshold_metric
        )
    
    # Evaluate on validation set
    logger.info("\n" + "=" * 60)
    val_metrics = model.evaluate(
        X=data_dict['X_val'],
        y=data_dict['y_val'],
        split_name="Validation"
    )
    
    # Point adjustment evaluation on validation
    if config['evaluation'].get('use_point_adjustment', True):
        val_preds = model.predict(data_dict['X_val'])
        val_adj_metrics = evaluate_with_adjustment(
            val_preds, data_dict['y_val'], model, 
            data_dict['X_val'], model.optimal_threshold
        )
        val_metrics.update(val_adj_metrics)
    
    # Evaluate on test set
    logger.info("\n" + "=" * 60)
    test_metrics = model.evaluate(
        X=data_dict['X_test'],
        y=data_dict['y_test'],
        split_name="Test"
    )
    
    # Point adjustment evaluation on test
    if config['evaluation'].get('use_point_adjustment', True):
        test_preds = model.predict(data_dict['X_test'])
        test_adj_metrics = evaluate_with_adjustment(
            test_preds, data_dict['y_test'], model,
            data_dict['X_test'], model.optimal_threshold
        )
        test_metrics.update(test_adj_metrics)
    
    # Get feature importance
    if config.get('feature_importance', {}).get('save_importance', True):
        importance_df = model.get_feature_importance()
        
        # Save feature importance
        importance_path = os.path.join(results_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance to: {importance_path}")
        
        # Plot feature importance
        top_n = config['feature_importance'].get('plot_top_n', 30)
        importance_plot_path = os.path.join(plots_dir, 'feature_importance.png')
        plot_feature_importance(importance_df, importance_plot_path, top_n=top_n)
    
    # Plot anomaly scores
    if config['evaluation'].get('plot_anomaly_scores', True):
        # Get scores for test data
        test_scores = model.predict_proba(data_dict['X_test'])
        scores_plot_path = os.path.join(plots_dir, 'anomaly_scores.png')
        plot_anomaly_scores(test_scores, data_dict['y_test'], scores_plot_path, model.optimal_threshold)
    
    # Plot confusion matrix
    cm_plot_path = os.path.join(plots_dir, 'confusion_matrix_test.png')
    plot_confusion_matrix(test_metrics['confusion_matrix'], cm_plot_path)
    
    # Save results
    all_results = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'optimal_threshold': model.optimal_threshold,
        'n_features': len(data_dict['feature_names']),
        'contamination': data_dict['contamination']
    }
    
    save_results(all_results, results_dir)
    
    # Save model
    model.save_model(model_dir)
    
    # Save predictions if enabled
    if config['evaluation'].get('save_predictions', True):
        logger.info("Saving predictions...")
        
        test_proba = model.predict_proba(data_dict['X_test'])
        test_preds = model.predict(data_dict['X_test'])
        
        predictions_df = pd.DataFrame({
            'true_label': data_dict['y_test'],
            'predicted_label': test_preds,
            'anomaly_score': test_proba
        })
        
        predictions_path = os.path.join(results_dir, 'test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)
        logger.info(f"Saved predictions to: {predictions_path}")
    
    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f"Experiment: {experiment_name}")
    logger.info(f"Optimal Threshold: {model.optimal_threshold:.4f}")
    logger.info(f"Contamination: {data_dict['contamination']:.4f}")
    logger.info(f"\nTest Performance:")
    logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    logger.info(f"  AUPRC: {test_metrics['auprc']:.4f}")
    
    if 'adj_f1' in test_metrics:
        logger.info(f"\nAdjusted Metrics:")
        logger.info(f"  Adjusted F1: {test_metrics['adj_f1']:.4f}")
        logger.info(f"  Adjusted Precision: {test_metrics['adj_precision']:.4f}")
        logger.info(f"  Adjusted Recall: {test_metrics['adj_recall']:.4f}")
    
    logger.info(f"\nResults saved to: {experiment_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Isolation Forest model for time series anomaly detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/isolation_forest_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the experiment"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        choices=['fast', 'standard', 'deep'],
        help="Model variant to use (overrides config)"
    )
    
    args = parser.parse_args()
    main(args)
