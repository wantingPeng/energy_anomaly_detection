"""
Training script for Random Forest-based time series anomaly detection.

This script trains a Random Forest model for energy anomaly detection
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
from src.training.random_forest.random_forest_model import RandomForestAnomalyDetector
from src.training.random_forest.dataloader import create_random_forest_data


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
        model: Random Forest model
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
    logger.info(f" predictions: {np.sum(preds_adj)} anomalies")
    logger.info(f" Accuracy: {adj_accuracy:.4f}")
    logger.info(f" Precision: {adj_precision:.4f}")
    logger.info(f" Recall: {adj_recall:.4f}")
    logger.info(f" F1: {adj_f1:.4f}")
    
    return {
        'accuracy': adj_accuracy,
        'precision': adj_precision,
        'recall': adj_recall,
        'f1': adj_f1
    }




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
    
    # Save summary as text (directly saving the config)
    summary_path = os.path.join(save_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        yaml.dump(results['config'], f, default_flow_style=False)
    
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
    experiment_name = args.experiment_name or f"random_forest_{timestamp}"
    
    output_dir = config['paths']['output_dir']
    experiment_dir = os.path.join(output_dir, 'experiments', experiment_name)
    model_dir = os.path.join(experiment_dir, 'model')
    results_dir = os.path.join(experiment_dir, 'results')
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Experiment directory: {experiment_dir}")
    
    # Load data
    data_path = config['data']['data_path']
    logger.info(f"Loading data from: {data_path}")
    
    data_loader, data_dict = create_random_forest_data(data_path, config)
    
    # Save scaler (dummy for Random Forest)
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    data_loader.save_scaler(scaler_path)
    
    # Create model
    logger.info("Creating Random Forest model...")
    model = RandomForestAnomalyDetector(config)
    
    # Set class weights if provided and not already set in config
    if data_dict['class_weights'] is not None and model.class_weight is None:
        config['model']['class_weight'] = data_dict['class_weights']
    
    # Train model
    model.train(
        X_train=data_dict['X_train'],
        y_train=data_dict['y_train'],
        X_val=data_dict['X_val'],
        y_val=data_dict['y_val'],
        feature_names=data_dict['feature_names'],
        class_weights=data_dict['class_weights']
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
        importance_type = config['feature_importance'].get('importance_type', 'impurity')
        importance_df = model.get_feature_importance(importance_type=importance_type)
        
        # Save feature importance
        importance_path = os.path.join(results_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance to: {importance_path}")
        
       
    # Save results
    all_results = {
        'experiment_name': experiment_name,
        'timestamp': timestamp,
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'optimal_threshold': model.optimal_threshold,
        'n_features': len(data_dict['feature_names']),
        'config': config
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
            'predicted_probability': test_proba
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
    logger.info(f"\nTest Performance:")

    
    if 'f1' in test_metrics:
        logger.info(f"\n Metrics:")
        logger.info(f"  F1: {test_metrics['f1']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
    
    logger.info(f"\nResults saved to: {experiment_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Random Forest model for time series anomaly detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/random_forest_config.yaml",
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

