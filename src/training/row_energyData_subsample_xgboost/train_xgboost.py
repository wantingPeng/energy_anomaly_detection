#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add the root directory to Python path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.training.row_energyData_subsample_xgboost.xgboost_dataLoader import XGBoostDataLoader
from src.training.row_energyData_subsample_xgboost.xgboost_model import XGBoostAnomalyDetector
from src.utils.logger import logger


def setup_logger(script_name):
    """
    Configure logger for the script.
    
    Args:
        script_name (str): Name of the script
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    # Logger is already set up in utils/logger.py, just need to update the file handler
    for handler in logger.handlers:
        if isinstance(handler, type(logger.handlers[0])):  # FileHandler type
            handler.close()
            logger.handlers.remove(handler)
    
    # Add a new file handler
    import logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Logger configured. Log file: {log_file}")
    return timestamp


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train XGBoost model for anomaly detection")
    
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_xgboost.yaml",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--tune",
        action="store_true",
        default=True,
        help="Perform hyperparameter tuning (enabled by default)"
    )
    
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip hyperparameter tuning"
    )
    
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate on test data after training"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to a saved model to load instead of training"
    )
    
    return parser.parse_args()


def main():
    """
    Main function to train and evaluate XGBoost model.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Setup logger
    timestamp = setup_logger("train_xgboost")
    
    # Start timing
    start_time = time.time()
    logger.info("Starting XGBoost anomaly detection training")
    logger.info(f"Configuration file: {args.config}")
    
    # Determine if we should tune hyperparameters
    tune_hyperparameters = args.tune and not args.no_tune
    if tune_hyperparameters:
        logger.info("Hyperparameter tuning is enabled")
    else:
        logger.info("Hyperparameter tuning is disabled")
    
    try:
        # Load configuration
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        
        # Update save directory with timestamp
        config['data_config']['save_dir'] = os.path.join(
            config['data_config']['save_dir'],
            f"xgboost_{timestamp}"
        )
        os.makedirs(config['data_config']['save_dir'], exist_ok=True)
        
        # Load data
        logger.info("Initializing data loader")
        data_loader = XGBoostDataLoader(args.config)
        
        X_train, y_train = data_loader.load_train_data()
        X_val, y_val = data_loader.load_val_data()
        
        # Report data shapes
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        # Initialize and train model
        if args.model_path:
            logger.info(f"Loading model from {args.model_path}")
            model = XGBoostAnomalyDetector.load_model(args.model_path)
        else:
            logger.info("Initializing new model")
            model = XGBoostAnomalyDetector(config)
            
            logger.info("Starting model training")
            model.train(X_train, y_train, X_val, y_val, tune_hyperparameters=tune_hyperparameters)
        
        # Evaluate on validation data
        logger.info("Evaluating model on validation data")
        val_metrics = model.evaluate(X_val, y_val)
        
        # Print a summary of validation metrics
        logger.info("Validation metrics summary:")
        logger.info(f"  F1 Score: {val_metrics['f1']:.4f}")
        logger.info(f"  Precision: {val_metrics['precision']:.4f}")
        logger.info(f"  Recall: {val_metrics['recall']:.4f}")
        logger.info(f"  Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"  ROC AUC: {val_metrics['roc_auc']:.4f}")
        logger.info(f"  Optimal threshold: {val_metrics['optimal_threshold']:.4f}")
        
        # Evaluate on test data if requested
        if args.test:
            logger.info("Loading test data")
            X_test, y_test = data_loader.load_test_data()
            
            logger.info(f"Test data shape: {X_test.shape}")
            
            logger.info("Evaluating model on test data")
            test_metrics = model.evaluate(X_test, y_test)
            
            # Print a summary of test metrics
            logger.info("Test metrics summary:")
            logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
            logger.info(f"  Precision: {test_metrics['precision']:.4f}")
            logger.info(f"  Recall: {test_metrics['recall']:.4f}")
            logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
            logger.info(f"  ROC AUC: {test_metrics['roc_auc']:.4f}")
            
        # Save the final model results in a summary JSON
        summary = {
            "training": {
                "timestamp": timestamp,
                "duration_seconds": time.time() - start_time,
                "config_file": args.config,
                "hyperparam_tuning": tune_hyperparameters
            },
            "data": {
                "train_samples": len(y_train),
                "val_samples": len(y_val),
                "test_samples": len(y_test) if args.test else "not evaluated",
                "features": X_train.shape[1],
                "class_distribution": {
                    "train": dict(pd.Series(y_train).value_counts().to_dict()),
                    "val": dict(pd.Series(y_val).value_counts().to_dict()),
                    "test": dict(pd.Series(y_test).value_counts().to_dict()) if args.test else "not evaluated"
                }
            },
            "model": {
                "optimal_threshold": model.optimal_threshold,
                "best_params": model.best_params if hasattr(model, 'best_params') and model.best_params else "not tuned",
            },
            "metrics": {
                "validation": {
                    "f1": val_metrics['f1'],
                    "precision": val_metrics['precision'],
                    "recall": val_metrics['recall'],
                    "accuracy": val_metrics['accuracy'],
                    "roc_auc": val_metrics['roc_auc']
                },
                "test": {
                    "f1": test_metrics['f1'],
                    "precision": test_metrics['precision'],
                    "recall": test_metrics['recall'],
                    "accuracy": test_metrics['accuracy'],
                    "roc_auc": test_metrics['roc_auc']
                } if args.test else "not evaluated"
            }
        }
        
        # Save summary
        with open(os.path.join(config['data_config']['save_dir'], "training_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")
        logger.info(f"Results saved to {config['data_config']['save_dir']}")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
