"""
Hyperparameter tuning for Random Forest using Optuna.

This script performs automatic hyperparameter optimization for Random Forest-based
time series anomaly detection using Optuna.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import joblib
import json
import copy

import optuna

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.training.random_forest.random_forest_model import RandomForestAnomalyDetector
from src.training.random_forest.dataloader import create_random_forest_data


def point_adjustment(gt, pred):
    """
    Point adjustment strategy for anomaly detection evaluation.
    
    This function adjusts predictions for time series anomaly detection by:
    1. If any point in an anomaly segment is detected, the entire segment is considered detected
    2. Reduces penalty for slightly delayed or early detection
    
    Args:
        gt: Ground truth labels (numpy array)
        pred: Predicted labels (numpy array)
    
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


class RandomForestOptunaTuner:
    """
    Optuna-based hyperparameter tuning for Random Forest anomaly detection.
    """
    
    def __init__(self, config, data_dict, optimization_metric='adj_f1'):
        """
        Initialize the tuner.
        
        Args:
            config: Configuration dictionary
            data_dict: Dictionary containing train/val/test data
            optimization_metric: Metric to optimize ('adj_f1' or 'auprc')
        """
        self.config = config
        self.data_dict = data_dict
        self.optimization_metric = optimization_metric
        
        # Store best model and params (we maintain our own to ensure consistency)
        self.best_model = None
        self.best_threshold = None
        self.best_value = None
        self.best_params = None
        self.best_trial_number = None
        
        # Log search space from config
        search_space = config.get('optuna', {}).get('search_space', {})
        logger.info(f"Initialized RandomForestOptunaTuner")
        logger.info(f"Optimization metric: {optimization_metric}")
        logger.info(f"Search space loaded from config with {len(search_space)} parameters:")
        for param_name, param_config in search_space.items():
            logger.info(f"  {param_name}: {param_config}")
    
    def objective(self, trial):
        """
        Optuna objective function for hyperparameter optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation metric to optimize
        """
        # Get search space from config
        search_space = self.config.get('optuna', {}).get('search_space', {})
        
        # Define hyperparameter search space from config
        params = {}
        for param_name, param_config in search_space.items():
            param_type = param_config.get('type', 'float')
            
            if param_type == 'int':
                low = param_config.get('low', 1)
                high = param_config.get('high', 10)
                step = param_config.get('step', 1)
                value = trial.suggest_int(param_name, low, high, step=step)
                
                # Special handling for max_features: convert percentage to decimal
                if param_name == 'max_features':
                    # Convert integer percentage (30, 40, 50, etc.) to decimal (0.3, 0.4, 0.5, etc.)
                    params[param_name] = value / 100.0
                else:
                    params[param_name] = value
                
            elif param_type == 'float':
                low = param_config.get('low', 0.0)
                high = param_config.get('high', 1.0)
                log = param_config.get('log', False)
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)
                
            elif param_type == 'categorical':
                choices = param_config.get('choices', [])
                params[param_name] = trial.suggest_categorical(param_name, choices)
        
        # Update config with suggested parameters (use deep copy to avoid modifying original config)
        tuning_config = copy.deepcopy(self.config)
        tuning_config['model'].update(params)
        
        try:
            # Create and train model
            model = RandomForestAnomalyDetector(tuning_config)
            model.train(
                X_train=self.data_dict['X_train'],
                y_train=self.data_dict['y_train'],
                X_val=self.data_dict['X_val'],
                y_val=self.data_dict['y_val'],
                feature_names=self.data_dict['feature_names'],
                class_weights=self.data_dict['class_weights']
            )
            
            # Get predictions for point adjustment
            val_preds = model.predict(self.data_dict['X_val'])
            val_labels = self.data_dict['y_val']
            
            # Apply point adjustment
            labels_adj, preds_adj = point_adjustment(val_labels, val_preds)
            
            # Calculate adjusted metrics (use as primary metrics)
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
            
            # Get probability predictions for AUPRC/AUROC
            val_proba = model.predict_proba(self.data_dict['X_val'])
            
            # Primary metrics are adjusted metrics
            accuracy = accuracy_score(labels_adj, preds_adj)
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_adj, preds_adj, average='binary', zero_division=0
            )
            auprc = average_precision_score(val_labels, val_proba)
            auroc = roc_auc_score(val_labels, val_proba)
            
            # Store metrics
            val_metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auprc': auprc,
                'auroc': auroc,
                'threshold': model.optimal_threshold
            }
            
            # Choose optimization metric from val_metrics (consistent access pattern)
            optimization_value = val_metrics[self.optimization_metric]
            
            logger.info(f"Trial {trial.number}: "
                       f"AUPRC={val_metrics['auprc']:.4f}, F1={val_metrics['f1']:.4f}, "
                       f"Precision={val_metrics['precision']:.4f}, Recall={val_metrics['recall']:.4f} | "
                       f"Optimizing {self.optimization_metric}={optimization_value:.4f}")
            
            # Store model if this is the best trial so far (use our own best_value for consistency)
            if self.best_value is None or optimization_value > self.best_value:
                self.best_model = model
                self.best_threshold = model.optimal_threshold
                self.best_value = optimization_value
                self.best_params = params  # Store the params that led to this best model
                self.best_trial_number = trial.number  # Store the trial number
                logger.info(f"  → New best model found! (value={optimization_value:.4f})")
            
            return optimization_value
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed with error: {str(e)}")
            raise optuna.TrialPruned()
    
    def tune(self, n_trials=100, timeout=None, n_jobs=1, study_name=None):
        """
        Run hyperparameter tuning.
        
        Args:
            n_trials: Number of trials to run
            timeout: Timeout in seconds (None for no timeout)
            n_jobs: Number of parallel jobs (1 for sequential)
            study_name: Name of the study
            
        Returns:
            Optuna study object
        """
        logger.info("=" * 60)
        logger.info("Starting Hyperparameter Tuning with Optuna")
        logger.info("=" * 60)
        logger.info(f"Number of trials: {n_trials}")
        logger.info(f"Timeout: {timeout}")
        logger.info(f"Parallel jobs: {n_jobs}")
        logger.info(f"Optimization metric: {self.optimization_metric}")
        logger.info(f"Note: All metrics (AUPRC, F1, Adjusted F1) are calculated for every trial")
        logger.info("=" * 60)
        
        # Create study
        if study_name is None:
            study_name = f"random_forest_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get pruner configuration from config (MedianPruner)
        pruner_config = self.config.get('optuna', {}).get('pruner', {})
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_config.get('n_startup_trials', 5),
            n_warmup_steps=pruner_config.get('n_warmup_steps', 0),
            interval_steps=pruner_config.get('interval_steps', 1)
        )
        
        # Get sampler configuration (TPESampler)
        sampler_config = self.config.get('optuna', {}).get('sampler', {})
        sampler = optuna.samplers.TPESampler(seed=sampler_config.get('seed', 42))
        
        # Get direction from config
        direction = self.config.get('optuna', {}).get('direction', 'maximize')
        
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            pruner=pruner,
            sampler=sampler
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=True
        )
        
        # Log results
        logger.info("=" * 60)
        logger.info("Hyperparameter Tuning Completed")
        logger.info("=" * 60)
        logger.info(f"Number of finished trials: {len(study.trials)}")
        logger.info(f"Best trial value ({self.optimization_metric}): {self.best_value:.4f}")
        logger.info(f"Best hyperparameters:")
        for key, value in self.best_params.items():
            logger.info(f"  {key}: {value}")
        logger.info("=" * 60)
        
        return study
    
    def save_results(self, study, save_dir):
        """
        Save tuning results.
        
        Args:
            study: Optuna study object
            save_dir: Directory to save results
        """
        os.makedirs(save_dir, exist_ok=True)
        
        logger.info(f"Saving tuning results to: {save_dir}")
        
        # Save study object
        study_path = os.path.join(save_dir, 'study.pkl')
        joblib.dump(study, study_path)
        logger.info(f"Saved study object to: {study_path}")
        
        # Save best parameters as YAML (use our own best_params for consistency)
        best_params_path = os.path.join(save_dir, 'best_params.yaml')
        with open(best_params_path, 'w') as f:
            yaml.dump(self.best_params, f, default_flow_style=False)
        logger.info(f"Saved best parameters to: {best_params_path}")
        
        # Save best parameters as JSON
        best_params_json_path = os.path.join(save_dir, 'best_params.json')
        with open(best_params_json_path, 'w') as f:
            json.dump(self.best_params, f, indent=2)
        
        # Save trials dataframe
        trials_df = study.trials_dataframe()
        trials_path = os.path.join(save_dir, 'trials.csv')
        trials_df.to_csv(trials_path, index=False)
        logger.info(f"Saved trials to: {trials_path}")
        
        # Save summary
        summary_path = os.path.join(save_dir, 'summary.txt')
        with open(summary_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Optuna Hyperparameter Tuning Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Study name: {study.study_name}\n")
            f.write(f"Number of trials: {len(study.trials)}\n")
            f.write(f"Best trial number: {self.best_trial_number}\n")
            f.write(f"Best value ({self.optimization_metric}): {self.best_value:.4f}\n")
            f.write(f"Data path: {self.config['data']['data_path']}\n\n")
            f.write("Best hyperparameters:\n")
            for key, value in self.best_params.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n" + "=" * 60 + "\n")
        logger.info(f"Saved summary to: {summary_path}")
        
        logger.info(f"All results saved to: {save_dir}")


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
    Main tuning function.
    
    Args:
        args: Command line arguments
    """
    # Check if loading pre-trained model or running tuning
    if args.model_path:
        # Load pre-trained model (not running tuning)
        logger.info("\n" + "=" * 60)
        logger.info("LOADING PRE-TRAINED MODEL (SKIPPING TUNING)")
        logger.info("=" * 60)
        logger.info(f"Model path: {args.model_path}")
        
        # Load config from the model directory
        model_config_path = os.path.join(args.model_path, 'model_params.pkl')
        import pickle
        with open(model_config_path, 'rb') as f:
            model_params = pickle.load(f)
        
        # Reconstruct config
        config = {
            'model': {
                'n_estimators': model_params['n_estimators'],
                'max_depth': model_params['max_depth'],
                'min_samples_split': model_params['min_samples_split'],
                'min_samples_leaf': model_params['min_samples_leaf'],
                'max_features': model_params['max_features'],
                'bootstrap': model_params['bootstrap'],
                'oob_score': model_params['oob_score'],
                'class_weight': model_params['class_weight']
            },
            'paths': {
                'output_dir': 'experiments/random_forest'
            },
            'data': {}  # Will be loaded from args.config if provided
        }
        
        # Load data config from args.config if provided
        if args.config:
            full_config = load_config(args.config)
            config['data'] = full_config.get('data', {})
        
        logger.info(f"Loaded model configuration")
        
        # Create output directory for evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = args.experiment_name or f"eval_pretrained_{timestamp}"
        output_dir = config['paths']['output_dir']
        tuning_dir = os.path.join(output_dir, 'hyperparameter_tuning', experiment_name)
        os.makedirs(tuning_dir, exist_ok=True)
        logger.info(f"Evaluation results will be saved to: {tuning_dir}")
        
        # Load data using config
        data_path = config['data'].get('data_path')
        if not data_path:
            raise ValueError("data_path must be specified in config when loading pre-trained model")
        
        logger.info(f"Loading data from: {data_path}")
        data_loader, data_dict = create_random_forest_data(data_path, config)
        
        logger.info(f"Data loaded:")
        logger.info(f"  Train samples: {len(data_dict['y_train'])}")
        logger.info(f"  Validation samples: {len(data_dict['y_val'])}")
        logger.info(f"  Test samples: {len(data_dict['y_test'])}")
        logger.info(f"  Features: {len(data_dict['feature_names'])}")
        
        # Load model
        model = RandomForestAnomalyDetector.load_model(args.model_path)
        
        logger.info(f"Model loaded successfully")
        logger.info(f"  Optimal threshold: {model.optimal_threshold}")
        logger.info(f"  Features: {len(model.feature_names)}")
        
        # Use the loaded model as best model
        best_model = model
        best_threshold = model.optimal_threshold
        
        # Create dummy values for saving (not applicable when loading)
        best_params = None
        best_value = None
        
    else:
        # Run hyperparameter tuning - load config from args.config
        config = load_config(args.config)
        
        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = args.experiment_name or f"optuna_tuning_{timestamp}"
        
        output_dir = config['paths']['output_dir']
        tuning_dir = os.path.join(output_dir, 'hyperparameter_tuning', experiment_name)
        os.makedirs(tuning_dir, exist_ok=True)
        
        logger.info(f"Tuning directory: {tuning_dir}")
        
        # Load data
        data_path = config['data']['data_path']
        logger.info(f"Loading data from: {data_path}")
        
        data_loader, data_dict = create_random_forest_data(data_path, config)
        
        logger.info(f"Data loaded:")
        logger.info(f"  Train samples: {len(data_dict['y_train'])}")
        logger.info(f"  Validation samples: {len(data_dict['y_val'])}")
        logger.info(f"  Test samples: {len(data_dict['y_test'])}")
        logger.info(f"  Features: {len(data_dict['feature_names'])}")
        
        # Get tuning parameters from config
        optuna_config = config.get('optuna', {})
        optimization_metric = optuna_config.get('optimization_metric', 'adj_f1')
        n_trials = optuna_config.get('n_trials', 100)
        timeout = optuna_config.get('timeout', None)
        n_jobs = optuna_config.get('n_jobs', 1)
        
        logger.info(f"Tuning configuration from config file:")
        logger.info(f"  n_trials: {n_trials}")
        logger.info(f"  timeout: {timeout}")
        logger.info(f"  n_jobs: {n_jobs}")
        logger.info(f"  optimization_metric: {optimization_metric}")
        
        # Create tuner
        tuner = RandomForestOptunaTuner(
            config=config,
            data_dict=data_dict,
            optimization_metric=optimization_metric
        )
        
        # Run tuning
        study = tuner.tune(
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            study_name=experiment_name
        )
        
        # Save results
        tuner.save_results(study, tuning_dir)
        
        # Use tuner's best model
        best_model = tuner.best_model
        best_threshold = tuner.best_threshold
        best_params = tuner.best_params
        best_value = tuner.best_value
    
    # Evaluate best model on test set
    if best_model is not None:
        logger.info("\n" + "=" * 60)
        logger.info("Evaluating Best Model on Test Set")
        logger.info("=" * 60)
        
        # Get predictions for point adjustment
        test_preds = best_model.predict(data_dict['X_test'])
        test_labels = data_dict['y_test']
        
        # Apply point adjustment
        labels_adj, preds_adj = point_adjustment(test_labels, test_preds)
        
        # Calculate adjusted metrics (use as primary metrics)
        from sklearn.metrics import (
            accuracy_score, precision_recall_fscore_support, 
            roc_auc_score, average_precision_score, confusion_matrix
        )
        
        # Get probability predictions for AUPRC/AUROC
        test_proba = best_model.predict_proba(data_dict['X_test'])
        
        # Primary metrics are adjusted metrics
        accuracy = accuracy_score(labels_adj, preds_adj)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_adj, preds_adj, average='binary', zero_division=0
        )
        auprc = average_precision_score(test_labels, test_proba)
        auroc = roc_auc_score(test_labels, test_proba)
        cm = confusion_matrix(labels_adj, preds_adj)
        
        # Get confusion matrix components
        tn, fp, fn, tp = cm.ravel()
        
        # Store metrics
        test_metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'auprc': float(auprc),
            'auroc': float(auroc),
            'threshold': float(best_threshold),
            'tn': float(tn),
            'fp': float(fp),
            'fn': float(fn),
            'tp': float(tp)
        }
        
        # Log metrics
        logger.info(f"\nTest Performance (Point-Adjusted Metrics):")
        logger.info(f"  Threshold: {best_threshold:.4f}")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  AUPRC: {auprc:.4f}")
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"  {cm}")
        logger.info(f"  True Positives: {tp}")
        logger.info(f"  False Positives: {fp}")
        logger.info(f"  True Negatives: {tn}")
        logger.info(f"  False Negatives: {fn}")
        
        # Save test results (with adjusted metrics)
        test_results = {
            'test_metrics': test_metrics,
            'best_params': best_params,
            'best_threshold': best_threshold,
            'best_trial_value': best_value
        }
        
        test_results_path = os.path.join(tuning_dir, 'test_results.json')
        
        def convert_numpy_types(obj):
            """Convert numpy types to Python native types"""
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
        
        test_results = convert_numpy_types(test_results)
        
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Saved test results to: {test_results_path}")
        
        # Save best model (only if running tuning, not when loading)
        if not args.model_path:
            model_dir = os.path.join(tuning_dir, 'best_model')
            os.makedirs(model_dir, exist_ok=True)
            best_model.save_model(model_dir)
            logger.info(f"Saved best model to: {model_dir}")
        
        # Print final summary
        logger.info("\n" + "=" * 60)
        if args.model_path:
            logger.info("EVALUATION COMPLETED (PRE-TRAINED MODEL)")
        else:
            logger.info("HYPERPARAMETER TUNING COMPLETED")
        logger.info("=" * 60)
        
        # Print tuning info only if tuning was performed
        if not args.model_path:
            logger.info(f"Best Trial Number: {tuner.best_trial_number}")
            logger.info(f"Best Validation Score ({tuner.optimization_metric}): {best_value:.4f}")
        
        logger.info(f"\nTest Performance Summary (Point-Adjusted):")
        logger.info(f"  F1 Score: {test_metrics['f1']:.4f}")
        logger.info(f"  Precision: {test_metrics['precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['recall']:.4f}")
        logger.info(f"  AUPRC: {test_metrics['auprc']:.4f}")
        logger.info(f"  AUROC: {test_metrics['auroc']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        
        logger.info(f"\nResults saved to: {tuning_dir}")
        logger.info("=" * 60)
    else:
        logger.warning("No best model found. All trials may have failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for Random Forest using Optuna"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/random_forest_tuning_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the tuning experiment (optional, will use timestamp if not provided)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to pre-trained model directory (skip tuning if provided)"
    )
    
    args = parser.parse_args()
    main(args)

