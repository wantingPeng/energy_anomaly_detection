"""
Hyperparameter tuning for XGBoost using Optuna.

This script performs automatic hyperparameter optimization for XGBoost-based
time series anomaly detection using Optuna with GPU acceleration support.
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
from src.training.xgboost.xgboost_model import XGBoostAnomalyDetector
from src.training.xgboost.dataloader import create_xgboost_data


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


class XGBoostOptunaTuner:
    """
    Optuna-based hyperparameter tuning for XGBoost anomaly detection.
    """
    
    def __init__(self, config, data_dict, optimization_metric='f1'):
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
        logger.info(f"Initialized XGBoostOptunaTuner (GPU mode)")
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
                params[param_name] = trial.suggest_int(param_name, low, high, step=step)
                
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
        
        # Force GPU for hyperparameter tuning
        tuning_config['model']['tree_method'] = 'gpu_hist'
        tuning_config['model']['device'] = 'cuda'
        
        # Get early stopping rounds from config (use training config as base, allow override in optuna config)
        early_stopping = self.config.get('optuna', {}).get('early_stopping_rounds', 
                                         self.config.get('training', {}).get('early_stopping_rounds', 30))
        tuning_config['training']['early_stopping_rounds'] = early_stopping
        
        try:
            # Create and train model
            model = XGBoostAnomalyDetector(tuning_config)
            model.train(
                X_train=self.data_dict['X_train'],
                y_train=self.data_dict['y_train'],
                X_val=self.data_dict['X_val'],
                y_val=self.data_dict['y_val'],
                feature_names=self.data_dict['feature_names']
            )
            
            # Optimize threshold
            model.optimize_threshold(
                X_val=self.data_dict['X_val'],
                y_val=self.data_dict['y_val'],
                metric='f1'
            )
            
            # Evaluate on validation set
            val_metrics = model.evaluate(
                X=self.data_dict['X_val'],
                y=self.data_dict['y_val'],
                split_name="Validation"
            )
            
            # Calculate point-adjusted metrics and overwrite originals for reporting/optimization
            val_preds = model.predict(self.data_dict['X_val'])
            val_labels = self.data_dict['y_val']
            
            # Apply point adjustment
            labels_adj, preds_adj = point_adjustment(val_labels, val_preds)
            
            # Calculate adjusted metrics and overwrite core metrics
            from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
            adj_accuracy = accuracy_score(labels_adj, preds_adj)
            adj_precision, adj_recall, adj_f1, _ = precision_recall_fscore_support(
                labels_adj, preds_adj, average='binary', zero_division=0
            )
            val_metrics['accuracy'] = adj_accuracy
            val_metrics['precision'] = adj_precision
            val_metrics['recall'] = adj_recall
            val_metrics['f1'] = adj_f1
            val_metrics['confusion_matrix'] = confusion_matrix(labels_adj, preds_adj)
            
            # Choose optimization metric from val_metrics (consistent access pattern)
            optimization_value = val_metrics[self.optimization_metric]

            logger.info(
                f"Trial {trial.number}: AUPRC={val_metrics['auprc']:.4f}, "
                f"F1={val_metrics['f1']:.4f}, Prec={val_metrics['precision']:.4f}, "
                f"Rec={val_metrics['recall']:.4f} | Optimizing {self.optimization_metric}={optimization_value:.4f}"
            )
            
            # Report intermediate values for pruning
            trial.report(optimization_value, model.best_iteration)
            
            # Handle pruning based on the intermediate value
            if trial.should_prune():
                logger.info(f"Trial {trial.number} pruned.")
                raise optuna.TrialPruned()
            
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
        logger.info(f"Device: GPU (CUDA)")
        logger.info(f"Optimization metric: {self.optimization_metric}")
        logger.info(f"Note: Reported F1/Precision/Recall are point-adjusted values")
        logger.info("=" * 60)
        
        # Create study
        if study_name is None:
            study_name = f"xgboost_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Get pruner configuration from config (MedianPruner)
        pruner_config = self.config.get('optuna', {}).get('pruner', {})
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=pruner_config.get('n_startup_trials', 5),
            n_warmup_steps=pruner_config.get('n_warmup_steps', 10),
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
            f.write(f"Best value ({self.optimization_metric}): {self.best_value:.4f}\n\n")
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
        # Load configuration from the model directory (not from args.config)
        logger.info("\n" + "=" * 60)
        logger.info("LOADING PRE-TRAINED MODEL (SKIPPING TUNING)")
        logger.info("=" * 60)
        logger.info(f"Model path: {args.model_path}")
        
        # Try to find config.json - check both root dir and best_model subdir
        model_config_path = os.path.join(args.model_path, 'config.json')
        best_model_config_path = os.path.join(args.model_path, 'best_model', 'config.json')
        
        if os.path.exists(model_config_path):
            config_path = model_config_path
        elif os.path.exists(best_model_config_path):
            config_path = best_model_config_path
            logger.info("Config not found in root, using best_model/config.json")
        else:
            raise FileNotFoundError(
                f"config.json not found in:\n"
                f"  - {model_config_path}\n"
                f"  - {best_model_config_path}\n"
                f"Please ensure the model directory contains config.json"
            )
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded model configuration from: {config_path}")
        
        # Create output directory for evaluation results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = args.experiment_name or f"eval_pretrained_{timestamp}"
        output_dir = config['paths']['output_dir']
        tuning_dir = os.path.join(output_dir, 'hyperparameter_tuning', experiment_name)
        os.makedirs(tuning_dir, exist_ok=True)
        logger.info(f"Evaluation results will be saved to: {tuning_dir}")
        
        # Load data using model's config
        data_path = config['data']['data_path']
        logger.info(f"Loading data from: {data_path}")
        data_loader, data_dict = create_xgboost_data(data_path, config)
        
        logger.info(f"Data loaded:")
        logger.info(f"  Train samples: {len(data_dict['y_train'])}")
        logger.info(f"  Validation samples: {len(data_dict['y_val'])}")
        logger.info(f"  Test samples: {len(data_dict['y_test'])}")
        logger.info(f"  Features: {len(data_dict['feature_names'])}")
        
        # Create model instance and load weights
        # Try to find the model directory (check both root and best_model subdir)
        model_json_path = os.path.join(args.model_path, 'xgboost_model.json')
        best_model_json_path = os.path.join(args.model_path, 'best_model', 'xgboost_model.json')
        
        if os.path.exists(model_json_path):
            load_path = args.model_path
        elif os.path.exists(best_model_json_path):
            load_path = os.path.join(args.model_path, 'best_model')
            logger.info(f"Model files found in best_model/ subdirectory")
        else:
            raise FileNotFoundError(
                f"xgboost_model.json not found in:\n"
                f"  - {model_json_path}\n"
                f"  - {best_model_json_path}\n"
                f"Please ensure the model directory contains xgboost_model.json"
            )
        
        model = XGBoostAnomalyDetector(config)
        model.load_model(load_path)
        
        logger.info(f"Model loaded successfully from: {load_path}")
        logger.info(f"  Optimal threshold: {model.optimal_threshold}")
        logger.info(f"  Best iteration: {model.best_iteration}")
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
        
        data_loader, data_dict = create_xgboost_data(data_path, config)
        
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
        logger.info(f"  device: GPU (CUDA)")
        logger.info(f"  optimization_metric: {optimization_metric}")
        
        # Create tuner (always use GPU for tuning)
        tuner = XGBoostOptunaTuner(
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
        
        test_metrics = best_model.evaluate(
            X=data_dict['X_test'],
            y=data_dict['y_test'],
            split_name="Test"
        )
        
        # Always apply point adjustment on test set for comprehensive evaluation
        test_preds = best_model.predict(data_dict['X_test'])
        test_labels = data_dict['y_test']
        
        # Apply point adjustment
        labels_adj, preds_adj = point_adjustment(test_labels, test_preds)
        
        # Calculate adjusted metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        adj_accuracy = accuracy_score(labels_adj, preds_adj)
        adj_precision, adj_recall, adj_f1, _ = precision_recall_fscore_support(
            labels_adj, preds_adj, average='binary', zero_division=0
        )
        
        test_metrics['adj_accuracy'] = float(adj_accuracy)
        test_metrics['adj_precision'] = float(adj_precision)
        test_metrics['adj_recall'] = float(adj_recall)
        test_metrics['adj_f1'] = float(adj_f1)
        
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
        
 
        
        logger.info(f"\nTest Performance (With Point Adjustment):")
        logger.info(f"  F1 Score: {test_metrics['adj_f1']:.4f}")
        logger.info(f"  Precision: {test_metrics['adj_precision']:.4f}")
        logger.info(f"  Recall: {test_metrics['adj_recall']:.4f}")
        logger.info(f"  Accuracy: {test_metrics['adj_accuracy']:.4f}")
        
        logger.info(f"\nResults saved to: {tuning_dir}")
        logger.info("=" * 60)
    else:
        logger.warning("No best model found. All trials may have failed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for XGBoost using Optuna"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/xgboost_tuning_config.yaml",
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

