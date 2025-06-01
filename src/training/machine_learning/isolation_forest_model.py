import os
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report
import joblib
from itertools import product
from src.utils.logger import logger
from src.training.visualizations.isolation_forest import (
    plot_anomaly_scores_distribution,
    plot_confusion_matrix,
    save_results
)
from datetime import datetime

class IsolationForestTrainer:
    def __init__(self, config_path="configs/isolation_forest.yaml"):
        """Initialize the Isolation Forest trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.best_model = None
        self.best_score = float('-inf')
        self.best_params = None
        
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _load_data(self):
        """Load training and validation data."""
        logger.info("Loading training and validation data...")
        train_data = pd.read_parquet(self.config['data']['train_path'])
        val_data = pd.read_parquet(self.config['data']['val_path'])
        
        # Remove excluded features
        exclude_cols = self.config['data']['features_to_exclude']
        X_train = train_data.drop(columns=exclude_cols)
        y_train = train_data['anomaly_label']
        X_val = val_data.drop(columns=exclude_cols)
        y_val = val_data['anomaly_label']
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def _create_model(self, params):
        """Create an Isolation Forest model with given parameters."""
        model_params = {
            'n_estimators': params['n_estimators'],
            'contamination': params['contamination'],
            'max_samples': params['max_samples'],
            'max_features': params['max_features'],
            'random_state': self.config['model']['random_state'],
            'n_jobs': -1  # Use all available cores
        }
        return IsolationForest(**model_params)
    
    def _calculate_metrics_from_confusion_matrix(self, y_true, y_pred):
        """Calculate metrics from confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # 计算各项指标
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'confusion_matrix': cm,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score
        }
        
        return metrics

    def _evaluate_model(self, model, X_val, y_val):
        """Evaluate model performance on validation set."""
        # Get anomaly scores (-1 for anomalies, 1 for normal)
        y_pred = model.predict(X_val)
        # Convert to 0 (normal) and 1 (anomaly)
        y_pred = (y_pred == -1).astype(int)
        
        # 计算指标
        metrics = self._calculate_metrics_from_confusion_matrix(y_val, y_pred)
        
        return metrics, y_pred
    
    def train_and_evaluate(self):
        """Train and evaluate Isolation Forest model with different hyperparameters."""
        X_train, y_train, X_val, y_val = self._load_data()
        
        # Generate all parameter combinations
        param_grid = self.config['hyperparameters']
        param_combinations = [dict(zip(param_grid.keys(), v)) 
                            for v in product(*param_grid.values())]
        
        results = []
        output_dir = self.config['output']['validation_report_path']
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for params in param_combinations:
            logger.info(f"Training model with parameters: {params}")
            model = self._create_model(params)
            
            # Fit model
            model.fit(X_train)
            
            # Evaluate
            metrics, y_pred = self._evaluate_model(model, X_val, y_val)
            
            results.append({
                'parameters': params,
                'metrics': metrics,
                'predictions': y_pred
            })
            
            # Update best model if current model is better
            if metrics['f1_score'] > self.best_score:
                self.best_score = metrics['f1_score']
                self.best_model = model
                self.best_params = params
                self.best_predictions = y_pred
                self.best_metrics = metrics
                logger.info(f"New best model found! F1 Score: {metrics['f1_score']:.4f}")
        
        # Generate visualizations for best model
        anomaly_scores = self.best_model.score_samples(X_val)
        
        # Plot anomaly scores distribution
        plot_anomaly_scores_distribution(
            anomaly_scores,
            y_val,
            os.path.join(output_dir, f'anomaly_scores_distribution_{timestamp}.png')
        )
        
        # Plot confusion matrix for best model
        plot_confusion_matrix(
            y_val,
            self.best_predictions,
            os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
        )
        
        # Save results to MD file in the same directory
        save_results(
            results, 
            os.path.join(output_dir, f'validation_results_{timestamp}.md')
        )
        
        # Save best model (也可以给模型文件添加时间戳)
        model_path = self.config['output']['model_path']
        model_dir = os.path.dirname(model_path)
        model_name = os.path.basename(model_path)
        base_name, ext = os.path.splitext(model_name)
        timestamped_model_path = os.path.join(model_dir, f'{base_name}_{timestamp}{ext}')
        
        os.makedirs(os.path.dirname(timestamped_model_path), exist_ok=True)
        joblib.dump(self.best_model, timestamped_model_path)
        logger.info(f"Best model saved to {timestamped_model_path}")
        
        return self.best_model, self.best_params

if __name__ == "__main__":
    trainer = IsolationForestTrainer()
    best_model, best_params = trainer.train_and_evaluate()