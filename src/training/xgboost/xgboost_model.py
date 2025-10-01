"""
XGBoost model wrapper for time series anomaly detection.

This module provides a wrapper class for XGBoost model with utilities
for training, prediction, and evaluation.
"""

import os
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import pickle
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger


class XGBoostAnomalyDetector:
    """
    XGBoost model wrapper for anomaly detection in time series data.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize XGBoost model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.evaluation_config = config.get('evaluation', {})
        
        # Model parameters
        self.params = self._prepare_params()
        
        # Model and results
        self.model = None
        self.best_iteration = None
        self.feature_names = None
        self.optimal_threshold = 0.5
        self.training_history = {}
        
    def _prepare_params(self) -> Dict:
        """
        Prepare XGBoost parameters from config.
        
        Returns:
            Dictionary of XGBoost parameters
        """
        params = {
            'objective': self.model_config.get('objective', 'binary:logistic'),
            'eval_metric': self.model_config.get('eval_metric', ['logloss', 'auc', 'aucpr']),
            'tree_method': self.model_config.get('tree_method', 'hist'),
            'device': self.model_config.get('device', 'cpu'),
            'max_depth': self.model_config.get('max_depth', 6),
            'min_child_weight': self.model_config.get('min_child_weight', 1),
            'gamma': self.model_config.get('gamma', 0.1),
            'alpha': self.model_config.get('alpha', 0.0),
            'lambda': self.model_config.get('lambda', 1.0),
            'learning_rate': self.model_config.get('learning_rate', 0.1),
            'subsample': self.model_config.get('subsample', 0.8),
            'colsample_bytree': self.model_config.get('colsample_bytree', 0.8),
            'colsample_bylevel': self.model_config.get('colsample_bylevel', 0.8),
            'colsample_bynode': self.model_config.get('colsample_bynode', 0.8),
            'random_state': self.model_config.get('random_state', 42),
            'n_jobs': self.model_config.get('n_jobs', -1),
            'verbosity': self.model_config.get('verbosity', 1)
        }
        
        # Handle scale_pos_weight
        scale_pos_weight = self.model_config.get('scale_pos_weight', 'auto')
        if scale_pos_weight != 'auto':
            params['scale_pos_weight'] = scale_pos_weight
        
        return params
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str] = None,
        scale_pos_weight: float = None
    ):
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            scale_pos_weight: Weight for positive class (if auto in config)
        """
        logger.info("=== Training XGBoost Model ===")
        
        self.feature_names = feature_names
        
        # Update scale_pos_weight if auto
        if self.params.get('scale_pos_weight') is None and scale_pos_weight is not None:
            self.params['scale_pos_weight'] = scale_pos_weight
            logger.info(f"Using scale_pos_weight: {scale_pos_weight:.2f}")
        
        # Log parameters
        logger.info("XGBoost Parameters:")
        for key, value in self.params.items():
            logger.info(f"  {key}: {value}")
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
        
        # Training parameters
        num_boost_round = self.model_config.get('n_estimators', 300)
        early_stopping_rounds = self.training_config.get('early_stopping_rounds', 30)
        verbose_eval = self.training_config.get('verbose_eval', 10)
        
        # Evaluation list
        evals = [(dtrain, 'train'), (dval, 'val')]
        
        # Store evaluation results
        evals_result = {}
        
        # Train model
        logger.info(f"Training with {num_boost_round} rounds and early stopping at {early_stopping_rounds}")
        
        self.model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            evals_result=evals_result
        )
        
        self.best_iteration = self.model.best_iteration
        self.training_history = evals_result
        
        logger.info(f"Training completed at iteration {self.best_iteration}")
        logger.info(f"Best train loss: {evals_result['train']['logloss'][self.best_iteration]:.4f}")
        logger.info(f"Best val loss: {evals_result['val']['logloss'][self.best_iteration]:.4f}")
        
        if 'auc' in evals_result['val']:
            logger.info(f"Best val AUC: {evals_result['val']['auc'][self.best_iteration]:.4f}")
        if 'aucpr' in evals_result['val']:
            logger.info(f"Best val AUCPR: {evals_result['val']['aucpr'][self.best_iteration]:.4f}")
    
    def predict_proba(self, X: np.ndarray, feature_names: List[str] = None) -> np.ndarray:
        """
        Predict probabilities.
        
        Args:
            X: Features to predict
            feature_names: Feature names
            
        Returns:
            Predicted probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        dtest = xgb.DMatrix(X, feature_names=feature_names or self.feature_names)
        
        # Get best iteration predictions
        use_best_iteration = self.training_config.get('use_best_iteration', True)
        if use_best_iteration and self.best_iteration is not None:
            proba = self.model.predict(dtest, iteration_range=(0, self.best_iteration + 1))
        else:
            proba = self.model.predict(dtest)
        
        return proba
    
    def predict(self, X: np.ndarray, threshold: float = None, feature_names: List[str] = None) -> np.ndarray:
        """
        Predict binary labels.
        
        Args:
            X: Features to predict
            threshold: Decision threshold (uses optimal if None)
            feature_names: Feature names
            
        Returns:
            Predicted binary labels
        """
        proba = self.predict_proba(X, feature_names)
        threshold = threshold or self.optimal_threshold
        return (proba >= threshold).astype(int)
    
    def optimize_threshold(self, X_val: np.ndarray, y_val: np.ndarray, metric: str = 'f1') -> float:
        """
        Optimize decision threshold on validation set.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
            
        Returns:
            Optimal threshold
        """
        logger.info(f"=== Optimizing Threshold for {metric.upper()} ===")
        
        # Get probabilities
        proba = self.predict_proba(X_val)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_val, proba)
        
        # Calculate metrics for each threshold
        best_score = 0
        best_threshold = 0.5
        
        for i, threshold in enumerate(thresholds):
            preds = (proba >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_val, preds, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_val, preds, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_val, preds, zero_division=0)
            elif metric == 'accuracy':
                score = accuracy_score(y_val, preds)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.4f}")
        logger.info(f"Best {metric}: {best_score:.4f}")
        
        return best_threshold
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, split_name: str = "Test") -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Features
            y: True labels
            split_name: Name of the data split (for logging)
            
        Returns:
            Dictionary of metrics
        """
        logger.info(f"=== Evaluating on {split_name} Set ===")
        
        # Get predictions
        proba = self.predict_proba(X)
        preds = self.predict(X, threshold=self.optimal_threshold)
        
        # Calculate metrics
        accuracy = accuracy_score(y, preds)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        
        # AUC metrics
        try:
            auroc = roc_auc_score(y, proba)
        except:
            auroc = 0.0
            logger.warning("Could not calculate AUROC")
        
        try:
            auprc = average_precision_score(y, proba)
        except:
            auprc = 0.0
            logger.warning("Could not calculate AUPRC")
        
        # Confusion matrix
        cm = confusion_matrix(y, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auroc': auroc,
            'auprc': auprc,
            'confusion_matrix': cm,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'threshold': self.optimal_threshold
        }
        
        # Log metrics
        logger.info(f"{split_name} Results (threshold={self.optimal_threshold:.4f}):")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  AUROC: {auroc:.4f}")
        logger.info(f"  AUPRC: {auprc:.4f}")
        logger.info(f"  Confusion Matrix:")
        logger.info(f"    TN={tn}, FP={fp}")
        logger.info(f"    FN={fn}, TP={tp}")
        
        return metrics
    
    def get_feature_importance(self, importance_type: str = 'gain') -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('weight', 'gain', 'cover')
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        importance_dict = self.model.get_score(importance_type=importance_type)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': list(importance_dict.keys()),
            'importance': list(importance_dict.values())
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        logger.info(f"Top 10 features by {importance_type}:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {idx+1}. {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_model(self, save_dir: str):
        """
        Save model and related artifacts.
        
        Args:
            save_dir: Directory to save model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save XGBoost model
        model_path = os.path.join(save_dir, 'xgboost_model.json')
        self.model.save_model(model_path)
        logger.info(f"Saved XGBoost model to: {model_path}")
        
        # Save model metadata
        metadata = {
            'best_iteration': self.best_iteration,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': self.feature_names,
            'params': self.params,
            'training_history': self.training_history
        }
        
        metadata_path = os.path.join(save_dir, 'model_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved model metadata to: {metadata_path}")
        
        # Save config
        config_path = os.path.join(save_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        logger.info(f"Saved config to: {config_path}")
    
    def load_model(self, save_dir: str):
        """
        Load model and related artifacts.
        
        Args:
            save_dir: Directory containing saved model
        """
        logger.info(f"Loading model from: {save_dir}")
        
        # Load XGBoost model
        model_path = os.path.join(save_dir, 'xgboost_model.json')
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        logger.info(f"Loaded XGBoost model from: {model_path}")
        
        # Load metadata
        metadata_path = os.path.join(save_dir, 'model_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.best_iteration = metadata['best_iteration']
        self.optimal_threshold = metadata['optimal_threshold']
        self.feature_names = metadata['feature_names']
        self.params = metadata['params']
        self.training_history = metadata['training_history']
        
        logger.info(f"Loaded model metadata from: {metadata_path}")
        logger.info(f"  Best iteration: {self.best_iteration}")
        logger.info(f"  Optimal threshold: {self.optimal_threshold}")
        logger.info(f"  Number of features: {len(self.feature_names)}")

