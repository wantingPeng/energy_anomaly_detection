"""
Random Forest model for time series anomaly detection.

This module implements a Random Forest classifier for supervised anomaly detection
in time series data.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix
)

# Add project root to path
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger


class RandomForestAnomalyDetector:
    """
    Random Forest anomaly detector for time series data.
    
    This is a supervised approach to anomaly detection using RandomForestClassifier.
    It requires labeled data (anomaly vs normal) for training.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Random Forest model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.model = None
        self.optimal_threshold = 0.5  # Default threshold
        self.feature_names = None
        
        # Model attributes
        self.random_state = self.model_config.get('random_state', 42)
        self.n_estimators = self.model_config.get('n_estimators', 100)
        self.max_depth = self.model_config.get('max_depth', None)
        self.min_samples_split = self.model_config.get('min_samples_split', 2)
        self.min_samples_leaf = self.model_config.get('min_samples_leaf', 1)
        self.max_features = self.model_config.get('max_features', 'sqrt')
        self.bootstrap = self.model_config.get('bootstrap', True)
        self.oob_score = self.model_config.get('oob_score', False)
        self.class_weight = self.model_config.get('class_weight', None)
        self.n_jobs = self.model_config.get('n_jobs', -1)
        self.verbose = self.model_config.get('verbose', 0)
        
        logger.info(f"Initialized Random Forest model with:")
        logger.info(f"  n_estimators: {self.n_estimators}")
        logger.info(f"  max_depth: {self.max_depth}")
        logger.info(f"  min_samples_split: {self.min_samples_split}")
        logger.info(f"  min_samples_leaf: {self.min_samples_leaf}")
        logger.info(f"  max_features: {self.max_features}")
        logger.info(f"  class_weight: {self.class_weight}")
        logger.info(f"  bootstrap: {self.bootstrap}")
        logger.info(f"  oob_score: {self.oob_score}")
        logger.info(f"  n_jobs: {self.n_jobs}")
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
        class_weights: Optional[Dict] = None,
    ):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            class_weights: Class weights for imbalanced data
        """
        # Save feature names
        self.feature_names = feature_names
        
        # Use provided class weights if available
        if class_weights is not None and self.class_weight is None:
            logger.info(f"Using provided class weights: {class_weights}")
            self.class_weight = class_weights
        
        # Create and train model
        logger.info("\nTraining Random Forest model...")
        logger.info(f"Training data shape: {X_train.shape}")
        
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            class_weight=self.class_weight
        )
        
        # Train the model with progress bar
        import time
        start_time = time.time()
        self.model.fit(X_train, y_train)
        train_time = time.time() - start_time
        logger.info(f"Training completed in {train_time:.2f} seconds")
        
        # Log OOB score if enabled
        if self.oob_score:
            logger.info(f"Out-of-bag score: {self.model.oob_score_:.4f}")
        
        # Evaluate on training data
        logger.info("\nEvaluating on training data:")
        train_metrics = self.evaluate(X_train, y_train, split_name="Training")
        
        # Evaluate on validation data
        if X_val is not None and y_val is not None:
            logger.info("\nEvaluating on validation data:")
            val_metrics = self.evaluate(X_val, y_val, split_name="Validation")
            
            # Optimize decision threshold
            if self.training_config.get('optimize_threshold', True):
                self.optimize_threshold(X_val, y_val)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from model.
        
        Args:
            X: Input features
            
        Returns:
            Probability of anomaly class (class 1)
        """
        if self.model is None:
            raise ValueError("Model not trained, call train() first")
        
        # Get class probabilities and return probability for class 1 (anomaly)
        probas = self.model.predict_proba(X)
        return probas[:, 1]
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict binary labels (0=normal, 1=anomaly).
        
        Args:
            X: Input features
            
        Returns:
            Binary predictions (0=normal, 1=anomaly)
        """
        if self.model is None:
            raise ValueError("Model not trained, call train() first")
        
        if self.optimal_threshold == 0.5:
            # Use the model's default threshold
            return self.model.predict(X)
        else:
            # Use optimized threshold
            probas = self.predict_proba(X)
            return (probas >= self.optimal_threshold).astype(int)
    
    def optimize_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "f1"
    ) -> float:
        """
        Optimize decision threshold on validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize (f1, precision, recall)
            
        Returns:
            Optimal threshold
        """
        logger.info(f"\nOptimizing decision threshold for {metric}...")
        
        # Get probabilities
        y_proba = self.predict_proba(X_val)
        
        # Try different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -1
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            if metric == "precision":
                score = precision_score(y_val, y_pred, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_val, y_pred)
            else:  # default to f1
                score = f1_score(y_val, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        logger.info(f"Optimal threshold: {best_threshold:.4f} ({metric}={best_score:.4f})")
        self.optimal_threshold = best_threshold
        return best_threshold
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        split_name: str = "Test"
    ) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Input features
            y: True labels
            split_name: Name of data split for logging
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained, call train() first")
        
        logger.info(f"\n--- {split_name} Evaluation ---")
        
        # Get probabilistic predictions
        y_proba = self.predict_proba(X)
        
        # Get binary predictions
        y_pred = self.predict(X)
        
        # Classification metrics
        auc_roc = roc_auc_score(y, y_proba)
        auprc = average_precision_score(y, y_proba)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred, zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        # Log metrics
        logger.info(f"Threshold: {self.optimal_threshold:.4f}")
        logger.info(f"AUC-ROC: {auc_roc:.4f}")
        logger.info(f"AUC-PR: {auprc:.4f}")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"Confusion Matrix:")
        logger.info(f"{cm}")
        
        # Calculate true positives, false positives, etc.
        tn, fp, fn, tp = cm.ravel()
        
        logger.info(f"True Positives: {tp}")
        logger.info(f"False Positives: {fp}")
        logger.info(f"True Negatives: {tn}")
        logger.info(f"False Negatives: {fn}")
        
        # Return metrics
        return {
            'auc_roc': auc_roc,
            'auprc': auprc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'threshold': self.optimal_threshold,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    
    def get_feature_importance(self, importance_type: str = 'impurity') -> pd.DataFrame:
        """
        Get feature importance.
        
        Args:
            importance_type: Type of importance ('impurity' or 'permutation')
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not trained or feature names not provided")
        
        if importance_type == 'permutation':
            # Permutation importance requires validation data
            raise NotImplementedError("Permutation importance not implemented yet")
        
        # Get feature importances from model (default impurity-based)
        importances = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        logger.info(f"\nTop 10 features by importance:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return importance_df
    
    def save_model(self, model_dir: str):
        """
        Save the trained model.
        
        Args:
            model_dir: Directory to save model
        """
        if self.model is None:
            raise ValueError("Model not trained, call train() first")
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(model_dir, 'random_forest_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save model parameters
        params_path = os.path.join(model_dir, 'model_params.pkl')
        model_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'oob_score': self.oob_score,
            'class_weight': self.class_weight,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': self.feature_names
        }
        
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Model parameters saved to: {params_path}")
    
    @classmethod
    def load_model(cls, model_dir: str) -> 'RandomForestAnomalyDetector':
        """
        Load a trained model.
        
        Args:
            model_dir: Directory with saved model
            
        Returns:
            Loaded RandomForestAnomalyDetector instance
        """
        # Load model parameters
        params_path = os.path.join(model_dir, 'model_params.pkl')
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
        
        # Create config
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
            }
        }
        
        # Create instance
        instance = cls(config)
        
        # Load model
        model_path = os.path.join(model_dir, 'random_forest_model.joblib')
        instance.model = joblib.load(model_path)
        
        # Set other attributes
        instance.optimal_threshold = model_params['optimal_threshold']
        instance.feature_names = model_params['feature_names']
        
        logger.info(f"Loaded model from: {model_path}")
        logger.info(f"Optimal threshold: {instance.optimal_threshold:.4f}")
        
        return instance

