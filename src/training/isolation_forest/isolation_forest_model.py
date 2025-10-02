"""
Isolation Forest model for time series anomaly detection.

This module implements the Isolation Forest algorithm for anomaly detection
in time series data.
"""

import os
import pickle
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Union, Optional
from sklearn.ensemble import IsolationForest
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


class IsolationForestAnomalyDetector:
    """
    Isolation Forest anomaly detector for time series data.
    
    Isolation Forest is an unsupervised method that explicitly identifies anomalies
    rather than normal data points. It isolates observations by randomly selecting a feature
    and then randomly selecting a split value between the maximum and minimum values.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the Isolation Forest model.
        
        Args:
            config: Configuration dictionary with model parameters
        """
        self.config = config
        self.model_config = config.get('model', {})
        self.training_config = config.get('training', {})
        self.model = None
        self.optimal_threshold = 0.5
        self.feature_names = None
        self.contamination = self.model_config.get('contamination', 'auto')
        
        # Optional early stopping
        self.use_early_stopping = self.training_config.get('use_early_stopping', False)
        self.early_stopping_rounds = self.training_config.get('early_stopping_rounds', 10)
        
        # Model attributes
        self.random_state = self.model_config.get('random_state', 42)
        self.n_estimators = self.model_config.get('n_estimators', 100)
        self.max_samples = self.model_config.get('max_samples', 'auto')
        self.max_features = self.model_config.get('max_features', 1.0)
        self.bootstrap = self.model_config.get('bootstrap', False)
        self.n_jobs = self.model_config.get('n_jobs', -1)
        self.verbose = self.model_config.get('verbose', 0)
        
        logger.info(f"Initialized Isolation Forest model with:")
        logger.info(f"  n_estimators: {self.n_estimators}")
        logger.info(f"  max_samples: {self.max_samples}")
        logger.info(f"  max_features: {self.max_features}")
        logger.info(f"  contamination: {self.contamination}")
        logger.info(f"  bootstrap: {self.bootstrap}")
        logger.info(f"  n_jobs: {self.n_jobs}")
        
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: List[str] = None,
        contamination: Optional[float] = None,
    ):
        """
        Train the Isolation Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels (used for evaluation, not training)
            X_val: Validation features
            y_val: Validation labels
            feature_names: List of feature names
            contamination: Contamination parameter (proportion of outliers)
        """
        # Update contamination if provided
        if contamination is not None:
            self.contamination = contamination
            logger.info(f"Updated contamination to: {self.contamination}")
        
        # Save feature names
        self.feature_names = feature_names
        
        # Create and train model
        logger.info("\nTraining Isolation Forest model...")
        logger.info(f"Training data shape: {X_train.shape}")
        
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Train the model
        self.model.fit(X_train)
        
        logger.info("Training completed")
        
        # Evaluate on training data
        if y_train is not None:
            logger.info("\nEvaluating on training data:")
            self.evaluate(X_train, y_train, split_name="Training")
        
        # Evaluate on validation data
        if X_val is not None and y_val is not None:
            logger.info("\nEvaluating on validation data:")
            self.evaluate(X_val, y_val, split_name="Validation")
            
            # Optimize decision threshold
            if self.training_config.get('optimize_threshold', True):
                self.optimize_threshold(X_val, y_val)
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores from model.
        
        For Isolation Forest, the raw anomaly score is the negated decision function.
        Lower decision function values (more negative) indicate anomalies.
        We negate this so higher scores indicate anomalies.
        
        Args:
            X: Input features
            
        Returns:
            Anomaly scores where higher values indicate more anomalous
        """
        if self.model is None:
            raise ValueError("Model not trained, call train() first")
        
        # Get raw decision scores (negative = anomaly)
        raw_scores = self.model.decision_function(X)
        
        # Negate so higher values = more anomalous
        return -raw_scores
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability-like scores (0 to 1) from model.
        
        Normalizes the anomaly scores to [0, 1] range where 1 indicates anomaly.
        This doesn't output true probabilities but normalized scores that can
        be treated as anomaly likelihood.
        
        Args:
            X: Input features
            
        Returns:
            Normalized scores where 1 indicates anomaly, 0 indicates normal
        """
        if self.model is None:
            raise ValueError("Model not trained, call train() first")
        
        # Get anomaly scores
        scores = self.predict_score(X)
        
        # MinMax scale to [0, 1]
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        # Avoid division by zero
        if max_score == min_score:
            normalized = np.ones_like(scores) * 0.5
        else:
            normalized = (scores - min_score) / (max_score - min_score)
        
        return normalized
    
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
            # Isolation Forest returns -1 for anomalies and 1 for normal
            # Convert to 0 for normal, 1 for anomaly
            raw_preds = self.model.predict(X)
            return (raw_preds == -1).astype(int)
        else:
            # Use optimized threshold
            scores = self.predict_proba(X)
            return (scores >= self.optimal_threshold).astype(int)
    
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
        
        # Get scores
        scores = self.predict_proba(X_val)
        
        # Try different thresholds
        thresholds = np.linspace(0.01, 0.99, 99)
        best_score = -1
        best_threshold = 0.5
        
        for threshold in thresholds:
            preds = (scores >= threshold).astype(int)
            
            if metric == "precision":
                score = precision_score(y_val, preds, zero_division=0)
            elif metric == "recall":
                score = recall_score(y_val, preds)
            else:  # default to f1
                score = f1_score(y_val, preds, zero_division=0)
            
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
        scores = self.predict_proba(X)
        
        # Get binary predictions
        preds = self.predict(X)
        
        # Classification metrics
        auc_roc = roc_auc_score(y, scores)
        auprc = average_precision_score(y, scores)
        precision = precision_score(y, preds, zero_division=0)
        recall = recall_score(y, preds)
        f1 = f1_score(y, preds, zero_division=0)
        cm = confusion_matrix(y, preds)
        
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
    
    def get_feature_importance(self, importance_type: str = 'depth') -> pd.DataFrame:
        """
        Get feature importance.
        
        Note: Isolation Forest doesn't have a direct feature importance metric,
        but we can approximate it by looking at average path depths for each feature.
        
        Args:
            importance_type: Type of importance ('depth' only)
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model not trained or feature names not provided")
        
        logger.info("\nCalculating feature importance approximation...")
        
        # We'll compute average depth for each feature
        # Note: This is a rough approximation
        n_trees = len(self.model.estimators_)
        n_features = len(self.feature_names)
        
        # Initialize feature importance array
        feature_importances = np.zeros(n_features)
        
        # For each tree
        for tree in self.model.estimators_:
            # Extract the tree structure
            tree_structure = tree.tree_
            
            # For each node in the tree, add to feature importance
            for i in range(tree_structure.node_count):
                if tree_structure.feature[i] != -2:  # Not a leaf node
                    feature_importances[tree_structure.feature[i]] += 1
        
        # Normalize
        feature_importances = feature_importances / n_trees
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': feature_importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        logger.info(f"Top 10 features by importance:")
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
        model_path = os.path.join(model_dir, 'isolation_forest_model.joblib')
        joblib.dump(self.model, model_path)
        
        # Save model parameters
        params_path = os.path.join(model_dir, 'model_params.pkl')
        model_params = {
            'n_estimators': self.n_estimators,
            'max_samples': self.max_samples,
            'contamination': self.contamination,
            'max_features': self.max_features,
            'bootstrap': self.bootstrap,
            'optimal_threshold': self.optimal_threshold,
            'feature_names': self.feature_names
        }
        
        with open(params_path, 'wb') as f:
            pickle.dump(model_params, f)
        
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Model parameters saved to: {params_path}")
    
    @classmethod
    def load_model(cls, model_dir: str) -> 'IsolationForestAnomalyDetector':
        """
        Load a trained model.
        
        Args:
            model_dir: Directory with saved model
            
        Returns:
            Loaded IsolationForestAnomalyDetector instance
        """
        # Load model parameters
        params_path = os.path.join(model_dir, 'model_params.pkl')
        with open(params_path, 'rb') as f:
            model_params = pickle.load(f)
        
        # Create config
        config = {
            'model': {
                'n_estimators': model_params['n_estimators'],
                'max_samples': model_params['max_samples'],
                'contamination': model_params['contamination'],
                'max_features': model_params['max_features'],
                'bootstrap': model_params['bootstrap']
            }
        }
        
        # Create instance
        instance = cls(config)
        
        # Load model
        model_path = os.path.join(model_dir, 'isolation_forest_model.joblib')
        instance.model = joblib.load(model_path)
        
        # Set other attributes
        instance.optimal_threshold = model_params['optimal_threshold']
        instance.feature_names = model_params['feature_names']
        
        logger.info(f"Loaded model from: {model_path}")
        logger.info(f"Optimal threshold: {instance.optimal_threshold:.4f}")
        
        return instance
