#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import xgboost as xgb
import json
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import time
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, roc_curve, auc, confusion_matrix,
    classification_report
)
from sklearn.model_selection import RandomizedSearchCV
from src.utils.logger import logger
import gc


class XGBoostAnomalyDetector:
    """
    XGBoost-based anomaly detection model with dynamic threshold optimization
    and automatic hyperparameter tuning.
    """
    
    def __init__(self, config):
        """
        Initialize the model with configuration.
        
        Args:
            config (dict): Model configuration
        """
        self.config = config
        self.model = None
        self.best_params = None
        self.optimal_threshold = 0.5  # Default threshold
        self.feature_importances = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_save_dir = Path(self.config['data_config']['save_dir']) / f"model_{self.timestamp}"
        self.feature_names = None
        
        # Create save directory
        os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Log initialization
        logger.info(f"XGBoost model initialized with timestamp: {self.timestamp}")
        
    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """
        Find optimal hyperparameters using RandomizedSearchCV.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation labels
            
        Returns:
            dict: Best hyperparameters
        """
        logger.info("Starting hyperparameter tuning...")
        
        # Use a subsample of training data for hyperparameter tuning to reduce memory usage
        if len(X_train) > 50000:
            logger.info(f"Using subsample of training data for hyperparameter tuning (50000/{len(X_train)} samples)")
            # Stratified sampling to maintain class distribution
            from sklearn.model_selection import train_test_split
            X_tune, _, y_tune, _ = train_test_split(X_train, y_train, 
                                                    train_size=50000, 
                                                    stratify=y_train,
                                                    random_state=42)
        else:
            X_tune, y_tune = X_train, y_train
            
        # Extract tuning parameters from config
        param_grid = {}
        for param, values in self.config['model_config']['tuning_params'].items():
            param_grid[param] = values
            
        # Set fixed parameters - get num_boost_round for n_estimators
        fixed_params = self.config['model_config']['fixed_params'].copy()
        n_estimators = fixed_params.pop('num_boost_round', 500)
        
        # Make sure we don't have n_estimators in both fixed_params and as a direct parameter
        if 'n_estimators' in fixed_params:
            fixed_params.pop('n_estimators')
            
        logger.info(f"Using {n_estimators} as n_estimators for hyperparameter tuning")
        
        # Add memory optimization parameters
        tree_method = self.config['model_config']['tree_method'] 
        device = self.config['model_config']['device']
        
        logger.info(f"Using tree_method={tree_method}, device={device}")
        
        # Create base estimator with fixed parameters
        base_estimator = xgb.XGBClassifier(
            objective=self.config['model_config']['objective'],
            tree_method=tree_method,
            device=device,
            n_estimators=n_estimators,
            **fixed_params
        )
        
        n_iter = 10  # Reduce number of parameter combinations to try
        n_jobs = self.config['training_config']['n_jobs']
        
        logger.info(f"RandomizedSearchCV with n_iter={n_iter}, n_jobs={n_jobs}")
        
        # Set up randomized search
        search = RandomizedSearchCV(
            estimator=base_estimator,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='f1',  # Optimize for F1 score
            cv=self.config['training_config']['cv_folds'],
            verbose=2,
            random_state=fixed_params['random_state'],
            n_jobs=n_jobs
        )
        
        # Fit on training data
        search.fit(X_tune, y_tune)
        
        # Log best parameters
        logger.info(f"Best hyperparameters found: {search.best_params_}")
        logger.info(f"Best CV score: {search.best_score_}")
        
        # Save best parameters
        self.best_params = search.best_params_
        
        # Save hyperparameter tuning results
        results_df = pd.DataFrame(search.cv_results_)
        results_df.to_csv(self.model_save_dir / "hyperparameter_tuning_results.csv", index=False)
        
        return search.best_params_
    
    def train(self, X_train, y_train, X_val, y_val, tune_hyperparameters=True):
        """
        Train the XGBoost model with optional hyperparameter tuning.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation labels
            tune_hyperparameters (bool): Whether to perform hyperparameter tuning
            
        Returns:
            self: Trained model
        """
        start_time = time.time()
        logger.info("Starting model training...")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        # Get device - use GPU if available
        device = self.config['model_config']['device']
        logger.info(f"Using device: {device}")
        
        # Create DMatrix objects for efficient training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Free some memory
        gc.collect()
        
        # Extract core parameters (not including num_boost_round which is used separately)
        fixed_params = self.config['model_config']['fixed_params'].copy()
        num_boost_round = fixed_params.pop('num_boost_round', 500)
        
        # Get parameters
        params = {
            'objective': self.config['model_config']['objective'],
            'eval_metric': self.config['model_config']['eval_metric'],
            'tree_method': self.config['model_config']['tree_method'],
            'device': device,
            **fixed_params
        }
        
        # Perform hyperparameter tuning if requested
        if tune_hyperparameters:
            best_params = self.hyperparameter_tuning(X_train, y_train, X_val, y_val)
            params.update(best_params)
        
        # Train the model
        logger.info(f"Training with parameters: {params}")
        logger.info(f"Training for up to {num_boost_round} rounds with early stopping...")
        
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=self.config['training_config']['early_stopping_rounds'],
            verbose_eval=self.config['training_config']['verbose']
        )
        
        logger.info(f"Model trained for {self.model.best_iteration + 1} rounds")
        logger.info(f"Best score: {self.model.best_score}")
        
        # Get feature importances
        self.feature_importances = self.model.get_score(importance_type=self.config['feature_config']['importance_type'])
        
        # Plot feature importance
        self.plot_feature_importance(top_n=self.config['feature_config'].get('top_n', 20))
        
        # Optimize threshold
        self.optimize_threshold(X_val, y_val)
        
        # Save model
        self.save_model()
        
        # Log training time
        training_time = time.time() - start_time
        logger.info(f"Model training completed in {training_time:.2f} seconds")
        
        return self
    
    def optimize_threshold(self, X_val, y_val):
        """
        Find optimal decision threshold using validation data.
        
        Args:
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation labels
        """
        logger.info("Optimizing decision threshold...")
        
        # Get predictions
        dval = xgb.DMatrix(X_val)
        y_pred_proba = self.model.predict(dval)
        
        # Determine metric to optimize
        optimize_metric = self.config['threshold_config']['optimize_for']
        logger.info(f"Optimizing threshold for {optimize_metric}")
        
        # Try different thresholds
        thresholds = np.linspace(0.1, 0.9, 81)
        best_score = 0
        best_threshold = 0.5
        
        results = []
        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            })
            
            # Update best threshold
            current_score = eval(f"{optimize_metric}")
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        logger.info(f"Optimal threshold found: {self.optimal_threshold} with {optimize_metric} = {best_score}")
        
        # Save threshold results
        threshold_df = pd.DataFrame(results)
        threshold_df.to_csv(self.model_save_dir / "threshold_optimization.csv", index=False)
        
        # Plot threshold vs metrics
        plt.figure(figsize=(10, 6))
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            plt.plot(threshold_df['threshold'], threshold_df[metric], label=metric)
        plt.axvline(x=self.optimal_threshold, color='r', linestyle='--', label=f'Optimal ({self.optimal_threshold:.2f})')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs. Threshold')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.model_save_dir / "threshold_optimization.png")
        
    def predict(self, X):
        """
        Predict anomalies with the trained model using the optimal threshold.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.array: Binary predictions (0: normal, 1: anomaly)
        """
        # Create DMatrix for prediction to ensure device compatibility
        dtest = xgb.DMatrix(X)
        # Get prediction probabilities
        y_pred_proba = self.model.predict(dtest)
        # Apply threshold
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        return y_pred
    
    def predict_proba(self, X):
        """
        Get probability predictions.
        
        Args:
            X (pd.DataFrame): Features
            
        Returns:
            np.array: Anomaly probabilities
        """
        # Create DMatrix for prediction to ensure device compatibility
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X, y):
        """
        Evaluate model performance on test data.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): True labels
            
        Returns:
            dict: Evaluation metrics
        """
        logger.info("Evaluating model performance...")
        
        # Get predictions
        y_pred_proba = self.predict_proba(X)
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        # Calculate AUC
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Generate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Create classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Log results
        logger.info(f"Evaluation results:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
        logger.info(f"  Confusion Matrix:\n{cm}")
        
        # Save evaluation results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'optimal_threshold': self.optimal_threshold
        }
        
        # Save metrics
        with open(self.model_save_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(results, f, indent=4)
        
        # Generate and save plots
        self._generate_eval_plots(y, y_pred_proba)
        
        return results
    
    def _generate_eval_plots(self, y_true, y_pred_proba):
        """
        Generate evaluation plots.
        
        Args:
            y_true (np.array): True labels
            y_pred_proba (np.array): Predicted probabilities
        """
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig(self.model_save_dir / "roc_curve.png")
        
        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'AUC = {pr_auc:.4f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.grid(True)
        plt.savefig(self.model_save_dir / "pr_curve.png")
        
        # Feature Importance Plot
        if self.feature_importances is not None and self.feature_names is not None:
            # Create sorted dataframe of feature importances
            importance_df = pd.DataFrame({
                'Feature': [self.feature_names[i] if i < len(self.feature_names) else f"f{i}" 
                           for i in range(len(self.feature_importances))],
                'Importance': list(self.feature_importances.values())
            })
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Plot top features
            top_n = min(self.config['feature_config']['top_features_to_plot'], len(importance_df))
            plt.figure(figsize=(12, 8))
            plt.barh(importance_df['Feature'][:top_n][::-1], importance_df['Importance'][:top_n][::-1])
            plt.xlabel('Importance')
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()
            plt.savefig(self.model_save_dir / "feature_importance.png")
            
            # Save feature importance data
            importance_df.to_csv(self.model_save_dir / "feature_importance.csv", index=False)
    
    def save_model(self):
        """
        Save the trained model and related artifacts.
        """
        logger.info(f"Saving model to {self.model_save_dir}")
        
        # Save model
        model_path = self.model_save_dir / "xgboost_model.json"
        self.model.save_model(str(model_path))
        
        # Save configuration
        config_path = self.model_save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        # Save optimal threshold
        threshold_path = self.model_save_dir / "optimal_threshold.json"
        with open(threshold_path, 'w') as f:
            json.dump({'optimal_threshold': self.optimal_threshold}, f, indent=4)
        
        # Save feature names
        if self.feature_names:
            with open(self.model_save_dir / "feature_names.json", 'w') as f:
                json.dump({'feature_names': self.feature_names}, f, indent=4)
    
    @classmethod
    def load_model(cls, model_dir):
        """
        Load a saved model.
        
        Args:
            model_dir (str): Path to model directory
            
        Returns:
            XGBoostAnomalyDetector: Loaded model
        """
        logger.info(f"Loading model from {model_dir}")
        model_dir = Path(model_dir)
        
        # Load configuration
        with open(model_dir / "config.json", 'r') as f:
            config = json.load(f)
        
        # Create instance
        instance = cls(config)
        
        # Load model
        instance.model = xgb.Booster()
        instance.model.load_model(str(model_dir / "xgboost_model.json"))
        
        # Load optimal threshold
        with open(model_dir / "optimal_threshold.json", 'r') as f:
            threshold_data = json.load(f)
            instance.optimal_threshold = threshold_data['optimal_threshold']
        
        # Load feature names if available
        try:
            with open(model_dir / "feature_names.json", 'r') as f:
                feature_data = json.load(f)
                instance.feature_names = feature_data['feature_names']
        except FileNotFoundError:
            logger.warning("Feature names file not found.")
        
        return instance

    def plot_feature_importance(self, top_n=20):
        """
        Plot and save feature importance.
        
        Args:
            top_n (int): Number of top features to display
            
        Returns:
            pd.DataFrame: DataFrame with all feature importances sorted
        """
        logger.info(f"Plotting feature importance (top {top_n} features)...")
        
        if self.feature_importances is None or not self.feature_names:
            logger.warning("Feature importances not available. Model might not be trained yet.")
            return None
            
        # Convert to DataFrame for easier manipulation
        importance_type = self.config['feature_config']['importance_type']
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': [self.feature_importances.get(f, 0) for f in self.feature_names]
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Save full feature importance to CSV
        importance_file = self.model_save_dir / f"feature_importance_{importance_type}.csv"
        importance_df.to_csv(importance_file, index=False)
        logger.info(f"Full feature importance saved to {importance_file}")
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1])
        plt.xlabel(f'Importance ({importance_type})')
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        # Save plot
        plot_file = self.model_save_dir / f"feature_importance_{importance_type}.png"
        plt.savefig(plot_file, dpi=300)
        plt.close()
        logger.info(f"Feature importance plot saved to {plot_file}")
        
        return importance_df
