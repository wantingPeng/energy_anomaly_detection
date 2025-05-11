import os
import yaml
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
from itertools import product
import matplotlib.pyplot as plt
from src.utils.logger import logger
from pathlib import Path

class RandomForestTrainer:
    def __init__(self, config_path="configs/random_forest.yaml"):
        """Initialize the Random Forest trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.best_model = None
        self.best_f1 = 0
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
        
        # Separate features and target
        target_col = self.config['data']['target_column']
        X_train = train_data.drop(columns=[target_col])
        y_train = train_data[target_col]
        X_val = val_data.drop(columns=[target_col])
        y_val = val_data[target_col]
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Validation data shape: {X_val.shape}")
        
        return X_train, y_train, X_val, y_val
    
    def _create_model(self, params):
        """Create a Random Forest model with given parameters."""
        model_params = {
            'n_estimators': params['n_estimators'],
            'max_depth': params['max_depth'],
            'min_samples_split': params['min_samples_split'],
            'class_weight': params['class_weight'],
            'random_state': self.config['model']['random_state'],
            'n_jobs': -1  # Use all available cores
        }
        return RandomForestClassifier(**model_params)
    
    def _evaluate_model(self, model, X_val, y_val):
        """Evaluate model performance on validation set."""
        y_pred = model.predict(X_val)
        report = classification_report(y_val, y_pred, output_dict=True)
        return report
    
    def _save_feature_importance(self, model, feature_names):
        """Save feature importance to CSV file."""
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config['output']['feature_importance_path']), exist_ok=True)
        importance_df.to_csv(self.config['output']['feature_importance_path'], index=False)
        logger.info(f"Feature importance saved to {self.config['output']['feature_importance_path']}")
    
    def _plot_roc_curve(self, model, X_val, y_val, save_path):
        """Plot and save ROC curve."""
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        logger.info(f"ROC curve saved to {save_path}")
    
    def train_and_evaluate(self):
        """Train and evaluate Random Forest model with different hyperparameters."""
        X_train, y_train, X_val, y_val = self._load_data()
        feature_names = X_train.columns
        
        # Generate all parameter combinations
        param_grid = self.config['hyperparameters']
        print(self.config["hyperparameters"]["max_depth"])

        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        
        results = []
        
        for params in param_combinations:
            logger.info(f"Training model with parameters: {params}")
            model = self._create_model(params)
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            report = self._evaluate_model(model, X_val, y_val)
            f1_score = report['weighted avg']['f1-score']
            
            results.append({
                'parameters': params,
                'f1_score': f1_score,
                'report': report
            })
            
            # Update best model if current model is better
            if f1_score > self.best_f1:
                self.best_f1 = f1_score
                self.best_model = model
                self.best_params = params
                logger.info(f"New best model found! F1 Score: {f1_score:.4f}")
        
        # Save results
        self._save_results(results)
        
        # Save best model
        os.makedirs(os.path.dirname(self.config['output']['model_path']), exist_ok=True)
        joblib.dump(self.best_model, self.config['output']['model_path'])
        logger.info(f"Best model saved to {self.config['output']['model_path']}")
        
        # Save feature importance
        self._save_feature_importance(self.best_model, feature_names)
        
        # Plot ROC curve for best model
        self._plot_roc_curve(
            self.best_model, 
            X_val, 
            y_val, 
            os.path.join(os.path.dirname(self.config['output']['model_path']), 'roc_curve.png')
        )
        
        return self.best_model, self.best_params
    
    def _save_results(self, results):
        """Save validation results to markdown file."""
        # Sort results by F1 score
        results.sort(key=lambda x: x['f1_score'], reverse=True)
        
        # Create markdown content
        md_content = "# Random Forest Model Validation Results\n\n"
        md_content += "## Performance Summary\n\n"
        md_content += "| Rank | F1 Score | Parameters |\n"
        md_content += "|------|----------|------------|\n"
        
        for i, result in enumerate(results, 1):
            params_str = ", ".join(f"{k}={v}" for k, v in result['parameters'].items())
            md_content += f"| {i} | {result['f1_score']:.4f} | {params_str} |\n"
        
        md_content += "\n## Detailed Classification Reports\n\n"
        
        for i, result in enumerate(results, 1):
            md_content += f"### Model {i} (F1 Score: {result['f1_score']:.4f})\n\n"
            md_content += f"Parameters: {result['parameters']}\n\n"
            md_content += "```\n"
            md_content += classification_report(
                y_true=None,  # We don't have the actual predictions here
                y_pred=None,
                target_names=['Normal', 'Anomaly'],
                digits=4,
                output_dict=False
            )
            md_content += "\n```\n\n"
        
        # Save to file
        os.makedirs(os.path.dirname(self.config['output']['validation_report_path']), exist_ok=True)
        with open(self.config['output']['validation_report_path'], 'w') as f:
            f.write(md_content)
        logger.info(f"Validation results saved to {self.config['output']['validation_report_path']}")

if __name__ == "__main__":
    trainer = RandomForestTrainer()
    best_model, best_params = trainer.train_and_evaluate()
