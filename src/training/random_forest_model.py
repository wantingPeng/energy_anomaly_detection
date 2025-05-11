import os
import yaml
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from imblearn.over_sampling import BorderlineSMOTE
from sklearn.impute import SimpleImputer
import joblib
from itertools import product
from src.utils.logger import logger
from src.training.visualizations.random_forest import (
    plot_roc_curve,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_metrics_comparison,
    save_results,
    calculate_metrics_from_confusion_matrix
)
import numpy as np

class RandomForestTrainer:
    def __init__(self, config_path="configs/random_forest.yaml"):
        """Initialize the Random Forest trainer with configuration."""
        self.config = self._load_config(config_path)
        self.model = None
        self.best_model = None
        self.best_f1_score = 0
        self.best_params = None
        self.best_threshold = None
        self.imputer = SimpleImputer(strategy='mean')
        
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
        
        # Log class distribution before SMOTE
        train_dist = pd.Series(y_train).value_counts()
        val_dist = pd.Series(y_val).value_counts()
        logger.info("Class distribution before SMOTE:")
        logger.info(f"Training set - Normal: {train_dist[0]}, Anomaly: {train_dist[1]}")
        logger.info(f"Training set anomaly ratio: {train_dist[1]/(train_dist[0]+train_dist[1]):.4f}")
        logger.info(f"Validation set - Normal: {val_dist[0]}, Anomaly: {val_dist[1]}")
        logger.info(f"Validation set anomaly ratio: {val_dist[1]/(val_dist[0]+val_dist[1]):.4f}")
        
        # Handle missing values
        logger.info("Handling missing values with SimpleImputer...")
        X_train_imputed = self.imputer.fit_transform(X_train)
        X_val_imputed = self.imputer.transform(X_val)
        
        # Convert back to DataFrame to preserve column names
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=X_train.columns)
        X_val_imputed = pd.DataFrame(X_val_imputed, columns=X_val.columns)
        
        # Log missing value statistics
        logger.info(f"Number of NaN values in training set before imputation: {X_train.isna().sum().sum()}")
        logger.info(f"Number of NaN values in validation set before imputation: {X_val.isna().sum().sum()}")
        logger.info(f"Number of NaN values in training set after imputation: {X_train_imputed.isna().sum().sum()}")
        logger.info(f"Number of NaN values in validation set after imputation: {X_val_imputed.isna().sum().sum()}")
        
        # Apply BorderlineSMOTE to training data only
        logger.info("Applying BorderlineSMOTE for oversampling...")
        borderline_smote = BorderlineSMOTE(
            sampling_strategy=self.config['data']['smote_sampling_strategy'],
            random_state=self.config['model']['random_state'],
            k_neighbors=self.config['data']['smote_k_neighbors'],
            m_neighbors=self.config['data'].get('smote_m_neighbors', 10),
            kind=self.config['data'].get('smote_kind', 'borderline-1')
        )
        X_train_resampled, y_train_resampled = borderline_smote.fit_resample(X_train_imputed, y_train)
        
        # Log class distribution after SMOTE
        resampled_dist = pd.Series(y_train_resampled).value_counts()
        logger.info("Class distribution after BorderlineSMOTE:")
        logger.info(f"Training set - Normal: {resampled_dist[0]}, Anomaly: {resampled_dist[1]}")
        logger.info(f"Training set anomaly ratio: {resampled_dist[1]/(resampled_dist[0]+resampled_dist[1]):.4f}")
        logger.info(f"Number of synthetic samples generated: {resampled_dist[1] - train_dist[1]}")
        
        logger.info(f"Training data shape: {X_train_resampled.shape}")
        logger.info(f"Validation data shape: {X_val_imputed.shape}")
        
        return X_train_resampled, y_train_resampled, X_val_imputed, y_val
    
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
    
    def _find_optimal_threshold(self, y_true, y_pred_proba):
        """Find optimal threshold based on Precision-Recall curve."""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        # 计算F1分数
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # 找到最佳阈值（F1分数最高的点）
        best_threshold_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_threshold_idx]
        
        # 记录最佳阈值下的指标
        best_metrics = {
            'threshold': best_threshold,
            'precision': precision[best_threshold_idx],
            'recall': recall[best_threshold_idx],
            'f1_score': f1_scores[best_threshold_idx]
        }
        
        return best_threshold, best_metrics

    def _evaluate_model(self, model, X_val, y_val):
        """Evaluate model performance on validation set."""
        # 获取预测概率
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # 找到最佳阈值
        best_threshold, threshold_metrics = self._find_optimal_threshold(y_val, y_pred_proba)
        
        # 使用最佳阈值进行预测
        y_pred = (y_pred_proba >= best_threshold).astype(int)
        
        # 计算混淆矩阵和指标
        cm = confusion_matrix(y_val, y_pred)
        metrics = calculate_metrics_from_confusion_matrix(cm)
        
        # 添加阈值信息到指标中
        metrics['threshold'] = best_threshold
        metrics['threshold_metrics'] = threshold_metrics
        
        return metrics, y_pred, y_pred_proba
    
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

    def train_and_evaluate(self):
        """Train and evaluate Random Forest model with different hyperparameters."""
        X_train, y_train, X_val, y_val = self._load_data()
        feature_names = X_train.columns
        
        # Generate all parameter combinations
        param_grid = self.config['hyperparameters']
        param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
        
        results = []
        output_dir = os.path.dirname(self.config['output']['validation_report_path'])
        
        for i, params in enumerate(param_combinations, 1):
            logger.info(f"Training model with parameters: {params}")
            model = self._create_model(params)
            model.fit(X_train, y_train)
            
            # 评估模型并获取预测结果
            metrics, y_pred, y_pred_proba = self._evaluate_model(model, X_val, y_val)
            
            results.append({
                'parameters': params,
                'y_true': y_val,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'metrics': metrics
            })
            
            # 生成可视化
            plot_confusion_matrix(
                y_val, 
                y_pred, 
                os.path.join(output_dir, f'confusion_matrix_{i}.png')
            )
            
            # 绘制PR曲线并标注最佳阈值点
            plot_precision_recall_curve(
                y_val, 
                y_pred_proba,
                metrics['threshold'],
                os.path.join(output_dir, f'pr_curve_{i}.png')
            )
            
            # 更新最佳模型
            if metrics['F1 Score'] > self.best_f1_score:
                self.best_f1_score = metrics['F1 Score']
                self.best_model = model
                self.best_params = params
                self.best_threshold = metrics['threshold']
                logger.info(f"New best model found! F1 Score: {metrics['F1 Score']:.4f}")
                logger.info(f"Optimal threshold: {metrics['threshold']:.4f}")
        
        # Generate metrics comparison plot
        plot_metrics_comparison(
            results,
            os.path.join(output_dir, 'metrics_comparison.png')
        )
        
        # Save results
        save_results(results, self.config['output']['validation_report_path'])
        
        # Save best model
        os.makedirs(os.path.dirname(self.config['output']['model_path']), exist_ok=True)
        joblib.dump(self.best_model, self.config['output']['model_path'])
        logger.info(f"Best model saved to {self.config['output']['model_path']}")
        
        # Save feature importance
        self._save_feature_importance(self.best_model, feature_names)
        
        # Plot ROC curve for best model
        plot_roc_curve(
            self.best_model, 
            X_val, 
            y_val, 
            os.path.join(output_dir, 'roc_curve.png')
        )
        
        return self.best_model, self.best_params

if __name__ == "__main__":
    trainer = RandomForestTrainer()
    best_model, best_params = trainer.train_and_evaluate()
