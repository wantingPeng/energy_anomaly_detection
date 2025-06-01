import os
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    auc,
    f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.utils.logger import logger

class XGBoostAnomalyDetector:
    def __init__(self, config_path="configs/xgboost.yaml"):
        self.config = self._load_config(config_path)
        self.model = None
        self.feature_columns = None
        self.scale_pos_weight = None

    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _prepare_data(self, data_path):
        logger.info(f"Loading data from {data_path}")
        df = pd.read_parquet(data_path)

        if self.feature_columns is None:
            self.feature_columns = [col for col in df.columns
                                    if col not in self.config['features']['exclude_columns']]

        X = df[self.feature_columns]
        y = df[self.config['features']['target_column']]
        return X, y

    def _calculate_scale_pos_weight(self, y_train):
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        if n_pos == 0:
            raise ValueError("No positive samples in training data.")
        self.scale_pos_weight = n_neg / n_pos
        logger.info(f"Calculated scale_pos_weight: {self.scale_pos_weight:.2f}")
        return self.scale_pos_weight

    def train(self):
        logger.info("Starting XGBoost model training")

        X_train, y_train = self._prepare_data(self.config['data']['train_path'])
        X_val, y_val = self._prepare_data(self.config['data']['val_path'])

        self._calculate_scale_pos_weight(y_train)

        model_params = self.config['model'].copy()
        model_params['scale_pos_weight'] = self.scale_pos_weight
        early_stopping_rounds = model_params.pop('early_stopping_rounds', 20)
        
        # Create DMatrix for training and validation
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Set up parameters for training
        params = {
            **model_params,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        
        # Train model with early stopping
        logger.info("Training model with early stopping")
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=model_params.get('n_estimators', 200),
            evals=[(dval, 'validation')],
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=True
        )
        
        # Save model
        self._save_model()
        
        # Evaluate model
        self.evaluate(X_val, y_val)

    def _save_model(self):
        save_path = self.config['data']['model_save_path']
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        self.model.save_model(save_path)
        logger.info(f"Model saved to {save_path}")

    def find_best_threshold(self, y_true, y_scores):
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        best_index = f1_scores.argmax()
        best_threshold = thresholds[best_index]
        best_f1 = f1_scores[best_index]
        logger.info(f"Best F1 Score: {best_f1:.2f} at threshold: {best_threshold:.2f}")
        return best_threshold

    def evaluate(self, X_val, y_val):
        logger.info("Evaluating model performance")

        y_pred_proba = self.model.predict(xgb.DMatrix(X_val))
        best_threshold = self.find_best_threshold(y_val, y_pred_proba)
        y_pred = (y_pred_proba >= best_threshold).astype(int)

        report = classification_report(y_val, y_pred)
        logger.info("\n" + report)

        report_dir = Path(self.config['data']['report_dir'])
        report_dir.mkdir(parents=True, exist_ok=True)

        with open(report_dir / "xgboost_eval_report.md", "w") as f:
            f.write("# XGBoost Model Evaluation Report\n\n")
            f.write("## Classification Report\n\n")
            f.write("```")
            f.write(report)
            f.write("\n```\n")
            f.write(f"Scale Pos Weight: {self.scale_pos_weight:.2f}\n")

        self._plot_confusion_matrix(y_val, y_pred, report_dir)
        self._plot_pr_curve(y_val, y_pred_proba, report_dir)
        self._plot_roc_curve(y_val, y_pred_proba, report_dir)

        logger.info(f"Evaluation results saved to {report_dir}")

    def _plot_confusion_matrix(self, y_true, y_pred, out_dir):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(out_dir / "xgboost_confusion_matrix.png")
        plt.close()

    def _plot_pr_curve(self, y_true, y_scores, out_dir):
        plt.figure(figsize=(8, 6))
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        plt.plot(recall, precision)
        plt.title('Precision-Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
        plt.savefig(out_dir / "xgboost_pr_curve.png")
        plt.close()

    def _plot_roc_curve(self, y_true, y_scores, out_dir):
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(out_dir / "xgboost_roc_curve.png")
        plt.close()


def main():
    try:
        detector = XGBoostAnomalyDetector()
        detector.train()
    except Exception as e:
        logger.error(f"Error in XGBoost training pipeline: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
