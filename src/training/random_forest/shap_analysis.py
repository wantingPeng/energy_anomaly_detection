"""
SHAP Analysis for Random Forest Anomaly Detection.

This script performs SHAP (SHapley Additive exPlanations) analysis on a trained
Random Forest model to explain model predictions and feature importance.

Features:
- Load trained Random Forest model
- Generate SHAP values
- Create summary plots (bar plot, dot plot, beeswarm plot)
- Create dependence plots for top features
- Create waterfall plots for individual predictions
- Create force plots for sample explanations
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# SHAP library
import shap

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.training.random_forest.random_forest_model import RandomForestAnomalyDetector
from src.training.random_forest.dataloader import create_random_forest_data


class RandomForestSHAPAnalyzer:
    """
    SHAP analyzer for Random Forest anomaly detection.
    
    This class provides comprehensive SHAP analysis including:
    - Global feature importance (summary plots)
    - Feature interactions (dependence plots)
    - Individual predictions explanation (waterfall plots)
    """
    
    def __init__(self, model, X, feature_names, output_dir, y=None, max_display=20, random_state=42):
        """
        Initialize SHAP analyzer.
        
        Args:
            model: Trained Random Forest model (sklearn)
            X: Features for SHAP analysis (numpy array)
            feature_names: List of feature names
            output_dir: Directory to save SHAP analysis results
            y: Labels for stratified sampling (optional, numpy array)
            max_display: Maximum number of features to display in plots
            random_state: Random seed for reproducibility
        """
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.output_dir = output_dir
        self.max_display = max_display
        self.random_state = random_state
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize SHAP explainer
        logger.info("Initializing SHAP TreeExplainer...")
        logger.info(f"  Model type: {type(self.model)}")
        logger.info(f"  Data shape: {self.X.shape}")
        logger.info(f"  Number of features: {len(self.feature_names)}")
        if self.y is not None:
            logger.info(f"  Labels provided: Yes (for stratified sampling)")
            logger.info(f"  Anomaly ratio: {self.y.mean():.4f}")
        else:
            logger.info(f"  Labels provided: No (random sampling only)")
        logger.info(f"  Random state: {self.random_state}")
        
        # Use TreeExplainer for tree-based models (efficient)
        self.explainer = shap.TreeExplainer(self.model)
        
        # Compute SHAP values
        logger.info("Computing SHAP values...")
        self.shap_values = None
        self.base_value = None
        
    def compute_shap_values(self, sample_size=None, sampling_strategy='stratified'):
        """
        Compute SHAP values for the dataset.
        
        Args:
            sample_size: Number of samples to use (None for all samples)
            sampling_strategy: Sampling strategy to use
                - 'random': Simple random sampling
                - 'stratified': Stratified sampling to preserve class ratio (requires y labels)
                - 'balanced': Balanced sampling (equal anomaly and normal samples)
                - 'anomaly_focused': Over-sample anomalies (e.g., 50% anomaly, 50% normal)
        """
        # Sample data if requested
        if sample_size is not None and sample_size < len(self.X):
            logger.info(f"Sampling {sample_size} samples from {len(self.X)} for SHAP analysis")
            logger.info(f"Sampling strategy: {sampling_strategy}")
            
            # Set random seed for reproducibility
            np.random.seed(self.random_state)
            
            if sampling_strategy == 'random':
                # Simple random sampling
                indices = np.random.choice(len(self.X), sample_size, replace=False)
                logger.info(f"  Using random sampling")
                
            elif sampling_strategy == 'balanced' and self.y is not None:
                # Balanced sampling: equal number of anomalies and normal samples
                anomaly_indices = np.where(self.y == 1)[0]
                normal_indices = np.where(self.y == 0)[0]
                
                n_per_class = sample_size // 2
                
                # Sample from each class
                if len(anomaly_indices) >= n_per_class:
                    sampled_anomaly = np.random.choice(anomaly_indices, n_per_class, replace=False)
                else:
                    # If not enough anomalies, use all and oversample
                    sampled_anomaly = np.random.choice(anomaly_indices, n_per_class, replace=True)
                    logger.warning(f"  Not enough anomalies ({len(anomaly_indices)}), using oversampling")
                
                sampled_normal = np.random.choice(normal_indices, n_per_class, replace=False)
                
                indices = np.concatenate([sampled_anomaly, sampled_normal])
                np.random.shuffle(indices)
                
                logger.info(f"  Balanced sampling: {n_per_class} normal, {n_per_class} anomaly")
                logger.info(f"  Anomaly ratio: 0.5000 (forced balance)")
                  
            else:
                # Fallback to random sampling
                if self.y is None:
                    logger.warning(f"  Strategy '{sampling_strategy}' requires labels, falling back to random sampling")
                indices = np.random.choice(len(self.X), sample_size, replace=False)
                logger.info(f"  Using random sampling (fallback)")
            
            X_sample = self.X[indices]
            
        else:
            X_sample = self.X
            indices = np.arange(len(self.X))
            logger.info(f"Using all {len(self.X)} samples for SHAP analysis")
            if self.y is not None:
                n_anomaly = self.y.sum()
                logger.info(f"  Total: {len(self.y) - n_anomaly} normal, {n_anomaly} anomaly")
        
        # Compute SHAP values
        logger.info("Computing SHAP values (this may take some time)...")
        
        # For binary classification, TreeExplainer returns SHAP values for both classes
        # We'll use the SHAP values for the positive class (anomaly class, index 1)
        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle both single and multi-class outputs
        if isinstance(shap_values, list):
            # Multi-class output (list format) - use anomaly class (class 1)
            self.shap_values = shap_values[1]
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                self.base_value = self.explainer.expected_value[1]
            else:
                self.base_value = self.explainer.expected_value
            logger.info(f"Multi-class output detected (list format). Using class 1 (anomaly) SHAP values")
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            # 3D array format: (n_samples, n_features, n_classes)
            # For binary classification, shape is (n_samples, n_features, 2)
            self.shap_values = shap_values[:, :, 1]  # Use class 1 (anomaly)
            if isinstance(self.explainer.expected_value, np.ndarray):
                self.base_value = float(self.explainer.expected_value[1])
            elif isinstance(self.explainer.expected_value, (list, tuple)):
                self.base_value = float(self.explainer.expected_value[1])
            else:
                self.base_value = float(self.explainer.expected_value)
            logger.info(f"Multi-class output detected (3D array format). Using class 1 (anomaly) SHAP values")
        else:
            # Single output
            self.shap_values = shap_values
            if isinstance(self.explainer.expected_value, np.ndarray):
                self.base_value = float(self.explainer.expected_value)
            elif isinstance(self.explainer.expected_value, (list, tuple)):
                self.base_value = float(self.explainer.expected_value[0])
            else:
                self.base_value = float(self.explainer.expected_value)
            logger.info(f"Single output detected")
        
        # Store the sample used and indices
        self.X_sample = X_sample
        self.sample_indices = indices
        
        logger.info(f"SHAP values computed successfully")
        logger.info(f"  SHAP values shape: {self.shap_values.shape}")
        logger.info(f"  Base value: {self.base_value:.4f}")
        
        return self.shap_values
    
   
    def plot_summary_dot(self):
        """
        Create summary dot plot (beeswarm plot) showing feature importance and effects.
        
        This plot shows:
        - Feature importance (vertical position)
        - Feature values (color)
        - SHAP values (horizontal position)
        """
        logger.info("Creating SHAP summary dot plot (beeswarm)...")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            self.X_sample,
            feature_names=self.feature_names,
            max_display=self.max_display,
            show=False
        )
        plt.title("SHAP Feature Effects", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'shap_summary_dot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved SHAP summary dot plot to: {save_path}")

    
    def plot_waterfall_examples(self, n_examples=1, example_type='auto'):
        """
        Create waterfall plots for example predictions.
        
        Waterfall plots show how each feature contributes to pushing the model
        prediction from the base value to the final prediction for individual samples.
        
        Args:
            n_examples: Number of examples to plot
            example_type: Type of examples to select:
                - 'auto': Select diverse examples (high/low predictions)
                - 'anomaly': Select anomaly examples
                - 'normal': Select normal examples
                - 'random': Select random examples
        """
        logger.info(f"Creating SHAP waterfall plots for {n_examples} examples (type: {example_type})...")
        
        # Select examples based on type
        if example_type == 'auto':
            # Select examples with highest and lowest predictions
            predictions = self.shap_values.sum(axis=1) + self.base_value
            high_indices = np.argsort(predictions)[-n_examples//2:]
            low_indices = np.argsort(predictions)[:n_examples//2 + n_examples%2]
            selected_indices = np.concatenate([high_indices, low_indices])
        elif example_type == 'random':
            selected_indices = np.random.choice(len(self.X_sample), n_examples, replace=False)
        else:
            # For anomaly/normal, we'd need labels (not implemented here)
            logger.warning(f"Example type '{example_type}' not fully implemented. Using 'auto' instead.")
            predictions = self.shap_values.sum(axis=1) + self.base_value
            high_indices = np.argsort(predictions)[-n_examples//2:]
            low_indices = np.argsort(predictions)[:n_examples//2 + n_examples%2]
            selected_indices = np.concatenate([high_indices, low_indices])
        
        # Create waterfall plots for selected examples
        for i, idx in enumerate(selected_indices):
            plt.figure(figsize=(10, 6))
            
            # Create explanation object for this sample
            explanation = shap.Explanation(
                values=self.shap_values[idx],
                base_values=self.base_value,
                data=self.X_sample[idx],
                feature_names=self.feature_names
            )
            
            # Create waterfall plot
            shap.plots.waterfall(explanation, max_display=15, show=False)
            plt.title(f"SHAP Waterfall Plot", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            save_path = os.path.join(self.output_dir, f'shap_waterfall_sample_{idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"  Saved waterfall plot for sample {idx} to: {save_path}")
    
    
    def save_feature_importance(self):
        """
        Save feature importance based on SHAP values.
        """
        logger.info("Saving SHAP-based feature importance...")
        
        # Calculate mean absolute SHAP value for each feature
        feature_importance = np.abs(self.shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'shap_importance': feature_importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('shap_importance', ascending=False).reset_index(drop=True)
        
        # Save to CSV
        save_path = os.path.join(self.output_dir, 'shap_feature_importance.csv')
        importance_df.to_csv(save_path, index=False)
        
        logger.info(f"Saved SHAP feature importance to: {save_path}")
        logger.info(f"\nTop 10 features by SHAP importance:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {idx+1}. {row['feature']}: {row['shap_importance']:.6f}")
        
        return importance_df
    
    def save_shap_values(self):
        """
        Save computed SHAP values to disk for future use.
        """
        logger.info("Saving SHAP values...")
        
        # Save as numpy array
        shap_values_path = os.path.join(self.output_dir, 'shap_values.npy')
        np.save(shap_values_path, self.shap_values)
        
        # Save base value
        base_value_path = os.path.join(self.output_dir, 'shap_base_value.pkl')
        with open(base_value_path, 'wb') as f:
            pickle.dump(self.base_value, f)
        
        # Save feature names
        feature_names_path = os.path.join(self.output_dir, 'feature_names.pkl')
        with open(feature_names_path, 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        logger.info(f"Saved SHAP values to: {shap_values_path}")
        logger.info(f"Saved base value to: {base_value_path}")
        logger.info(f"Saved feature names to: {feature_names_path}")
    
    def generate_full_analysis(self, sample_size=None, sampling_strategy='stratified', 
                             n_waterfall_examples=1):
        """
        Generate complete SHAP analysis with all plots and reports.
        
        Args:
            sample_size: Number of samples to use for SHAP analysis (None for all)
            sampling_strategy: Strategy for sampling ('random', 'stratified', 'balanced', 'anomaly_focused')
            n_waterfall_examples: Number of examples for waterfall plots
        """
        logger.info("\n" + "=" * 60)
        logger.info("STARTING COMPREHENSIVE SHAP ANALYSIS")
        logger.info("=" * 60)
        
        # Step 1: Compute SHAP values
        self.compute_shap_values(sample_size=sample_size, sampling_strategy=sampling_strategy)
        
        # Step 2: Create summary plots
        logger.info("\n--- Creating Summary Plots ---")
        self.plot_summary_dot()
        
        # Step 4: Create waterfall plots
        logger.info("\n--- Creating Waterfall Plots ---")
        self.plot_waterfall_examples(n_examples=n_waterfall_examples, example_type='auto')
        
        # Step 6: Save feature importance
        logger.info("\n--- Saving Feature Importance ---")
        self.save_feature_importance()
        
        # Step 7: Save SHAP values
        logger.info("\n--- Saving SHAP Values ---")
        self.save_shap_values()
        
        logger.info("\n" + "=" * 60)
        logger.info("SHAP ANALYSIS COMPLETED")
        logger.info("=" * 60)
        logger.info(f"All results saved to: {self.output_dir}")


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
    Main SHAP analysis function.
    
    Args:
        args: Command line arguments
    """
    logger.info("\n" + "=" * 60)
    logger.info("RANDOM FOREST SHAP ANALYSIS")
    logger.info("=" * 60)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Config path: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Load model
    logger.info("\n--- Loading Model ---")
    model = RandomForestAnomalyDetector.load_model(args.model_path)
    logger.info(f"Model loaded successfully")
    logger.info(f"  Optimal threshold: {model.optimal_threshold}")
    logger.info(f"  Number of features: {len(model.feature_names)}")
    
    # Load data
    logger.info("\n--- Loading Data ---")
    data_path = config['data']['data_path']
    logger.info(f"Data path: {data_path}")
    
    data_loader, data_dict = create_random_forest_data(data_path, config)
    
    logger.info(f"Data loaded:")
    logger.info(f"  Train samples: {len(data_dict['y_train'])}")
    logger.info(f"  Validation samples: {len(data_dict['y_val'])}")
    logger.info(f"  Test samples: {len(data_dict['y_test'])}")
    logger.info(f"  Number of features: {len(data_dict['feature_names'])}")
    
    # Choose dataset for SHAP analysis
    if args.dataset == 'train':
        X_analysis = data_dict['X_train']
        y_analysis = data_dict['y_train']
        dataset_name = 'train'
    elif args.dataset == 'val':
        X_analysis = data_dict['X_val']
        y_analysis = data_dict['y_val']
        dataset_name = 'validation'
    elif args.dataset == 'test':
        X_analysis = data_dict['X_test']
        y_analysis = data_dict['y_test']
        dataset_name = 'test'
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = args.experiment_name or f"shap_analysis_{timestamp}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"\nOutput directory: {output_dir}")
    
    # Save analysis configuration
    analysis_config = {
        'model_path': args.model_path,
        'config_path': args.config,
        'dataset': args.dataset,
        'dataset_size': len(X_analysis),
        'anomaly_ratio': float(y_analysis.mean()),
        'n_anomalies': int(y_analysis.sum()),
        'sample_size': args.sample_size,
        'sampling_strategy': args.sampling_strategy,
        'random_state': args.random_state,
        'top_n_dependence': args.top_n_dependence,
        'n_waterfall_examples': args.n_waterfall_examples,
        'max_display': args.max_display,
        'timestamp': timestamp
    }
    
    config_save_path = os.path.join(output_dir, 'analysis_config.json')
    with open(config_save_path, 'w') as f:
        json.dump(analysis_config, f, indent=2)
    logger.info(f"Saved analysis configuration to: {config_save_path}")
    
    # Create SHAP analyzer
    logger.info("\n--- Initializing SHAP Analyzer ---")
    analyzer = RandomForestSHAPAnalyzer(
        model=model.model,  # Get the sklearn model
        X=X_analysis,
        y=y_analysis,  # Pass labels for stratified sampling
        feature_names=data_dict['feature_names'],
        output_dir=output_dir,
        max_display=args.max_display,
        random_state=args.random_state
    )
    
    # Run comprehensive analysis
    analyzer.generate_full_analysis(
        sample_size=args.sample_size,
        sampling_strategy=args.sampling_strategy,
        n_waterfall_examples=args.n_waterfall_examples
    )
    
    # Save analysis summary
    logger.info("\n--- Saving Analysis Summary ---")
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RANDOM FOREST SHAP ANALYSIS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Analysis Date: {timestamp}\n")
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Config Path: {args.config}\n")
        f.write(f"Data Path: {data_path}\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Dataset Size: {len(X_analysis)}\n")
        f.write("  - feature_names.pkl\n\n")
        f.write("Output Directory:\n")
        f.write(f"  {output_dir}\n")
        f.write("\n" + "=" * 60 + "\n")
    
    logger.info(f"Saved analysis summary to: {summary_path}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SHAP analysis for Random Forest anomaly detection"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/random_forest_tuning_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="test",
        choices=['train', 'val', 'test'],
        help="Dataset to use for SHAP analysis (default: test)"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples to use for SHAP analysis (None for all samples)"
    )
    parser.add_argument(
        "--top_n_dependence",
        type=int,
        default=5,
        help="Number of top features for dependence plots"
    )
    parser.add_argument(
        "--n_waterfall_examples",
        type=int,
        default=5,
        help="Number of examples for waterfall plots"
    )
    parser.add_argument(
        "--max_display",
        type=int,
        default=20,
        help="Maximum number of features to display in summary plots"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/random_forest/shap_analysis",
        help="Base output directory for SHAP analysis results"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Name of the analysis experiment (optional, will use timestamp if not provided)"
    )
    parser.add_argument(
        "--sampling_strategy",
        type=str,
        default="stratified",
        choices=['random', 'stratified', 'balanced', 'anomaly_focused'],
        help="Sampling strategy: random, stratified (preserve ratio), balanced (50-50), anomaly_focused (over-sample anomalies)"
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    main(args)