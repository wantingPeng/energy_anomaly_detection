import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
import shap
import warnings
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.logger import logger

def load_data(file_path):
    """Load data from parquet file."""
    logger.info(f"Loading data from {file_path}")
    try:
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df):
    """Preprocess the data for feature importance analysis."""
    logger.info("Preprocessing data")
    
    # Drop non-feature columns if present (adjust as needed)
    features = df.drop(['TimeStamp', 'anomaly_label'], axis=1, errors='ignore')
    
    # Handle missing values if any
    if features.isnull().sum().sum() > 0:
        logger.info(f"Handling missing values. Missing counts: {features.isnull().sum().sum()}")
        features = features.fillna(features.mean())
    
    # Get the target variable if present
    target = df['anomaly_label'] if 'anomaly_label' in df.columns else None
    
    return features, target

def analyze_correlation(features, target=None, output_dir='experiments/feature_analysis'):
    """Analyze feature correlations and plot heatmap."""
    logger.info("Analyzing feature correlations")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate correlation matrix
    corr_matrix = features.corr()
    
    # Plot correlation heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                center=0, linewidths=.5, fmt='.2f', square=True)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find highly correlated features
    high_corr = {}
    for i, col1 in enumerate(corr_matrix.columns):
        for col2 in corr_matrix.columns[i+1:]:
            if abs(corr_matrix.loc[col1, col2]) > 0.8:
                high_corr[(col1, col2)] = corr_matrix.loc[col1, col2]
    
    # Sort by correlation strength
    high_corr = {k: v for k, v in sorted(high_corr.items(), key=lambda item: abs(item[1]), reverse=True)}
    
    # Output high correlation pairs
    if high_corr:
        logger.info("High correlation pairs (|r| > 0.8):")
        with open(f"{output_dir}/high_correlations.txt", 'w') as f:
            for (col1, col2), corr in high_corr.items():
                line = f"{col1} - {col2}: {corr:.4f}"
                logger.info(line)
                f.write(line + '\n')
    else:
        logger.info("No high correlation pairs found.")
    
    # If target is provided, analyze correlation with target
    if target is not None:
        feature_target_corr = pd.DataFrame()
        feature_target_corr['feature'] = features.columns
        feature_target_corr['correlation'] = [features[col].corr(target) for col in features.columns]
        feature_target_corr['abs_correlation'] = feature_target_corr['correlation'].abs()
        feature_target_corr = feature_target_corr.sort_values('abs_correlation', ascending=False)
        
        logger.info("Top 10 features by correlation with target:")
        with open(f"{output_dir}/target_correlations.txt", 'w') as f:
            for i, row in feature_target_corr.head(10).iterrows():
                line = f"{row['feature']}: {row['correlation']:.4f}"
                logger.info(line)
                f.write(line + '\n')
        
        # Plot top features by correlation with target
        plt.figure(figsize=(12, 8))
        sns.barplot(x='correlation', y='feature', data=feature_target_corr.head(15))
        plt.title('Top 15 Features by Correlation with Target')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/target_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    return corr_matrix

def analyze_mutual_information(features, target, output_dir='experiments/feature_analysis'):
    """Analyze feature importance using mutual information."""
    if target is None:
        logger.warning("Target variable not provided, skipping mutual information analysis")
        return None
    
    logger.info("Analyzing mutual information")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate mutual information
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        mi_scores = mutual_info_classif(features, target)
    
    # Create DataFrame with results
    mi_df = pd.DataFrame()
    mi_df['feature'] = features.columns
    mi_df['mutual_info'] = mi_scores
    mi_df = mi_df.sort_values('mutual_info', ascending=False)
    
    logger.info("Top 10 features by mutual information:")
    with open(f"{output_dir}/mutual_information.txt", 'w') as f:
        for i, row in mi_df.head(10).iterrows():
            line = f"{row['feature']}: {row['mutual_info']:.4f}"
            logger.info(line)
            f.write(line + '\n')
    
    # Plot mutual information
    plt.figure(figsize=(12, 8))
    sns.barplot(x='mutual_info', y='feature', data=mi_df.head(15))
    plt.title('Top 15 Features by Mutual Information with Target')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/mutual_information.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return mi_df

def analyze_tree_based_importance(features, target, output_dir='experiments/feature_analysis'):
    """Analyze feature importance using tree-based models."""
    if target is None:
        logger.warning("Target variable not provided, skipping tree-based importance analysis")
        return None
    
    logger.info("Analyzing tree-based feature importance")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Handle imbalanced classes if necessary
    class_counts = target.value_counts()
    if len(class_counts) > 1:
        class_weights = 'balanced'
        logger.info(f"Class distribution: {class_counts.to_dict()}")
    else:
        class_weights = None
        logger.warning("Only one class found in target variable")
    
    # Random Forest feature importance
    try:
        rf = RandomForestClassifier(n_estimators=100, class_weight=class_weights, random_state=42, n_jobs=-1)
        rf.fit(features, target)
        
        # Get feature importance
        rf_importance = pd.DataFrame()
        rf_importance['feature'] = features.columns
        rf_importance['importance'] = rf.feature_importances_
        rf_importance = rf_importance.sort_values('importance', ascending=False)
        
        logger.info("Top 10 features by Random Forest importance:")
        with open(f"{output_dir}/rf_importance.txt", 'w') as f:
            for i, row in rf_importance.head(10).iterrows():
                line = f"{row['feature']}: {row['importance']:.4f}"
                logger.info(line)
                f.write(line + '\n')
        
        # Plot Random Forest feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=rf_importance.head(15))
        plt.title('Top 15 Features by Random Forest Importance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/rf_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Permutation importance (more reliable than built-in feature_importances_)
        logger.info("Calculating permutation importance")
        perm_importance = permutation_importance(rf, features, target, n_repeats=10, random_state=42, n_jobs=-1)
        
        perm_imp_df = pd.DataFrame()
        perm_imp_df['feature'] = features.columns
        perm_imp_df['importance'] = perm_importance.importances_mean
        perm_imp_df = perm_imp_df.sort_values('importance', ascending=False)
        
        logger.info("Top 10 features by permutation importance:")
        with open(f"{output_dir}/permutation_importance.txt", 'w') as f:
            for i, row in perm_imp_df.head(10).iterrows():
                line = f"{row['feature']}: {row['importance']:.4f}"
                logger.info(line)
                f.write(line + '\n')
        
        # Plot permutation importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=perm_imp_df.head(15))
        plt.title('Top 15 Features by Permutation Importance')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/permutation_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        logger.error(f"Error in Random Forest analysis: {e}")
        rf_importance = None
    
    # Try SHAP values for a deeper understanding
    try:
        logger.info("Calculating SHAP values (this may take time)")
        # Create a smaller model for SHAP analysis
        sample_size = min(10000, len(features))
        indices = np.random.choice(len(features), sample_size, replace=False)
        X_sample = features.iloc[indices]
        y_sample = target.iloc[indices]
        
        # Train a model on the sample
        model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_sample, y_sample)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Plot SHAP summary
        plt.figure(figsize=(10, 12))
        if isinstance(shap_values, list):
            # For multi-class, use the first class
            shap.summary_plot(shap_values[0], X_sample, plot_type="bar", show=False)
        else:
            shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/shap_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("SHAP analysis completed")
    except Exception as e:
        logger.error(f"Error in SHAP analysis: {e}")
    
    return rf_importance

def analyze_pca(features, output_dir='experiments/feature_analysis'):
    """Analyze features using Principal Component Analysis."""
    logger.info("Analyzing features using PCA")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Apply PCA
    pca = PCA()
    pca.fit(scaled_features)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Plot explained variance
    plt.figure(figsize=(12, 6))
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
    plt.step(range(1, len(cumulative_variance) + 1), cumulative_variance, where='mid', label='Cumulative')
    plt.axhline(y=0.95, linestyle='--', color='r', label='95% Explained Variance')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_explained_variance.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Find number of components needed to explain 95% variance
    n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    logger.info(f"Number of components needed for 95% variance: {n_components_95}")
    
    # Get feature importance from first few principal components
    n_components = min(5, len(features.columns))
    pca = PCA(n_components=n_components)
    pca.fit(scaled_features)
    
    # For each principal component, get the feature loadings
    loadings = pd.DataFrame(
        data=pca.components_.T * np.sqrt(pca.explained_variance_), 
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=features.columns
    )
    
    # Get absolute loadings and mean importance across components
    abs_loadings = loadings.abs()
    mean_abs_loading = abs_loadings.mean(axis=1).sort_values(ascending=False)
    
    logger.info("Top 10 features by PCA loading importance:")
    with open(f"{output_dir}/pca_loadings.txt", 'w') as f:
        for feature, importance in mean_abs_loading.head(10).items():
            line = f"{feature}: {importance:.4f}"
            logger.info(line)
            f.write(line + '\n')
    
    # Plot top features by PCA loading
    plt.figure(figsize=(12, 8))
    mean_abs_loading.head(15).plot(kind='bar')
    plt.title('Top 15 Features by PCA Loading Magnitude')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_loadings.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot the loadings heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(loadings.iloc[:15], annot=True, cmap='coolwarm', center=0)
    plt.title('PCA Loadings for Top 15 Features')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_loadings_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return loadings, mean_abs_loading

def generate_summary_report(correlation_results=None, mi_results=None, rf_results=None, pca_results=None, 
                            output_dir='experiments/feature_analysis'):
    """Generate a summary report of feature importance from all methods."""
    logger.info("Generating summary report")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine results from all methods
    all_methods = {}
    
    if isinstance(rf_results, pd.DataFrame):
        all_methods['Random Forest'] = rf_results.set_index('feature')['importance']
    
    if isinstance(mi_results, pd.DataFrame):
        all_methods['Mutual Information'] = mi_results.set_index('feature')['mutual_info']
    
    if isinstance(pca_results, tuple) and len(pca_results) == 2:
        all_methods['PCA Loading'] = pca_results[1]
    
    if not all_methods:
        logger.warning("No results available for summary report")
        return
    
    # Create a summary dataframe
    summary = pd.DataFrame(all_methods)
    
    # Add a mean rank column
    for col in summary.columns:
        summary[f'{col}_rank'] = summary[col].rank(ascending=False)
    
    rank_cols = [col for col in summary.columns if col.endswith('_rank')]
    summary['mean_rank'] = summary[rank_cols].mean(axis=1)
    summary = summary.sort_values('mean_rank')
    
    # Save the summary to a CSV file
    summary.to_csv(f"{output_dir}/feature_importance_summary.csv")
    
    # Create a summary plot of the top features
    top_features = summary.head(15).index
    
    # Plot the importance of top features across methods
    summary_data = []
    for feature in top_features:
        for method in [col for col in summary.columns if not col.endswith('_rank') and col != 'mean_rank']:
            if feature in summary.index and method in summary.columns:
                summary_data.append({
                    'Feature': feature,
                    'Method': method,
                    'Importance': summary.loc[feature, method]
                })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Normalize importance scores within each method
    for method in summary_df['Method'].unique():
        max_val = summary_df[summary_df['Method'] == method]['Importance'].max()
        if max_val > 0:  # Avoid division by zero
            summary_df.loc[summary_df['Method'] == method, 'Normalized Importance'] = \
                summary_df.loc[summary_df['Method'] == method, 'Importance'] / max_val
    
    # Plot normalized importance
    plt.figure(figsize=(14, 10))
    g = sns.barplot(x='Feature', y='Normalized Importance', hue='Method', data=summary_df)
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Importance Across Methods (Normalized)')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance_summary.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Log the top 10 features by mean rank
    logger.info("Top 10 features by mean rank across methods:")
    with open(f"{output_dir}/top_features.txt", 'w') as f:
        for feature, row in summary.head(10).iterrows():
            line = f"{feature}: Mean Rank {row['mean_rank']:.2f}"
            logger.info(line)
            f.write(line + '\n')
    
    return summary

def main():
    # Set up logging
    import logging
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/feature_importance_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Analyze feature importance in a dataset')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the parquet file containing the dataset')
    parser.add_argument('--output_dir', type=str, default='experiments/feature_analysis',
                        help='Directory to save analysis results')
    parser.add_argument('--target_column', type=str, default='anomaly_label',
                        help='Name of the target column (if any)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    df = load_data(args.input_file)
    
    # Check if target column exists
    has_target = args.target_column in df.columns
    if not has_target:
        logger.warning(f"Target column '{args.target_column}' not found in the dataset. "
                      "Some analyses will be skipped.")
    
    # Preprocess data
    features, target = preprocess_data(df)
    
    # Run analyses
    corr_results = analyze_correlation(features, target, args.output_dir)
    
    # Only run these if target is available
    mi_results = None
    rf_results = None
    if has_target:
        mi_results = analyze_mutual_information(features, target, args.output_dir)
        rf_results = analyze_tree_based_importance(features, target, args.output_dir)
    
    # PCA analysis doesn't need the target
    pca_results = analyze_pca(features, args.output_dir)
    
    # Generate summary report
    summary = generate_summary_report(corr_results, mi_results, rf_results, pca_results, args.output_dir)
    
    logger.info(f"Feature importance analysis completed. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
