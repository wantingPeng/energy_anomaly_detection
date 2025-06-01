# src/preprocessing/energy/feature_engineering.py

import os
import yaml
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.logger import logger
from src.utils.correlation import analyze_correlations_method1
from dataclasses import dataclass
from functools import partial

@dataclass
class FeatureSelectionConfig:
    input_dir: str
    output_dir: str
    report_path: str
    correlation_threshold: float
    high_correlation_threshold: float
    multicollinearity_threshold: float

def load_config(config_path: str = "configs/feature_selection.yaml") -> FeatureSelectionConfig:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Successfully loaded config from {config_path}")
        return FeatureSelectionConfig(
            input_dir=config['data_paths']['input_dir'],
            output_dir=config['data_paths']['output_dir'],
            report_path=config['report']['output_path'],
            correlation_threshold=config['feature_selection']['correlation_threshold'],
            high_correlation_threshold=config['feature_selection']['high_correlation_threshold'],
            multicollinearity_threshold=config['feature_selection']['multicollinearity_threshold']
        )
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        raise

def determine_component_type(filename: str) -> Optional[str]:
    """Determine component type from filename"""
    if "ring" in filename.lower():
        return "ring"
    elif "pcb" in filename.lower():
        return "pcb"
    elif "kontaktieren" in filename.lower():
        return "contact"
    return None

def load_single_file(file_path: str) -> Optional[Tuple[str, pd.DataFrame]]:
    """Load a single parquet file and return component type and dataframe"""
    try:
        filename = os.path.basename(file_path)
        component_type = determine_component_type(filename)
        
        if component_type is None:
            logger.warning(f"Unknown component type for file: {filename}")
            return None
            
        df = pd.read_parquet(file_path)
        df['component_type'] = component_type
        logger.info(f"Loaded {component_type} data with shape {df.shape}")
        return component_type, df
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {str(e)}")
        return None

def load_data(input_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all parquet files from directory"""
    file_paths = glob.glob(os.path.join(input_dir, "*.parquet"))
    loaded_data = [load_single_file(fp) for fp in file_paths]
    return {comp_type: df for comp_type, df in loaded_data if df is not None}

def analyze_correlations(data_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
    """Analyze correlations for each component type"""
    correlations = {}
    for component, df in data_dict.items():
        correlations[component] = analyze_correlations_method1(df)
        logger.info(f"Analyzed correlations for {component}")
    return correlations

def select_features(correlations: Dict[str, pd.Series], 
                   data_dict: Dict[str, pd.DataFrame],
                   threshold: float, 
                   high_threshold: float) -> Tuple[pd.DataFrame, List[str]]:
    """Select features based on correlation thresholds and create intersection DataFrame"""
    report_content = []
    selected_by_component = {}
    
    # Select features for each component
    for component, corr in correlations.items():
        selected = set(corr[corr > threshold].index)
        selected_by_component[component] = selected
        
        important_features = set(corr[corr > high_threshold].index)
        report_content.extend([
            f"\nImportant features ({component}, correlation > {high_threshold}):",
            "\n".join(f"- {f}" for f in important_features)
        ])

    # Find common features
    common_features = set.intersection(*selected_by_component.values())
    
    # Add important features that weren't in the intersection
    all_important_features = set()
    for component, corr in correlations.items():
        important_features = set(corr[corr > high_threshold].index)
        all_important_features.update(important_features)
    
    # Update common features with all important features
    common_features.update(all_important_features)
    
    # Create intersection DataFrame with additional columns
    additional_columns = ['window_start', 'window_end', 'component_type', 'anomaly_label']
    all_columns = list(common_features) + additional_columns
    intersection_df = pd.DataFrame()
    
    # Process each component's data
    for component, df in data_dict.items():
        # Find missing columns for this component
        missing_cols = [col for col in common_features if col not in df.columns]
        
        # Create a copy of the dataframe
        component_df = df.copy()
        
        # Add missing columns with NaN values
        for col in missing_cols:
            component_df[col] = np.nan
            logger.info(f"Added missing column '{col}' with NaN values for {component}")
        
        # Select required columns
        component_df = component_df[all_columns].copy()
        intersection_df = pd.concat([intersection_df, component_df], axis=0)
    
    # Convert anomaly_label to binary (0/1)
    intersection_df['anomaly_label'] = intersection_df['anomaly_label'].astype(int)
    
    # Perform one-hot encoding for component_type and convert to int
    component_dummies = pd.get_dummies(intersection_df['component_type'], prefix='component').astype(int)
    intersection_df = pd.concat([intersection_df, component_dummies], axis=1)
    
    logger.info(f"Selected {len(common_features)} features (including important features)")
    return intersection_df, report_content

def save_processed_data(df: pd.DataFrame, output_dir: str) -> None:
    """Save processed data to parquet file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'merged_features.parquet')
    
    # Define custom ordering for component_type
    component_order = {'contact': 0, 'pcb': 1, 'ring': 2}
    
    # Create a temporary column for sorting component_type
    df['component_order'] = df['component_type'].map(component_order)
    
    # Sort by component_type first, then by window_start
    df = df.sort_values(['component_order', 'window_start'])
    
    # Remove the temporary sorting column
    df = df.drop(['component_type','component_order'], axis=1)

    df.to_parquet(output_path)
    logger.info(f"Saved merged dataset with shape {df.shape}")

def save_report(report_content: List[str], 
                selected_features: Set[str], 
                report_path: str,
                df: pd.DataFrame) -> None:
    """Save analysis report"""
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    
    # Calculate anomaly statistics
    total_samples = len(df)
    anomaly_samples = df['anomaly_label'].sum()
    anomaly_rate = (anomaly_samples / total_samples) * 100
    
    final_report = [
        "# Feature Selection Analysis Report\n\n",
        "## Anomaly Statistics\n",
        f"Total samples: {total_samples}\n",
        f"Anomaly samples: {anomaly_samples}\n",
        f"Overall anomaly rate: {anomaly_rate:.2f}%\n\n",
        "### Anomaly Rate by Component Type\n",
        "\n## Selected Features\n",
        *report_content,
        "\nFinal selected features:",
        "\n".join(f"- {f}" for f in sorted(selected_features))
    ]
    
    with open(report_path, 'w') as f:
        f.write("\n".join(final_report))
    
    logger.info(f"Saved analysis report to {report_path}")
    
    # Log anomaly statistics
    logger.info(f"Overall anomaly rate: {anomaly_rate:.2f}%")

def main():
    # Load configuration
    config = load_config()
    
    # Create processing pipeline
    data_dict = load_data(config.input_dir)
    correlations = analyze_correlations(data_dict)
    
    final_df, feature_report = select_features(
        correlations,
        data_dict,
        config.correlation_threshold,
        config.high_correlation_threshold
    )
    
    # Save results
    save_processed_data(final_df, config.output_dir)
    save_report(
        feature_report,
        set(final_df.columns) - {'window_start', 'window_end', 'anomaly_label'} - 
        set(col for col in final_df.columns if col.startswith('component_')),
        config.report_path,
        final_df
    )

if __name__ == "__main__":
    main()