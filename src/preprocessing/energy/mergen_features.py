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
from dataclasses import dataclass
from functools import partial

@dataclass
class FeatureSelectionConfig:
    input_dir: str
    output_dir: str
    report_path: str

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


def add_missing_columns(df: pd.DataFrame, feature_columns: Set[str], default_value: float) -> pd.DataFrame:
    """Add missing columns to the DataFrame with a default value."""
    missing_cols = [col for col in feature_columns if col not in df.columns]
    for col in missing_cols:
        df[col] = default_value
        logger.info(f"Added missing column '{col}' with {default_value} values")
    return df

def encode_and_convert_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Convert anomaly_label to binary and perform one-hot encoding for component_type."""
    df['anomaly_label'] = df['anomaly_label'].astype(int)
    component_dummies = pd.get_dummies(df['component_type'], prefix='component').astype(int)
    df = pd.concat([df, component_dummies], axis=1)
    return df

def merge_features(data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, List[str]]:
    """Merge features from all component types, aligning columns and filling missing values with 0."""
    report_content = []
    all_columns = set()
    for df in data_dict.values():
        all_columns.update(df.columns)
    
    non_feature_cols = {'window_start', 'window_end', 'component_type', 'anomaly_label'}
    feature_columns = all_columns - non_feature_cols
    additional_columns = ['window_start', 'window_end', 'component_type', 'anomaly_label']
    all_columns = list(feature_columns) + additional_columns
    merged_df = pd.DataFrame()
    
    for component, df in data_dict.items():
        df = add_missing_columns(df, feature_columns, default_value=0)
        df = df[all_columns].copy()
        df = encode_and_convert_labels(df)
        merged_df = pd.concat([merged_df, df], axis=0)
    
    report_content.extend([
        f"\nTotal number of features: {len(feature_columns)}",
        "\nFeature columns:",
        "\n".join(f"- {f}" for f in sorted(feature_columns))
    ])
    
    logger.info(f"Merged {len(feature_columns)} features from all components")
    return merged_df, report_content


def save_processed_data(df: pd.DataFrame, output_dir: str) -> None:
    """Save processed data to parquet file"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'merged_features_no_correlation.parquet')
    
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
    
    final_df, feature_report = merge_features(data_dict)
    
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