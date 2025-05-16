#!/usr/bin/env python
"""
Script to run the sliding window preprocessing for LSTM training data.
"""

import os
import yaml
import argparse
from datetime import datetime
from pathlib import Path

from src.utils.logger import logger
from src.preprocessing.energy.lstm.slinding_window import main

def create_default_config():
    """Create default configuration if config file doesn't exist."""
    config = {
        'paths': {
            'input_dir': 'Data/processed/lsmt/standerScaler',
            'output_dir': 'Data/processed/lsmt/sliding_window',
            'reports_dir': 'reports/sliding_window',
            'anomaly_dict': 'Data/processed/anomaly_dict.pkl'
        },
        'sliding_window': {
            'window_size': 60,  # 60 seconds window
            'step_size': 10,    # 10 seconds step
            'anomaly_threshold': 0.5  # 50% overlap required to label as anomaly
        }
    }
    
    config_dir = Path('configs')
    config_dir.mkdir(exist_ok=True)
    
    config_path = config_dir / 'lsmt_preprocessing.yaml'
    if not config_path.exists():
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        logger.info(f"Created default config at {config_path}")
    else:
        logger.info(f"Config already exists at {config_path}")
    
    return config_path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run sliding window preprocessing")
    parser.add_argument(
        "--config", 
        type=str,
        help="Path to config file (default: configs/lsmt_preprocessing.yaml)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    # Set up logging with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/sliding_window_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Parse arguments
    args = parse_args()
    
    # Create default config if not specified
    if not args.config:
        config_path = create_default_config()
        logger.info(f"Using default config at {config_path}")
    
    # Run the main function
    logger.info("Starting sliding window preprocessing")
    main()
    logger.info("Sliding window preprocessing completed") 