#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import time

# Create a logger
logger = logging.getLogger("energy_anomaly_detection")
logger.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")

# Create console handler and set level to info
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add console handler to logger
logger.addHandler(console_handler)

# Function to add file handler dynamically
def add_file_handler(log_file=None):
    """
    Add a file handler to the logger
    
    Args:
        log_file (str): Path to the log file. If None, a default path will be used.
        
    Returns:
        str: Path to the log file
    """
    if log_file is None:
        # Create default log file name based on timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_file = f"experiments/logs/default_{timestamp}.log"
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create file handler and set level to info
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Add file handler to logger
    logger.addHandler(file_handler)
    
    return log_file

# Export the logger
__all__ = ["logger", "add_file_handler"]