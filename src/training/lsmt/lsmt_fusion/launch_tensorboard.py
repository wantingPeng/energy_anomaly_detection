#!/usr/bin/env python
"""
Launch TensorBoard to visualize training metrics for LSTM Late Fusion models.

This script sets up TensorBoard to monitor and visualize training metrics from
the LSTM Late Fusion models.
"""

import os
import argparse
import subprocess
from pathlib import Path
from src.utils.logger import logger

def main():
    """
    Launch TensorBoard server to visualize training metrics.
    """
    parser = argparse.ArgumentParser(description="Launch TensorBoard for LSTM Late Fusion models")
    parser.add_argument(
        "--logdir", 
        type=str, 
        default="experiments/lstm_late_fusion/tensorboard",
        help="Directory containing TensorBoard event files"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=6006,
        help="Port to run TensorBoard server on"
    )
    args = parser.parse_args()
    
    # Ensure log directory exists
    log_dir = Path(args.logdir)
    if not log_dir.exists():
        logger.warning(f"Log directory {log_dir} does not exist. Creating it.")
        os.makedirs(log_dir, exist_ok=True)
    
    # Launch TensorBoard
    logger.info(f"Launching TensorBoard server at http://localhost:{args.port}")
    logger.info(f"Monitoring log directory: {log_dir.absolute()}")
    
    cmd = f"tensorboard --logdir={args.logdir} --port={args.port}"
    logger.info(f"Running command: {cmd}")
    
    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        logger.info("TensorBoard server stopped by user")
    except Exception as e:
        logger.error(f"Error running TensorBoard: {e}")
        logger.info("Make sure TensorBoard is installed: pip install tensorboard")

if __name__ == "__main__":
    main()
