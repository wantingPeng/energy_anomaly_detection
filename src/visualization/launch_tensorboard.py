import argparse
import os
import subprocess
from pathlib import Path
from src.utils.logger import logger

def launch_tensorboard(log_dir="experiments/tensorboard", port=6006):
    """
    Launch TensorBoard to visualize training metrics
    
    Args:
        log_dir (str): Directory containing TensorBoard logs
        port (int): Port to run TensorBoard on
    """
    # Ensure log directory exists
    if not os.path.exists(log_dir):
        logger.error(f"TensorBoard log directory not found: {log_dir}")
        logger.info(f"Creating empty directory: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
    
    # Build the command
    cmd = f"tensorboard --logdir {log_dir} --port {port} --bind_all"
    
    logger.info(f"Launching TensorBoard with command: {cmd}")
    logger.info(f"TensorBoard will be available at: http://localhost:{port}")
    logger.info("Press Ctrl+C to stop TensorBoard")
    
    # Start TensorBoard
    try:
        subprocess.run(cmd, shell=True)
    except KeyboardInterrupt:
        logger.info("TensorBoard stopped by user")
    except Exception as e:
        logger.error(f"Error launching TensorBoard: {str(e)}")

def find_latest_run(base_dir="experiments/tensorboard/lstm"):
    """
    Find the latest run directory in the TensorBoard logs
    
    Args:
        base_dir (str): Base directory containing run folders
        
    Returns:
        str: Path to the latest run directory
    """
    if not os.path.exists(base_dir):
        logger.warning(f"Base directory not found: {base_dir}")
        return base_dir
    
    # List all run directories
    run_dirs = [d for d in os.listdir(base_dir) if d.startswith('run_')]
    
    if not run_dirs:
        logger.warning(f"No run directories found in {base_dir}")
        return base_dir
    
    # Sort runs by timestamp (assuming format run_YYYYMMDD_HHMMSS)
    latest_run = sorted(run_dirs)[-1]
    latest_run_path = os.path.join(base_dir, latest_run)
    
    logger.info(f"Latest run found: {latest_run_path}")
    return latest_run_path

def main():
    """Main function to run the script"""
    parser = argparse.ArgumentParser(description='Launch TensorBoard to visualize training metrics')
    parser.add_argument('--logdir', type=str, help='Directory containing TensorBoard logs')
    parser.add_argument('--port', type=int, default=6006, help='Port to run TensorBoard on')
    parser.add_argument('--latest', action='store_true', help='View only the latest run (for LSTM model)')
    
    args = parser.parse_args()
    
    log_dir = args.logdir
    
    # If latest flag is set, find the latest run
    if args.latest:
        if args.logdir:
            base_dir = args.logdir
        else:
            base_dir = "experiments/tensorboard/lstm"
            
        log_dir = find_latest_run(base_dir)
    elif not log_dir:
        # Default log directory
        log_dir = "experiments/tensorboard"
    
    # Launch TensorBoard
    launch_tensorboard(log_dir, args.port)

if __name__ == "__main__":
    main() 