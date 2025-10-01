"""
Quick test script to verify XGBoost setup and data loading.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger

def test_imports():
    """Test if all required packages are installed."""
    logger.info("Testing imports...")
    
    try:
        import xgboost as xgb
        logger.info(f"✓ XGBoost version: {xgb.__version__}")
    except ImportError as e:
        logger.error(f"✗ XGBoost not installed: {e}")
        return False
    
    try:
        import pandas as pd
        logger.info(f"✓ Pandas version: {pd.__version__}")
    except ImportError as e:
        logger.error(f"✗ Pandas not installed: {e}")
        return False
    
    try:
        import numpy as np
        logger.info(f"✓ NumPy version: {np.__version__}")
    except ImportError as e:
        logger.error(f"✗ NumPy not installed: {e}")
        return False
    
    try:
        import sklearn
        logger.info(f"✓ Scikit-learn version: {sklearn.__version__}")
    except ImportError as e:
        logger.error(f"✗ Scikit-learn not installed: {e}")
        return False
    
    try:
        import yaml
        logger.info(f"✓ PyYAML installed")
    except ImportError as e:
        logger.error(f"✗ PyYAML not installed: {e}")
        return False
    
    try:
        import matplotlib
        logger.info(f"✓ Matplotlib version: {matplotlib.__version__}")
    except ImportError as e:
        logger.error(f"✗ Matplotlib not installed: {e}")
        return False
    
    try:
        import seaborn
        logger.info(f"✓ Seaborn version: {seaborn.__version__}")
    except ImportError as e:
        logger.error(f"✗ Seaborn not installed: {e}")
        return False
    
    return True


def test_config():
    """Test if config file exists and is valid."""
    logger.info("\nTesting config file...")
    
    import yaml
    config_path = project_root / "configs" / "xgboost_timeseries_config.yaml"
    
    if not config_path.exists():
        logger.error(f"✗ Config file not found: {config_path}")
        return False
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"✓ Config file loaded successfully")
        logger.info(f"  Data path: {config['data']['data_path']}")
        logger.info(f"  Model objective: {config['model']['objective']}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return False


def test_data_file():
    """Test if data file exists."""
    logger.info("\nTesting data file...")
    
    import yaml
    config_path = project_root / "configs" / "xgboost_timeseries_config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = project_root / config['data']['data_path']
    
    if not data_path.exists():
        logger.error(f"✗ Data file not found: {data_path}")
        return False
    
    logger.info(f"✓ Data file exists: {data_path}")
    
    # Check file size
    import os
    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)
    logger.info(f"  File size: {file_size_mb:.2f} MB")
    
    return True


def test_data_loading():
    """Test if data can be loaded."""
    logger.info("\nTesting data loading...")
    
    try:
        import pandas as pd
        import yaml
        
        config_path = project_root / "configs" / "xgboost_timeseries_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        data_path = project_root / config['data']['data_path']
        
        # Load first 1000 rows as a test
        df = pd.read_parquet(data_path)
        
        logger.info(f"✓ Data loaded successfully")
        logger.info(f"  Shape: {df.shape}")
        logger.info(f"  Columns: {list(df.columns)[:5]}...")
        
        if 'anomaly_label' in df.columns:
            anomaly_ratio = df['anomaly_label'].mean()
            logger.info(f"  Anomaly ratio: {anomaly_ratio:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}")
        return False


def test_module_imports():
    """Test if our custom modules can be imported."""
    logger.info("\nTesting custom module imports...")
    
    try:
        from src.training.xgboost.dataloader import XGBoostDataLoader, create_xgboost_data
        logger.info("✓ DataLoader imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import DataLoader: {e}")
        return False
    
    try:
        from src.training.xgboost.xgboost_model import XGBoostAnomalyDetector
        logger.info("✓ XGBoostAnomalyDetector imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import XGBoostAnomalyDetector: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("XGBoost Setup Test")
    logger.info("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_config()
    all_passed &= test_data_file()
    all_passed &= test_data_loading()
    all_passed &= test_module_imports()
    
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ All tests passed! Ready to train.")
        logger.info("\nTo start training, run:")
        logger.info("  python src/training/xgboost/train_xgboost.py")
    else:
        logger.error("✗ Some tests failed. Please fix the issues above.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()


