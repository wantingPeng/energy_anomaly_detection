"""
Test script for the simplified XGBoost dataloader.
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))

from src.utils.logger import logger
from src.training.xgboost.dataloader import XGBoostDataLoader, create_xgboost_data


def test_direct_loading():
    """Test direct data loading with XGBoostDataLoader."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Direct Data Loading")
    logger.info("=" * 60)
    
    data_path = "experiments/statistic_30_window_features_contact/filtered_window_features.parquet"
    
    try:
        loader = XGBoostDataLoader(
            data_path=data_path,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            target_column="anomaly_label"
        )
        
        logger.info("✓ Data loader created successfully")
        
        # Get data
        X_train, y_train = loader.get_train_data()
        X_val, y_val = loader.get_val_data()
        X_test, y_test = loader.get_test_data()
        
        logger.info(f"✓ Train data: X={X_train.shape}, y={y_train.shape}")
        logger.info(f"✓ Val data: X={X_val.shape}, y={y_val.shape}")
        logger.info(f"✓ Test data: X={X_test.shape}, y={y_test.shape}")
        
        # Get feature names
        feature_names = loader.get_feature_names()
        logger.info(f"✓ Features: {len(feature_names)}")
        logger.info(f"  First 5: {feature_names[:5]}")
        
        # Get scale_pos_weight
        scale_pos_weight = loader.get_scale_pos_weight()
        logger.info(f"✓ Scale pos weight: {scale_pos_weight:.2f}")
        
        # Get data info
        info = loader.get_data_info()
        logger.info(f"✓ Data info keys: {list(info.keys())}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_config():
    """Test data loading with configuration file."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Loading with Config File")
    logger.info("=" * 60)
    
    config_path = "configs/xgboost_timeseries_config.yaml"
    
    try:
        # Load config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Config loaded from: {config_path}")
        logger.info(f"  Data path: {config['data']['data_path']}")
        
        # Create data loader
        data_path = config['data']['data_path']
        loader, data_dict = create_xgboost_data(data_path, config)
        
        logger.info("✓ Data loader and dict created successfully")
        logger.info(f"  Data dict keys: {list(data_dict.keys())}")
        logger.info(f"  X_train shape: {data_dict['X_train'].shape}")
        logger.info(f"  X_val shape: {data_dict['X_val'].shape}")
        logger.info(f"  X_test shape: {data_dict['X_test'].shape}")
        logger.info(f"  Features: {len(data_dict['feature_names'])}")
        logger.info(f"  Scale pos weight: {data_dict['scale_pos_weight']:.2f}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_quality():
    """Test data quality checks."""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Data Quality Checks")
    logger.info("=" * 60)
    
    data_path = "experiments/statistic_30_window_features_contact/filtered_window_features.parquet"
    
    try:
        loader = XGBoostDataLoader(data_path=data_path)
        
        X_train, y_train = loader.get_train_data()
        
        # Check for NaN
        import numpy as np
        has_nan = np.isnan(X_train).any()
        logger.info(f"  Has NaN values: {has_nan}")
        if has_nan:
            logger.warning(f"    NaN count: {np.isnan(X_train).sum()}")
        else:
            logger.info("✓ No NaN values")
        
        # Check for Inf
        has_inf = np.isinf(X_train).any()
        logger.info(f"  Has Inf values: {has_inf}")
        if has_inf:
            logger.warning(f"    Inf count: {np.isinf(X_train).sum()}")
        else:
            logger.info("✓ No Inf values")
        
        # Check label distribution
        unique_labels = np.unique(y_train)
        logger.info(f"  Unique labels: {unique_labels}")
        logger.info(f"  Label 0 count: {(y_train == 0).sum()}")
        logger.info(f"  Label 1 count: {(y_train == 1).sum()}")
        logger.info("✓ Label distribution checked")
        
        # Check data types
        logger.info(f"  X_train dtype: {X_train.dtype}")
        logger.info(f"  y_train dtype: {y_train.dtype}")
        logger.info("✓ Data types checked")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("XGBoost DataLoader Test Suite")
    logger.info("=" * 60)
    
    results = []
    
    results.append(("Direct Loading", test_direct_loading()))
    results.append(("Config Loading", test_with_config()))
    results.append(("Data Quality", test_data_quality()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("✓ ALL TESTS PASSED!")
        logger.info("\nReady to train. Run:")
        logger.info("  python src/training/xgboost/train_xgboost.py")
    else:
        logger.error("✗ SOME TESTS FAILED")
        logger.error("Please fix the issues above before training")
    logger.info("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

