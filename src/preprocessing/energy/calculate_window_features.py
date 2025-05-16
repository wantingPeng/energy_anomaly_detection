import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from src.utils.logger import logger



def calculate_ks_test(window_data: np.ndarray) -> float:
    """
    计算Kolmogorov-Smirnov检验统计量
    """
    try:
        # 与正态分布进行比较
        ks_statistic, _ = stats.kstest(window_data, 'norm')
        return ks_statistic
    except Exception as e:
        logger.warning(f"KS test calculation failed: {str(e)}")
        return 0.0


def calculate_window_features(window: pd.DataFrame) -> pd.Series:
    """
    Calculate statistical features for a given window.
    
    Args:
        window: DataFrame containing the window data
        
    Returns:
        pd.Series: Statistical features for the window
    """
    features = {}
    
    # 需要排除的元数据列
    metadata_cols = ['IsOutlier', 'time_diff', 'segment_id', 'ID', 'TimeStamp', 'Station']
    
    # 获取数值类型的列，排除元数据列
    numeric_cols = window.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # 对每个特征列计算统计量
    for col in feature_cols:
        values = window[col].values
        if len(values) > 0:  # 确保窗口内有数据
            # 基本统计量
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)
            
            features.update({
                f"{col}_mean": mean_val,
                f"{col}_std": std_val,
                f"{col}_min": min_val,
                f"{col}_max": max_val,
                f"{col}_range": max_val - min_val,
            })
            
            # 突变检测
            features[f"{col}_change_point_score"] = np.max(np.abs(np.diff(values)))
            
            # 计算绝对能量
            features[f"{col}_abs_energy"] = np.sum(values ** 2)
            
            # 计算尖峰计数
            spike_threshold = mean_val + 3 * std_val
            features[f"{col}_spike_count"] = np.sum(values > spike_threshold)
            
            # 统计检验
            features[f"{col}_ks_test_statistic"] = calculate_ks_test(values)
            
            # 安全计算z-score
            if std_val > 1e-10:  # 避免除以接近零的标准差
                features[f"{col}_z_score"] = np.mean((values - mean_val) / std_val)
            else:
                features[f"{col}_z_score"] = 0.0
            
            # 安全计算偏度
            try:
                if std_val > 1e-10:
                    skew_val = stats.skew(values)
                    features[f"{col}_skew"] = skew_val if not np.isnan(skew_val) else 0.0
                else:
                    features[f"{col}_skew"] = 0.0
            except Exception as e:
                logger.warning(f"Skewness calculation failed for {col}: {str(e)}")
                features[f"{col}_skew"] = 0.0
            
            # 计算趋势斜率
            try:
                x = np.arange(len(values))
                if len(values) > 1 and np.any(np.diff(values) != 0):
                    slope, _ = np.polyfit(x, values, 1)
                    features[f"{col}_trend_slope"] = slope
                else:
                    features[f"{col}_trend_slope"] = 0.0
            except Exception as e:
                logger.warning(f"Trend slope calculation failed for {col}: {str(e)}")
                features[f"{col}_trend_slope"] = 0.0
            
            # 计算差分特征
            if len(values) > 1:
                diffs = np.diff(values)
                features[f"{col}_diff_mean"] = np.mean(diffs) if len(diffs) > 0 else 0.0
                features[f"{col}_diff_std"] = np.std(diffs) if len(diffs) > 0 else 0.0
            else:
                features[f"{col}_diff_mean"] = 0.0
                features[f"{col}_diff_std"] = 0.0
    
    # 添加时间相关的元数据
    features['window_start'] = window['TimeStamp'].min()
    features['window_end'] = window['TimeStamp'].max()
    features['segment_id'] = window['segment_id'].iloc[0]
    
    return pd.Series(features)

