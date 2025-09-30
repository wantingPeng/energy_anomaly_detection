import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from scipy import stats
from scipy.fft import fft, fftfreq
from scipy.signal import welch
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


def calculate_frequency_features(values: np.ndarray, sampling_rate: float = 1.0) -> dict:
    """
    计算频域特征（精简版）
    
    Args:
        values: 时间序列数据
        sampling_rate: 采样率，默认为1.0
        
    Returns:
        dict: 精简后的频域特征字典
    """
    freq_features = {}
    
    try:
        if len(values) < 4:  # FFT需要足够的数据点
            return {
                'dominant_frequency': 0.0,
                'spectral_peak': 0.0,
                'spectral_flatness': 0.0,
                'freq_energy_ratio_low_high': 0.0
            }
        
        # 计算FFT
        fft_values = np.abs(fft(values))
        freqs = fftfreq(len(values), d=1/sampling_rate)
        
        # 只取正频率部分
        positive_freq_idx = freqs > 0
        fft_positive = fft_values[positive_freq_idx]
        freqs_positive = freqs[positive_freq_idx]
        
        if len(fft_positive) == 0:
            return {
                'dominant_frequency': 0.0,
                'spectral_peak': 0.0,
                'spectral_flatness': 0.0,
                'freq_energy_ratio_low_high': 0.0
            }
        
        # 1. 主频率 (dominant frequency)
        dominant_freq_idx = np.argmax(fft_positive)
        dominant_frequency = freqs_positive[dominant_freq_idx]
        
        # 2. 频谱峰值
        spectral_peak = np.max(fft_positive)
        
        # 3. 频谱平坦度 (spectral flatness)
        # 几何平均 / 算术平均
        geometric_mean = np.exp(np.mean(np.log(fft_positive + 1e-10)))
        arithmetic_mean = np.mean(fft_positive)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)
        
        # 4. 频带能量比
        max_freq = freqs_positive[-1]
        low_freq_threshold = max_freq / 3
        high_freq_threshold = 2 * max_freq / 3
        
        low_freq_mask = freqs_positive <= low_freq_threshold
        high_freq_mask = freqs_positive > high_freq_threshold
        
        low_freq_energy = np.sum(fft_positive[low_freq_mask])
        high_freq_energy = np.sum(fft_positive[high_freq_mask])
        
        freq_energy_ratio_low_high = low_freq_energy / (high_freq_energy + 1e-10)
        
        freq_features = {
            'dominant_frequency': dominant_frequency,
            'spectral_peak': spectral_peak,
            'spectral_flatness': spectral_flatness,
            'freq_energy_ratio_low_high': freq_energy_ratio_low_high
        }
        
    except Exception as e:
        logger.warning(f"Frequency features calculation failed: {str(e)}")
        freq_features = {
            'dominant_frequency': 0.0,
            'spectral_peak': 0.0,
            'spectral_flatness': 0.0,
            'freq_energy_ratio_low_high': 0.0
        }
    
    return freq_features


def calculate_welch_features(values: np.ndarray, sampling_rate: float = 1.0) -> dict:
    """
    使用Welch方法计算功率谱密度特征（精简版）
    
    Args:
        values: 时间序列数据
        sampling_rate: 采样率
        
    Returns:
        dict: 精简后的Welch方法频域特征
    """
    welch_features = {}
    
    try:
        if len(values) < 8:  # Welch方法需要足够的数据点
            return {
                'welch_peak_power': 0.0,
                'welch_total_power': 0.0
            }
        
        # 计算Welch功率谱密度
        freqs, psd = welch(values, fs=sampling_rate, nperseg=min(len(values)//2, 256))
        
        if len(psd) == 0:
            return {
                'welch_peak_power': 0.0,
                'welch_total_power': 0.0
            }
        
        # 峰值功率
        welch_peak_power = np.max(psd)
        
        # 总功率
        welch_total_power = np.sum(psd)
        
        welch_features = {
            'welch_peak_power': welch_peak_power,
            'welch_total_power': welch_total_power
        }
        
    except Exception as e:
        logger.warning(f"Welch features calculation failed: {str(e)}")
        welch_features = {
            'welch_peak_power': 0.0,
            'welch_total_power': 0.0
        }
    
    return welch_features


def calculate_window_features(window: pd.DataFrame) -> pd.Series:
    """
    Calculate statistical features for a given window (精简版).
    移除了高度相关的统计特征，以降低维度并减少计算时间
    
    Args:
        window: DataFrame containing the window data
        
    Returns:
        pd.Series: 精简后的统计特征
    """
    features = {}
    
    # 需要排除的元数据列
    metadata_cols = ['TimeStamp','anomaly_label']
    
    # 获取数值类型的列，排除元数据列
    numeric_cols = window.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # 对每个特征列计算统计量
    for col in feature_cols:
        values = window[col].values
        if len(values) > 0:  # 确保窗口内有数据
            # 基本统计量 - 保留最具辨别力的特征
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            features.update({
                f"{col}_mean": mean_val,
                f"{col}_std": std_val,
            })
            
            # 突变检测 - 对异常检测非常有用
            features[f"{col}_change_point_score"] = np.max(np.abs(np.diff(values))) if len(values) > 1 else 0.0
            
            # 统计检验 - 非常适合检测分布变化
            features[f"{col}_ks_test_statistic"] = calculate_ks_test(values)
            
            # 保留偏度和峰度 - 描述分布形状的重要指标
            try:
                if std_val > 1e-10:
                    skew_val = stats.skew(values)
                    features[f"{col}_skew"] = skew_val if not np.isnan(skew_val) else 0.0
                else:
                    features[f"{col}_skew"] = 0.0
            except Exception as e:
                logger.warning(f"Skewness calculation failed for {col}: {str(e)}")
                features[f"{col}_skew"] = 0.0
            
            # 计算趋势斜率 - 捕捉序列趋势信息
            try:
                if len(values) > 1 and np.any(np.diff(values) != 0):
                    x = np.arange(len(values))
                    slope, _ = np.polyfit(x, values, 1)
                    features[f"{col}_trend_slope"] = slope
                else:
                    features[f"{col}_trend_slope"] = 0.0
            except Exception as e:
                logger.warning(f"Trend slope calculation failed for {col}: {str(e)}")
                features[f"{col}_trend_slope"] = 0.0
            
            # 计算精简后的频域特征
            freq_features = calculate_frequency_features(values, sampling_rate=1.0)
            for freq_key, freq_val in freq_features.items():
                features[f"{col}_{freq_key}"] = freq_val
            
            # 计算精简后的Welch功率谱密度特征
            welch_features = calculate_welch_features(values, sampling_rate=1.0)
            for welch_key, welch_val in welch_features.items():
                features[f"{col}_{welch_key}"] = welch_val
    
    # 添加时间相关的元数据
    features['TimeStamp'] = window['TimeStamp'].min()  # 使用窗口起始时间作为TimeStamp

    return pd.Series(features)

