"""
探索性频域分析脚本
对Contacting机器的能源数据进行频域分析，对比正常和异常时段的频谱特征
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import logger

# 设置中文字体和绘图风格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class FourierAnalysisExploration:
    """频域分析探索类"""
    
    def __init__(self, data_path, anomaly_path, output_dir):
        """
        初始化
        
        Args:
            data_path: 数据文件路径
            anomaly_path: 异常时间字典路径
            output_dir: 输出目录
        """
        self.data_path = data_path
        self.anomaly_path = anomaly_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.sampling_rate = 1.0  # 1 Hz
        self.df = None
        self.anomaly_dict = None
        
        logger.info(f"初始化频域分析，输出目录: {self.output_dir}")
    
    def load_data(self):
        """加载数据"""
        logger.info(f"开始加载数据: {self.data_path}")
        
        # 只加载需要的列
        columns_to_load = [
            'TimeStamp', 'aActivePower_L1', 'aActivePower_L2', 'aActivePower_L3',
            'rTotalActivePower', 'aCurrentL1', 'aCurrentL2', 'aCurrentL3'
        ]
        
        self.df = pd.read_parquet(self.data_path, columns=columns_to_load)
        logger.info(f"数据加载完成，shape: {self.df.shape}")
        logger.info(f"时间范围: {self.df['TimeStamp'].min()} 到 {self.df['TimeStamp'].max()}")
        
        # 加载异常时间字典
        with open(self.anomaly_path, 'rb') as f:
            self.anomaly_dict = pickle.load(f)
        
        logger.info(f"异常时间段数量: {len(self.anomaly_dict.get('Kontaktieren', []))}")
        
        return self.df, self.anomaly_dict
    
    def sample_normal_windows(self, window_size_seconds=3600, n_samples=20):
        """
        采样正常时段
        
        Args:
            window_size_seconds: 窗口大小（秒）
            n_samples: 采样数量
            
        Returns:
            正常时段的数据窗口列表
        """
        logger.info(f"开始采样正常时段，窗口大小={window_size_seconds}秒，采样数量={n_samples}")
        
        anomaly_periods = self.anomaly_dict.get('Kontaktieren', [])
        
        # 创建所有时间的索引
        all_timestamps = set(self.df['TimeStamp'])
        
        # 创建异常时间的集合
        anomaly_timestamps = set()
        for start, end in anomaly_periods:
            # 扩展异常时间段，前后各加10分钟缓冲
            start_extended = start - timedelta(minutes=10)
            end_extended = end + timedelta(minutes=10)
            
            mask = (self.df['TimeStamp'] >= start_extended) & (self.df['TimeStamp'] <= end_extended)
            anomaly_timestamps.update(self.df.loc[mask, 'TimeStamp'])
        
        logger.info(f"异常时间点数量（含缓冲）: {len(anomaly_timestamps)}")
        
        # 正常时间点
        normal_timestamps = sorted(all_timestamps - anomaly_timestamps)
        logger.info(f"正常时间点数量: {len(normal_timestamps)}")
        
        # 随机采样正常窗口
        normal_windows = []
        max_attempts = n_samples * 10
        attempts = 0
        
        while len(normal_windows) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # 随机选择一个起始时间
            start_idx = np.random.randint(0, len(normal_timestamps) - window_size_seconds)
            start_time = normal_timestamps[start_idx]
            end_time = start_time + timedelta(seconds=window_size_seconds)
            
            # 提取窗口数据
            mask = (self.df['TimeStamp'] >= start_time) & (self.df['TimeStamp'] < end_time)
            window_data = self.df.loc[mask].copy()
            
            # 检查窗口是否完整且连续
            if len(window_data) >= window_size_seconds * 0.95:  # 允许5%的数据缺失
                normal_windows.append({
                    'start': start_time,
                    'end': end_time,
                    'data': window_data
                })
        
        logger.info(f"成功采样正常窗口数量: {len(normal_windows)}")
        return normal_windows
    
    def sample_anomaly_windows(self, window_size_seconds=3600, n_samples=20):
        """
        采样异常时段
        
        Args:
            window_size_seconds: 窗口大小（秒）
            n_samples: 采样数量
            
        Returns:
            异常时段的数据窗口列表
        """
        logger.info(f"开始采样异常时段，窗口大小={window_size_seconds}秒，采样数量={n_samples}")
        
        anomaly_periods = self.anomaly_dict.get('Kontaktieren', [])
        
        # 筛选出足够长的异常时段
        valid_anomalies = []
        for start, end in anomaly_periods:
            duration = (end - start).total_seconds()
            if duration >= window_size_seconds:
                valid_anomalies.append((start, end, duration))
        
        logger.info(f"足够长的异常时段数量: {len(valid_anomalies)}")
        
        # 随机采样异常窗口
        anomaly_windows = []
        
        if len(valid_anomalies) > 0:
            # 如果异常时段数量足够，随机采样
            sample_size = min(n_samples, len(valid_anomalies))
            sampled_anomalies = np.random.choice(len(valid_anomalies), sample_size, replace=False)
            
            for idx in sampled_anomalies:
                start, end, duration = valid_anomalies[idx]
                
                # 从异常时段中随机选择一个窗口
                max_offset = int(duration - window_size_seconds)
                if max_offset > 0:
                    offset = np.random.randint(0, max_offset)
                    window_start = start + timedelta(seconds=offset)
                else:
                    window_start = start
                
                window_end = window_start + timedelta(seconds=window_size_seconds)
                
                # 提取窗口数据
                mask = (self.df['TimeStamp'] >= window_start) & (self.df['TimeStamp'] < window_end)
                window_data = self.df.loc[mask].copy()
                
                if len(window_data) >= window_size_seconds * 0.95:
                    anomaly_windows.append({
                        'start': window_start,
                        'end': window_end,
                        'data': window_data
                    })
        
        logger.info(f"成功采样异常窗口数量: {len(anomaly_windows)}")
        return anomaly_windows
    
    def compute_fft_features(self, data_array, feature_name):
        """
        计算FFT和功率谱密度
        
        Args:
            data_array: 时间序列数据
            feature_name: 特征名称
            
        Returns:
            频率、FFT幅值、功率谱密度
        """
        # 去趋势
        data_detrended = signal.detrend(data_array)
        
        # 应用汉宁窗减少频谱泄漏
        window = signal.windows.hann(len(data_detrended))
        data_windowed = data_detrended * window
        
        # FFT
        fft_values = fft(data_windowed)
        n = len(data_windowed)
        fft_freq = fftfreq(n, d=1/self.sampling_rate)
        
        # 只取正频率部分
        positive_freq_idx = fft_freq > 0
        fft_freq_positive = fft_freq[positive_freq_idx]
        fft_amplitude = np.abs(fft_values[positive_freq_idx]) * 2 / n
        
        # 功率谱密度
        psd_freq, psd = signal.welch(data_detrended, fs=self.sampling_rate, 
                                      nperseg=min(256, len(data_detrended)))
        
        return fft_freq_positive, fft_amplitude, psd_freq, psd
    
    def analyze_windows(self, windows, window_type='normal'):
        """
        分析窗口的频域特征
        
        Args:
            windows: 窗口数据列表
            window_type: 窗口类型（'normal' 或 'anomaly'）
            
        Returns:
            频域特征统计
        """
        logger.info(f"开始分析{window_type}窗口的频域特征，数量: {len(windows)}")
        
        features_to_analyze = ['aActivePower_L1', 'aActivePower_L2', 'aActivePower_L3', 'rTotalActivePower']
        
        results = {feature: {
            'fft_freqs': [],
            'fft_amplitudes': [],
            'psd_freqs': [],
            'psd_values': [],
            'dominant_freqs': [],
            'dominant_amps': [],
            'total_power': []
        } for feature in features_to_analyze}
        
        for i, window in enumerate(windows):
            data = window['data']
            
            for feature in features_to_analyze:
                if feature not in data.columns:
                    continue
                
                data_array = data[feature].values
                
                # 检查数据有效性
                if len(data_array) < 100 or np.isnan(data_array).all():
                    continue
                
                # 处理NaN值
                data_array = pd.Series(data_array).interpolate(method='linear').fillna(method='bfill').fillna(method='ffill').values
                
                # 计算FFT
                fft_freq, fft_amp, psd_freq, psd = self.compute_fft_features(data_array, feature)
                
                results[feature]['fft_freqs'].append(fft_freq)
                results[feature]['fft_amplitudes'].append(fft_amp)
                results[feature]['psd_freqs'].append(psd_freq)
                results[feature]['psd_values'].append(psd)
                
                # 找出主导频率（前5个最大幅值对应的频率）
                top_n = 5
                top_indices = np.argsort(fft_amp)[-top_n:][::-1]
                results[feature]['dominant_freqs'].append(fft_freq[top_indices])
                results[feature]['dominant_amps'].append(fft_amp[top_indices])
                
                # 总功率
                total_power = np.sum(psd)
                results[feature]['total_power'].append(total_power)
        
        logger.info(f"{window_type}窗口频域分析完成")
        return results
    
    def plot_comparison(self, normal_results, anomaly_results, feature_name):
        """
        绘制正常和异常时段的频谱对比图
        
        Args:
            normal_results: 正常时段的频域结果
            anomaly_results: 异常时段的频域结果
            feature_name: 特征名称
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Frequency Domain Analysis: {feature_name}', fontsize=16, fontweight='bold')
        
        # 1. FFT幅值谱对比（取平均）
        ax1 = axes[0, 0]
        if len(normal_results[feature_name]['fft_amplitudes']) > 0:
            # 计算平均FFT谱
            normal_fft_freq = normal_results[feature_name]['fft_freqs'][0]
            normal_fft_amp_mean = np.mean(normal_results[feature_name]['fft_amplitudes'], axis=0)
            ax1.semilogy(normal_fft_freq, normal_fft_amp_mean, label='Normal', alpha=0.7, linewidth=2)
        
        if len(anomaly_results[feature_name]['fft_amplitudes']) > 0:
            anomaly_fft_freq = anomaly_results[feature_name]['fft_freqs'][0]
            anomaly_fft_amp_mean = np.mean(anomaly_results[feature_name]['fft_amplitudes'], axis=0)
            ax1.semilogy(anomaly_fft_freq, anomaly_fft_amp_mean, label='Anomaly', alpha=0.7, linewidth=2)
        
        ax1.set_xlabel('Frequency (Hz)')
        ax1.set_ylabel('Amplitude')
        ax1.set_title('Average FFT Spectrum')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 功率谱密度对比
        ax2 = axes[0, 1]
        if len(normal_results[feature_name]['psd_values']) > 0:
            normal_psd_freq = normal_results[feature_name]['psd_freqs'][0]
            normal_psd_mean = np.mean(normal_results[feature_name]['psd_values'], axis=0)
            ax2.semilogy(normal_psd_freq, normal_psd_mean, label='Normal', alpha=0.7, linewidth=2)
        
        if len(anomaly_results[feature_name]['psd_values']) > 0:
            anomaly_psd_freq = anomaly_results[feature_name]['psd_freqs'][0]
            anomaly_psd_mean = np.mean(anomaly_results[feature_name]['psd_values'], axis=0)
            ax2.semilogy(anomaly_psd_freq, anomaly_psd_mean, label='Anomaly', alpha=0.7, linewidth=2)
        
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power Spectral Density')
        ax2.set_title('Average Power Spectral Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 主导频率分布
        ax3 = axes[1, 0]
        normal_dom_freqs = np.concatenate(normal_results[feature_name]['dominant_freqs']) if normal_results[feature_name]['dominant_freqs'] else []
        anomaly_dom_freqs = np.concatenate(anomaly_results[feature_name]['dominant_freqs']) if anomaly_results[feature_name]['dominant_freqs'] else []
        
        if len(normal_dom_freqs) > 0:
            ax3.hist(normal_dom_freqs, bins=30, alpha=0.6, label='Normal', edgecolor='black')
        if len(anomaly_dom_freqs) > 0:
            ax3.hist(anomaly_dom_freqs, bins=30, alpha=0.6, label='Anomaly', edgecolor='black')
        
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Count')
        ax3.set_title('Dominant Frequency Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 总功率对比
        ax4 = axes[1, 1]
        normal_powers = normal_results[feature_name]['total_power']
        anomaly_powers = anomaly_results[feature_name]['total_power']
        
        box_data = []
        labels = []
        if len(normal_powers) > 0:
            box_data.append(normal_powers)
            labels.append('Normal')
        if len(anomaly_powers) > 0:
            box_data.append(anomaly_powers)
            labels.append('Anomaly')
        
        if box_data:
            bp = ax4.boxplot(box_data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
                patch.set_facecolor(color)
        
        ax4.set_ylabel('Total Power')
        ax4.set_title('Total Power Comparison')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = self.output_dir / f'fourier_comparison_{feature_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"保存频谱对比图: {output_path}")
        plt.close()
    
    def extract_frequency_bands(self, normal_results, anomaly_results):
        """
        提取不同频段的能量分布
        
        Args:
            normal_results: 正常时段结果
            anomaly_results: 异常时段结果
            
        Returns:
            频段能量统计
        """
        logger.info("开始提取频段能量分布")
        
        # 定义频段
        freq_bands = {
            'very_low': (0, 0.0001),      # 长周期趋势
            'low': (0.0001, 0.001),        # 小时级周期
            'mid': (0.001, 0.01),          # 分钟级周期
            'high': (0.01, 0.5)            # 秒级周期
        }
        
        results_summary = {}
        
        for feature in ['aActivePower_L1', 'aActivePower_L2', 'aActivePower_L3', 'rTotalActivePower']:
            results_summary[feature] = {
                'normal': {band: [] for band in freq_bands},
                'anomaly': {band: [] for band in freq_bands}
            }
            
            # 处理正常时段
            for psd_freq, psd in zip(normal_results[feature]['psd_freqs'], 
                                     normal_results[feature]['psd_values']):
                for band_name, (f_low, f_high) in freq_bands.items():
                    mask = (psd_freq >= f_low) & (psd_freq < f_high)
                    band_power = np.sum(psd[mask])
                    results_summary[feature]['normal'][band_name].append(band_power)
            
            # 处理异常时段
            for psd_freq, psd in zip(anomaly_results[feature]['psd_freqs'],
                                     anomaly_results[feature]['psd_values']):
                for band_name, (f_low, f_high) in freq_bands.items():
                    mask = (psd_freq >= f_low) & (psd_freq < f_high)
                    band_power = np.sum(psd[mask])
                    results_summary[feature]['anomaly'][band_name].append(band_power)
        
        # 保存结果
        summary_file = self.output_dir / 'frequency_band_summary.txt'
        with open(summary_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("频段能量分布统计\n")
            f.write("=" * 80 + "\n\n")
            
            for feature in results_summary:
                f.write(f"\n{'='*80}\n")
                f.write(f"特征: {feature}\n")
                f.write(f"{'='*80}\n\n")
                
                for band_name in freq_bands:
                    normal_powers = results_summary[feature]['normal'][band_name]
                    anomaly_powers = results_summary[feature]['anomaly'][band_name]
                    
                    f.write(f"  频段: {band_name} ({freq_bands[band_name][0]:.6f} - {freq_bands[band_name][1]:.6f} Hz)\n")
                    
                    if normal_powers:
                        f.write(f"    正常时段平均能量: {np.mean(normal_powers):.6e} ± {np.std(normal_powers):.6e}\n")
                    if anomaly_powers:
                        f.write(f"    异常时段平均能量: {np.mean(anomaly_powers):.6e} ± {np.std(anomaly_powers):.6e}\n")
                    
                    if normal_powers and anomaly_powers:
                        ratio = np.mean(anomaly_powers) / np.mean(normal_powers) if np.mean(normal_powers) > 0 else 0
                        f.write(f"    异常/正常比值: {ratio:.3f}\n")
                    
                    f.write("\n")
        
        logger.info(f"频段能量统计保存至: {summary_file}")
        return results_summary
    
    def run(self):
        """运行完整的频域分析流程"""
        logger.info("="*80)
        logger.info("开始探索性频域分析")
        logger.info("="*80)
        
        # 1. 加载数据
        self.load_data()
        
        # 2. 采样窗口（1小时窗口）
        window_size = 3600  # 1小时
        n_samples = 20
        
        normal_windows = self.sample_normal_windows(window_size_seconds=window_size, n_samples=n_samples)
        anomaly_windows = self.sample_anomaly_windows(window_size_seconds=window_size, n_samples=n_samples)
        
        # 3. 频域分析
        normal_results = self.analyze_windows(normal_windows, window_type='normal')
        anomaly_results = self.analyze_windows(anomaly_windows, window_type='anomaly')
        
        # 4. 可视化对比
        features_to_plot = ['aActivePower_L1', 'aActivePower_L2', 'aActivePower_L3', 'rTotalActivePower']
        for feature in features_to_plot:
            self.plot_comparison(normal_results, anomaly_results, feature)
        
        # 5. 频段能量分析
        freq_band_summary = self.extract_frequency_bands(normal_results, anomaly_results)
        
        logger.info("="*80)
        logger.info("探索性频域分析完成！")
        logger.info(f"结果保存在: {self.output_dir}")
        logger.info("="*80)


def main():
    """主函数"""
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 配置路径
    data_path = "Data/machine/cleaning_utc/Contacting_cleaned_1.parquet"
    anomaly_path = "Data/machine/Anomaly_Data/anomaly_dict_merged.pkl"
    output_dir = f"experiments/fourier_analysis_exploration/{timestamp}"
    
    # 创建分析对象并运行
    analyzer = FourierAnalysisExploration(
        data_path=data_path,
        anomaly_path=anomaly_path,
        output_dir=output_dir
    )
    
    analyzer.run()
    
    logger.info(f"所有结果已保存至: {output_dir}")


if __name__ == "__main__":
    main()
