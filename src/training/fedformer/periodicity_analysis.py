import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import dask.dataframe as dd
from scipy import signal
from scipy.fft import fft, fftfreq
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
import warnings
warnings.filterwarnings('ignore')

from src.utils.logger import logger

def load_and_merge_datasets(base_path: str) -> pd.DataFrame:
    """
    加载并合并train、val、test数据集
    """
    logger.info("开始加载数据集...")
    
    base_dir = Path(base_path)
    all_dfs = []
    
    # 遍历train、val、test目录
    for split in ['train', 'val', 'test']:
        split_dir = base_dir / split / 'contact'
        
        if not split_dir.exists():
            logger.warning(f"目录不存在: {split_dir}")
            continue
            
        logger.info(f"正在加载 {split} 数据...")
        
        # 遍历每个batch目录
        for batch_dir in split_dir.iterdir():
            if batch_dir.is_dir():
                logger.info(f"  处理 {batch_dir.name}...")
                
                # 使用dask加载该batch下的所有parquet文件
                parquet_pattern = str(batch_dir / "*.parquet")
                try:
                    df_batch = dd.read_parquet(parquet_pattern).compute()
                    df_batch['dataset_split'] = split
                    df_batch['batch_id'] = batch_dir.name
                    all_dfs.append(df_batch)
                    logger.info(f"    加载了 {len(df_batch)} 条记录")
                except Exception as e:
                    logger.error(f"    加载 {batch_dir} 时出错: {e}")
    
    if not all_dfs:
        raise ValueError("没有找到任何数据文件")
    
    # 合并所有数据
    logger.info("正在合并所有数据...")
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # 确保TimeStamp列是datetime类型
    combined_df['TimeStamp'] = pd.to_datetime(combined_df['TimeStamp'])
    
    # 按时间排序
    combined_df = combined_df.sort_values('TimeStamp').reset_index(drop=True)
    
    logger.info(f"合并完成！总共 {len(combined_df)} 条记录")
    logger.info(f"时间范围: {combined_df['TimeStamp'].min()} 到 {combined_df['TimeStamp'].max()}")
    
    return combined_df

def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    提取时间特征用于周期性分析
    """
    logger.info("提取时间特征...")
    
    df = df.copy()
    
    # 基本时间特征
    df['hour'] = df['TimeStamp'].dt.hour
    df['day_of_week'] = df['TimeStamp'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['day_of_year'] = df['TimeStamp'].dt.dayofyear
    df['month'] = df['TimeStamp'].dt.month
    df['week_of_year'] = df['TimeStamp'].dt.isocalendar().week
    
    # 工作日/周末标识
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # 一天中的分钟数（用于分析日内周期）
    df['minute_of_day'] = df['TimeStamp'].dt.hour * 60 + df['TimeStamp'].dt.minute
    
    # 一周中的小时数（用于分析周周期）
    df['hour_of_week'] = df['day_of_week'] * 24 + df['hour']
    
    logger.info("时间特征提取完成")
    return df

def analyze_daily_pattern(df: pd.DataFrame, target_col: str = 'rTotalActivePower'):
    """
    分析日周期模式
    """
    logger.info("分析日周期模式...")
    
    # 按小时聚合
    hourly_pattern = df.groupby('hour')[target_col].agg(['mean', 'std', 'median']).reset_index()
    
    # 绘制日周期图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Daily Patterns Analysis', fontsize=16)
    
    # 平均功率按小时
    axes[0, 0].plot(hourly_pattern['hour'], hourly_pattern['mean'], marker='o')
    axes[0, 0].fill_between(hourly_pattern['hour'], 
                           hourly_pattern['mean'] - hourly_pattern['std'],
                           hourly_pattern['mean'] + hourly_pattern['std'], 
                           alpha=0.3)
    axes[0, 0].set_title('Average Power by Hour')
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Average Power (kW)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 工作日vs周末对比
    workday_pattern = df[df['is_weekend'] == 0].groupby('hour')[target_col].mean()
    weekend_pattern = df[df['is_weekend'] == 1].groupby('hour')[target_col].mean()
    
    axes[0, 1].plot(workday_pattern.index, workday_pattern.values, 
                   marker='o', label='Workdays', linewidth=2)
    axes[0, 1].plot(weekend_pattern.index, weekend_pattern.values, 
                   marker='s', label='Weekends', linewidth=2)
    axes[0, 1].set_title('Workdays vs Weekends')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Average Power (kW)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 热力图：小时 vs 工作日
    pivot_data = df.groupby(['day_of_week', 'hour'])[target_col].mean().unstack()
    im = axes[1, 0].imshow(pivot_data.values, cmap='YlOrRd', aspect='auto')
    axes[1, 0].set_title('Power Consumption Heatmap')
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Day of Week (0=Mon, 6=Sun)')
    axes[1, 0].set_xticks(range(0, 24, 4))
    axes[1, 0].set_xticklabels(range(0, 24, 4))
    axes[1, 0].set_yticks(range(7))
    axes[1, 0].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    plt.colorbar(im, ax=axes[1, 0])
    
    # 日内变异系数
    hourly_cv = df.groupby('hour')[target_col].apply(lambda x: x.std() / x.mean())
    axes[1, 1].bar(hourly_cv.index, hourly_cv.values)
    axes[1, 1].set_title('Coefficient of Variation by Hour')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('CV (std/mean)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/data_periode/daily_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return hourly_pattern

def analyze_weekly_pattern(df: pd.DataFrame, target_col: str = 'rTotalActivePower'):
    """
    分析周周期模式
    """
    logger.info("分析周周期模式...")
    
    # 按星期几聚合
    weekly_pattern = df.groupby('day_of_week')[target_col].agg(['mean', 'std', 'median']).reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Weekly Patterns Analysis', fontsize=16)
    
    # 平均功率按星期几
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[0, 0].bar(range(7), weekly_pattern['mean'], 
                  yerr=weekly_pattern['std'], capsize=5)
    axes[0, 0].set_title('Average Power by Day of Week')
    axes[0, 0].set_xlabel('Day of Week')
    axes[0, 0].set_ylabel('Average Power (kW)')
    axes[0, 0].set_xticks(range(7))
    axes[0, 0].set_xticklabels(day_names)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 168小时周期图（一周=168小时）
    weekly_hourly = df.groupby('hour_of_week')[target_col].mean()
    axes[0, 1].plot(weekly_hourly.index, weekly_hourly.values)
    axes[0, 1].set_title('168-Hour Weekly Cycle')
    axes[0, 1].set_xlabel('Hour of Week')
    axes[0, 1].set_ylabel('Average Power (kW)')
    
    # 添加垂直线标识每天的开始
    for day in range(1, 7):
        axes[0, 1].axvline(x=day*24, color='red', linestyle='--', alpha=0.5)
    
    # 添加星期几标签
    day_centers = [i*24 + 12 for i in range(7)]
    axes[0, 1].set_xticks(day_centers)
    axes[0, 1].set_xticklabels(day_names)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 工作日内的日均值分布
    workdays_data = df[df['is_weekend'] == 0].groupby(['day_of_week', 'hour'])[target_col].mean()
    weekend_data = df[df['is_weekend'] == 1].groupby(['day_of_week', 'hour'])[target_col].mean()
    
    # 箱线图：工作日vs周末的分布
    workday_values = df[df['is_weekend'] == 0][target_col]
    weekend_values = df[df['is_weekend'] == 1][target_col]
    
    axes[1, 0].boxplot([workday_values, weekend_values], 
                      labels=['Workdays', 'Weekends'])
    axes[1, 0].set_title('Power Distribution: Workdays vs Weekends')
    axes[1, 0].set_ylabel('Power (kW)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 周内变异分析
    daily_std = df.groupby('day_of_week')[target_col].std()
    axes[1, 1].bar(range(7), daily_std.values)
    axes[1, 1].set_title('Daily Standard Deviation')
    axes[1, 1].set_xlabel('Day of Week')
    axes[1, 1].set_ylabel('Standard Deviation (kW)')
    axes[1, 1].set_xticks(range(7))
    axes[1, 1].set_xticklabels(day_names)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/data_periode/weekly_pattern_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return weekly_pattern

def fourier_analysis(df: pd.DataFrame, target_col: str = 'rTotalActivePower'):
    """
    傅里叶变换分析频域特征
    """
    logger.info("进行傅里叶频域分析...")
    
    # 对时间序列进行傅里叶变换
    time_series = df[target_col].values
    n = len(time_series)
    
    # 计算FFT
    yf = fft(time_series)
    xf = fftfreq(n, d=1.0)  # 假设采样间隔为1分钟
    
    # 计算功率谱密度
    power_spectrum = np.abs(yf) ** 2
    
    # 只保留正频率部分
    positive_freq_mask = xf > 0
    frequencies = xf[positive_freq_mask]
    power = power_spectrum[positive_freq_mask]
    
    # 转换为周期（以小时为单位）
    periods_hours = 1 / (frequencies * 60)  # 从分钟转换为小时
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Fourier Analysis of Power Consumption', fontsize=16)
    
    # 功率谱密度图
    axes[0, 0].loglog(frequencies, power)
    axes[0, 0].set_title('Power Spectral Density')
    axes[0, 0].set_xlabel('Frequency (cycles/minute)')
    axes[0, 0].set_ylabel('Power')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 周期域图
    valid_periods = (periods_hours >= 0.5) & (periods_hours <= 7*24)  # 0.5小时到一周
    axes[0, 1].loglog(periods_hours[valid_periods], power[valid_periods])
    axes[0, 1].set_title('Power vs Period')
    axes[0, 1].set_xlabel('Period (hours)')
    axes[0, 1].set_ylabel('Power')
    axes[0, 1].axvline(x=24, color='red', linestyle='--', label='24h (daily)')
    axes[0, 1].axvline(x=168, color='blue', linestyle='--', label='168h (weekly)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 寻找显著周期
    # 找到功率谱中的峰值
    peaks, _ = signal.find_peaks(power, height=np.percentile(power, 95))
    significant_periods = periods_hours[peaks]
    significant_powers = power[peaks]
    
    # 过滤合理的周期范围
    valid_peaks = (significant_periods >= 0.5) & (significant_periods <= 7*24)
    significant_periods = significant_periods[valid_peaks]
    significant_powers = significant_powers[valid_peaks]
    
    # 按功率排序
    sorted_indices = np.argsort(significant_powers)[::-1]
    top_periods = significant_periods[sorted_indices][:10]
    top_powers = significant_powers[sorted_indices][:10]
    
    axes[1, 0].bar(range(len(top_periods)), top_powers)
    axes[1, 0].set_title('Top 10 Significant Periods')
    axes[1, 0].set_xlabel('Period Rank')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].set_xticks(range(len(top_periods)))
    axes[1, 0].set_xticklabels([f'{p:.1f}h' for p in top_periods], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 时域信号的一部分
    sample_hours = min(7*24, len(time_series))  # 显示最多一周的数据
    sample_indices = range(sample_hours)
    axes[1, 1].plot(sample_indices, time_series[:sample_hours])
    axes[1, 1].set_title(f'Time Series Sample ({sample_hours/60:.1f} hours)')
    axes[1, 1].set_xlabel('Time (minutes)')
    axes[1, 1].set_ylabel('Power (kW)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/data_periode/fourier_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出显著周期
    logger.info("发现的显著周期:")
    for i, (period, power) in enumerate(zip(top_periods, top_powers)):
        logger.info(f"  {i+1}. 周期: {period:.2f} 小时, 功率: {power:.2e}")
    
    return top_periods, top_powers

def autocorrelation_analysis(df: pd.DataFrame, target_col: str = 'rTotalActivePower', max_lags: int = 7*24*60):
    """
    自相关分析
    """
    logger.info("进行自相关分析...")
    
    time_series = df[target_col].values
    
    # 计算自相关函数 (限制最大滞后为一周)
    max_lags = min(max_lags, len(time_series) // 4)
    autocorr = acf(time_series, nlags=max_lags, fft=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Autocorrelation Analysis', fontsize=16)
    
    # 完整自相关图
    lags_hours = np.arange(len(autocorr)) / 60  # 转换为小时
    axes[0, 0].plot(lags_hours, autocorr)
    axes[0, 0].set_title('Full Autocorrelation Function')
    axes[0, 0].set_xlabel('Lag (hours)')
    axes[0, 0].set_ylabel('Autocorrelation')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 日周期范围的自相关
    daily_range = 48 * 60  # 48小时的滞后
    if len(autocorr) > daily_range:
        axes[0, 1].plot(lags_hours[:daily_range], autocorr[:daily_range])
        axes[0, 1].axvline(x=24, color='red', linestyle='--', label='24h')
        axes[0, 1].set_title('Autocorrelation (0-48 hours)')
        axes[0, 1].set_xlabel('Lag (hours)')
        axes[0, 1].set_ylabel('Autocorrelation')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    
    # 寻找自相关峰值
    peaks, _ = signal.find_peaks(autocorr[1:], height=0.1)  # 排除lag=0的峰值
    peak_lags_hours = (peaks + 1) / 60
    peak_values = autocorr[peaks + 1]
    
    # 过滤合理范围的峰值
    valid_peaks = (peak_lags_hours >= 1) & (peak_lags_hours <= 7*24)
    peak_lags_hours = peak_lags_hours[valid_peaks]
    peak_values = peak_values[valid_peaks]
    
    # 按峰值大小排序
    sorted_indices = np.argsort(peak_values)[::-1]
    top_peak_lags = peak_lags_hours[sorted_indices][:10]
    top_peak_values = peak_values[sorted_indices][:10]
    
    axes[1, 0].bar(range(len(top_peak_lags)), top_peak_values)
    axes[1, 0].set_title('Top Autocorrelation Peaks')
    axes[1, 0].set_xlabel('Peak Rank')
    axes[1, 0].set_ylabel('Autocorrelation')
    axes[1, 0].set_xticks(range(len(top_peak_lags)))
    axes[1, 0].set_xticklabels([f'{lag:.1f}h' for lag in top_peak_lags], rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 周周期范围的自相关详图
    weekly_range = min(10*24*60, len(autocorr))  # 10天的滞后
    weekly_step = 60  # 每小时一个点
    weekly_lags = lags_hours[:weekly_range:weekly_step]
    weekly_autocorr = autocorr[:weekly_range:weekly_step]
    
    axes[1, 1].plot(weekly_lags, weekly_autocorr)
    axes[1, 1].axvline(x=24, color='red', linestyle='--', alpha=0.7, label='24h')
    axes[1, 1].axvline(x=168, color='blue', linestyle='--', alpha=0.7, label='168h')
    axes[1, 1].set_title('Weekly Range Autocorrelation')
    axes[1, 1].set_xlabel('Lag (hours)')
    axes[1, 1].set_ylabel('Autocorrelation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiments/data_periode/autocorrelation_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 输出显著的自相关峰值
    logger.info("发现的显著自相关峰值:")
    for i, (lag, corr) in enumerate(zip(top_peak_lags, top_peak_values)):
        logger.info(f"  {i+1}. 滞后: {lag:.2f} 小时, 自相关: {corr:.3f}")
    
    return top_peak_lags, top_peak_values

def seasonal_decomposition_analysis(df: pd.DataFrame, target_col: str = 'rTotalActivePower'):
    """
    季节性分解分析
    """
    logger.info("进行季节性分解分析...")
    
    # 创建时间序列
    ts = df.set_index('TimeStamp')[target_col]
    
    # 确保时间序列是等间隔的
    ts = ts.asfreq('1T', method='ffill')  # 1分钟间隔，前向填充缺失值
    
    # 季节性分解 (使用日周期=1440分钟)
    try:
        decomposition = seasonal_decompose(ts, model='additive', period=1440)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('Seasonal Decomposition (Daily Period)', fontsize=16)
        
        # 原始时间序列
        axes[0].plot(decomposition.observed.index, decomposition.observed.values)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Power (kW)')
        axes[0].grid(True, alpha=0.3)
        
        # 趋势
        axes[1].plot(decomposition.trend.index, decomposition.trend.values)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Power (kW)')
        axes[1].grid(True, alpha=0.3)
        
        # 季节性
        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values)
        axes[2].set_title('Seasonal Component (Daily)')
        axes[2].set_ylabel('Power (kW)')
        axes[2].grid(True, alpha=0.3)
        
        # 残差
        axes[3].plot(decomposition.resid.index, decomposition.resid.values)
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Power (kW)')
        axes[3].set_xlabel('Time')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiments/data_periode/seasonal_decomposition_daily.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 计算各组件的方差贡献
        total_var = decomposition.observed.var()
        trend_var = decomposition.trend.var()
        seasonal_var = decomposition.seasonal.var()
        resid_var = decomposition.resid.var()
        
        logger.info("方差分解结果:")
        logger.info(f"  趋势组件方差占比: {trend_var/total_var*100:.2f}%")
        logger.info(f"  季节性组件方差占比: {seasonal_var/total_var*100:.2f}%")
        logger.info(f"  残差组件方差占比: {resid_var/total_var*100:.2f}%")
        
    except Exception as e:
        logger.error(f"季节性分解失败: {e}")
        
    # 尝试周周期分解
    try:
        weekly_decomposition = seasonal_decompose(ts, model='additive', period=7*1440)
        
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle('Seasonal Decomposition (Weekly Period)', fontsize=16)
        
        axes[0].plot(weekly_decomposition.observed.index, weekly_decomposition.observed.values)
        axes[0].set_title('Original Time Series')
        axes[0].set_ylabel('Power (kW)')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(weekly_decomposition.trend.index, weekly_decomposition.trend.values)
        axes[1].set_title('Trend Component')
        axes[1].set_ylabel('Power (kW)')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(weekly_decomposition.seasonal.index, weekly_decomposition.seasonal.values)
        axes[2].set_title('Seasonal Component (Weekly)')
        axes[2].set_ylabel('Power (kW)')
        axes[2].grid(True, alpha=0.3)
        
        axes[3].plot(weekly_decomposition.resid.index, weekly_decomposition.resid.values)
        axes[3].set_title('Residual Component')
        axes[3].set_ylabel('Power (kW)')
        axes[3].set_xlabel('Time')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('experiments/data_periode/seasonal_decomposition_weekly.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        logger.error(f"周周期分解失败: {e}")

def generate_periodicity_report(df: pd.DataFrame, target_col: str = 'rTotalActivePower'):
    """
    生成周期性分析报告
    """
    logger.info("生成周期性分析综合报告...")
    
    report = []
    report.append("=" * 60)
    report.append("能源数据周期性分析报告")
    report.append("=" * 60)
    report.append(f"分析时间: {pd.Timestamp.now()}")
    report.append(f"数据范围: {df['TimeStamp'].min()} 到 {df['TimeStamp'].max()}")
    report.append(f"数据点数: {len(df):,}")
    report.append(f"分析特征: {target_col}")
    report.append("")
    
    # 基本统计信息
    report.append("基本统计信息:")
    report.append(f"  平均功率: {df[target_col].mean():.2f} kW")
    report.append(f"  标准差: {df[target_col].std():.2f} kW")
    report.append(f"  最小值: {df[target_col].min():.2f} kW")
    report.append(f"  最大值: {df[target_col].max():.2f} kW")
    report.append(f"  变异系数: {df[target_col].std()/df[target_col].mean():.3f}")
    report.append("")
    
    # 日周期分析
    report.append("日周期特征:")
    hourly_stats = df.groupby('hour')[target_col].agg(['mean', 'std'])
    peak_hour = hourly_stats['mean'].idxmax()
    low_hour = hourly_stats['mean'].idxmin()
    report.append(f"  用电高峰时段: {peak_hour}:00 (平均 {hourly_stats.loc[peak_hour, 'mean']:.2f} kW)")
    report.append(f"  用电低谷时段: {low_hour}:00 (平均 {hourly_stats.loc[low_hour, 'mean']:.2f} kW)")
    report.append(f"  峰谷差: {hourly_stats['mean'].max() - hourly_stats['mean'].min():.2f} kW")
    report.append("")
    
    # 周周期分析
    report.append("周周期特征:")
    daily_stats = df.groupby('day_of_week')[target_col].agg(['mean', 'std'])
    day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    peak_day = daily_stats['mean'].idxmax()
    low_day = daily_stats['mean'].idxmin()
    report.append(f"  用电最高日: {day_names[peak_day]} (平均 {daily_stats.loc[peak_day, 'mean']:.2f} kW)")
    report.append(f"  用电最低日: {day_names[low_day]} (平均 {daily_stats.loc[low_day, 'mean']:.2f} kW)")
    
    workday_mean = df[df['is_weekend'] == 0][target_col].mean()
    weekend_mean = df[df['is_weekend'] == 1][target_col].mean()
    report.append(f"  工作日平均: {workday_mean:.2f} kW")
    report.append(f"  周末平均: {weekend_mean:.2f} kW")
    report.append(f"  工作日/周末比: {workday_mean/weekend_mean:.3f}")
    report.append("")
    
    # 保存报告
    report_text = "\n".join(report)
    with open('experiments/data_periode/periodicity_analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    logger.info("周期性分析报告已保存至: experiments/data_periode/periodicity_analysis_report.txt")
    print(report_text)
    
    return report_text

def main():
    """
    主函数：执行完整的周期性分析流程
    """
    logger.info("开始能源数据周期性分析...")
    
    # 数据路径
    base_path = "Data/row_energyData_subsample_Transform/downsampled_1min"
    target_column = "rTotalActivePower"
    
    try:
        # 1. 加载和合并数据
        df = load_and_merge_datasets(base_path)
        
        # 2. 提取时间特征
        df = extract_time_features(df)
        
        # 3. 创建输出目录
        Path("experiments/data").mkdir(parents=True, exist_ok=True)
        Path("experiments/data_periode").mkdir(parents=True, exist_ok=True)
        
        # 4. 保存合并后的数据
        output_path = "experiments/data/combined_energy_data.parquet"
        df.to_parquet(output_path)
        logger.info(f"合并数据已保存至: {output_path}")
        
        # 5. 进行各种周期性分析
        logger.info("开始多维度周期性分析...")
        
        # 日周期分析
        daily_pattern = analyze_daily_pattern(df, target_column)
        
        # 周周期分析  
        weekly_pattern = analyze_weekly_pattern(df, target_column)
        
        # 傅里叶频域分析
        fourier_periods, fourier_powers = fourier_analysis(df, target_column)
        
        # 自相关分析
        autocorr_lags, autocorr_values = autocorrelation_analysis(df, target_column)
        
        # 季节性分解
        seasonal_decomposition_analysis(df, target_column)
        
        # 生成综合报告
        generate_periodicity_report(df, target_column)
        
        logger.info("周期性分析完成！结果已保存至 experiments/data_periode/ 目录")
        
    except Exception as e:
        logger.error(f"分析过程中出错: {e}")
        raise

if __name__ == "__main__":
    main() 