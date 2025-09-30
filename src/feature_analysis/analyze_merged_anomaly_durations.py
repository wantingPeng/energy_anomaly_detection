#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析三个数据集（Kontaktieren, Pcb, Ringmontage）的异常持续时间分布

此脚本从anomaly_dict_merged.pkl中加载异常数据，计算并可视化三个数据集的异常持续时间分布。
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

# Add the project root to the path so we can import the utils module
project_root = str(Path(__file__).parent.parent.parent)
sys.path.append(project_root)

from utils.logger import logger

def load_anomaly_dict(file_path):
    """
    从pickle文件加载异常数据
    
    Args:
        file_path: anomaly_dict_merged.pkl文件路径
        
    Returns:
        dict: 包含三个数据集异常时间范围的字典
    """
    try:
        with open(file_path, 'rb') as f:
            anomaly_dict = pickle.load(f)
        
        logger.info(f"成功加载异常数据，包含 {len(anomaly_dict)} 个站点")
        return anomaly_dict
    except Exception as e:
        logger.error(f"加载异常数据时出错: {str(e)}")
        raise

def calculate_durations(anomaly_intervals):
    """
    计算异常持续时间（分钟）
    
    Args:
        anomaly_intervals: 异常时间区间列表，每个元素是(开始时间, 结束时间)
        
    Returns:
        list: 包含所有异常持续时间的列表（单位：分钟）
    """
    durations = []
    for start, end in anomaly_intervals:
        duration = (end - start).total_seconds() / 60  # 转换为分钟
        durations.append(duration)
    
    return durations

def analyze_station_durations(durations, station_name, output_dir):
    """
    分析单个站点的异常持续时间分布
    
    Args:
        durations: 异常持续时间列表（分钟）
        station_name: 站点名称
        output_dir: 输出目录
    """
    if not durations:
        logger.warning(f"{station_name}: 未发现异常事件")
        return
    
    # 计算基本统计信息
    stats = {
        'count': len(durations),
        'mean': np.mean(durations),
        'std': np.std(durations),
        'min': np.min(durations),
        'q1': np.percentile(durations, 25),
        'median': np.median(durations),
        'q3': np.percentile(durations, 75),
        'max': np.max(durations),
    }
    
    # 创建输出目录
    station_output_dir = os.path.join(output_dir, station_name)
    os.makedirs(station_output_dir, exist_ok=True)
    
    # 保存基本统计信息
    stats_df = pd.DataFrame([stats])
    stats_path = os.path.join(station_output_dir, f'{station_name}_duration_stats.csv')
    stats_df.to_csv(stats_path, index=False)
    logger.info(f"{station_name}: 基本统计信息已保存到 {stats_path}")
    
    # 时长分布分箱
    max_duration = np.max(durations)
    
    # 特殊情况：如果只有一个数据点或所有数据点相同
    if np.min(durations) == np.max(durations):
        logger.info(f"{station_name}: 所有持续时间都相同: {max_duration}")
        bins = [0, max_duration + 0.1]
        labels = [f"{max_duration}"]
        
        # 不使用pd.cut，直接创建分布
        duration_distribution = pd.Series([len(durations)], index=labels)
        logger.info(f"{station_name}: 使用单一区间: {bins}")
        logger.info(f"{station_name}: 对应标签: {labels}")
    else:
        # 正常情况：构建标准阈值
        standard_thresholds = [0, 5, 10, 15, 30, 60, 120, 240, 480]
        
        # 过滤掉大于max_duration的阈值
        filtered_thresholds = [t for t in standard_thresholds if t <= max_duration]
        
        # 确保bins包含0
        if 0 not in filtered_thresholds:
            filtered_thresholds.insert(0, 0)
            
        # 添加最大值+1作为最后一个bin
        bins = filtered_thresholds + [max_duration + 1]
        
        # 构建labels
        labels = []
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:  # 最后一个区间
                labels.append(f"{bins[i]}+")
            else:
                labels.append(f"{bins[i]}-{bins[i+1]}")
        
        logger.info(f"{station_name}: 使用以下区间: {bins}")
        logger.info(f"{station_name}: 对应标签: {labels}")
        
        # 使用pd.cut进行分箱
        duration_series = pd.Series(durations)
        duration_binned = pd.cut(duration_series, bins=bins, labels=labels)
        duration_distribution = duration_binned.value_counts().sort_index()
    
    # 保存分布数据
    distribution_df = pd.DataFrame({'分钟区间': duration_distribution.index, '事件数量': duration_distribution.values})
    distribution_path = os.path.join(station_output_dir, f'{station_name}_duration_distribution.csv')
    distribution_df.to_csv(distribution_path, index=False)
    logger.info(f"{station_name}: 分布数据已保存到 {distribution_path}")
    
    # 设置绘图样式
    plt.style.use('ggplot')
    plt.figure(figsize=(12, 8))
    
    # 1. 持续时间分布条形图
    plt.subplot(2, 2, 1)
    sns.barplot(x=duration_distribution.index, y=duration_distribution.values)
    plt.title(f'{station_name} 异常持续时间分布', fontsize=14)
    plt.xlabel('持续时间 (分钟)', fontsize=12)
    plt.ylabel('事件数量', fontsize=12)
    plt.xticks(rotation=45)
    
    # 2. 持续时间箱线图
    plt.subplot(2, 2, 2)
    sns.boxplot(x=durations)
    plt.title(f'{station_name} 异常持续时间箱线图', fontsize=14)
    plt.xlabel('持续时间 (分钟)', fontsize=12)
    plt.xlim(0, np.percentile(durations, 95))  # 限制x轴范围，避免异常值影响可视化
    
    # 3. 持续时间直方图
    plt.subplot(2, 2, 3)
    sns.histplot(durations, bins=30, kde=True)
    plt.title(f'{station_name} 异常持续时间直方图', fontsize=14)
    plt.xlabel('持续时间 (分钟)', fontsize=12)
    plt.ylabel('频率', fontsize=12)
    plt.xlim(0, np.percentile(durations, 95))  # 限制x轴范围，避免异常值影响可视化
    
    # 4. 持续时间累积分布函数
    plt.subplot(2, 2, 4)
    durations_sorted = np.sort(durations)
    p = 1. * np.arange(len(durations)) / (len(durations) - 1)
    plt.plot(durations_sorted, p)
    plt.title(f'{station_name} 异常持续时间累积分布', fontsize=14)
    plt.xlabel('持续时间 (分钟)', fontsize=12)
    plt.ylabel('累积概率', fontsize=12)
    plt.xlim(0, np.percentile(durations, 95))  # 限制x轴范围，避免异常值影响可视化
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(station_output_dir, f'{station_name}_duration_plots.png')
    plt.savefig(plot_path, dpi=300)
    logger.info(f"{station_name}: 可视化图表已保存到 {plot_path}")
    
    # 关闭图表
    plt.close()
    
    # 打印摘要信息
    logger.info(f"{station_name}: 异常事件总数: {stats['count']}")
    logger.info(f"{station_name}: 平均持续时间: {stats['mean']:.2f} 分钟")
    logger.info(f"{station_name}: 最短持续时间: {stats['min']:.2f} 分钟")
    logger.info(f"{station_name}: 最长持续时间: {stats['max']:.2f} 分钟")
    logger.info(f"{station_name}: 中位数持续时间: {stats['median']:.2f} 分钟")
    
    return stats, duration_distribution

def main():
    """主函数"""
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/merged_anomaly_duration_distribution_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 设置输入和输出路径
    anomaly_dict_path = '/home/wanting/energy_anomaly_detection/Data/machine/Anomaly_Data/anomaly_dict_merged.pkl'
    
    # 检查文件是否存在，如果不存在则尝试使用另一个文件名
    if not os.path.exists(anomaly_dict_path):
        logger.warning(f"文件不存在: {anomaly_dict_path}")
        alternative_path = '/home/wanting/energy_anomaly_detection/Data/machine/Anomaly_Data/anomaly_dict_mergend.pkl'
        if os.path.exists(alternative_path):
            logger.info(f"使用替代文件: {alternative_path}")
            anomaly_dict_path = alternative_path
        else:
            logger.error(f"找不到文件: {anomaly_dict_path} 或 {alternative_path}")
    
    output_dir = '/home/wanting/energy_anomaly_detection/experiments/reports/merged_anomaly_distribution'
    
    logger.info("开始分析三个站点的异常持续时间分布")
    logger.info(f"异常数据文件路径: {anomaly_dict_path}")
    logger.info(f"输出目录: {output_dir}")
    
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载异常数据
        logger.info("加载异常数据...")
        anomaly_dict = load_anomaly_dict(anomaly_dict_path)
        
        # 确认站点名称
        stations = list(anomaly_dict.keys())
        logger.info(f"分析的站点: {stations}")
        
        # 保存每个站点的统计信息和分布数据
        all_station_stats = {}
        all_station_distributions = {}
        
        # 分别分析每个站点的异常持续时间分布
        for station in stations:
            logger.info(f"分析 {station} 站点的异常持续时间分布...")
            
            # 计算异常持续时间
            anomaly_intervals = anomaly_dict[station]
            durations = calculate_durations(anomaly_intervals)
            
            # 分析并保存结果
            stats, distribution = analyze_station_durations(durations, station, output_dir)
            
            # 保存用于比较分析的数据
            all_station_stats[station] = {
                'count': stats['count'],
                'mean': stats['mean'],
                'median': stats['median'],
                'min': stats['min'],
                'max': stats['max'],
                'std': stats['std'],
                'durations': durations  # 保存原始持续时间数据用于箱线图
            }
            all_station_distributions[station] = distribution
      
        
        logger.info("分析完成")
        
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
