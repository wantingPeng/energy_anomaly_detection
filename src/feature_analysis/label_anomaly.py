#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
异常标签生成和数据统计

此脚本从原始数据和异常时间字典中生成异常标签，并统计数据分布情况。

示例用法:
python -m src.feature_analysis.label_anomaly \
    --input_file Data/filtered_feature_s/top_features_Contacting_cleaned_1.parquet \
    --anomaly_dict Data/machine/Anomaly_Data/anomaly_dict_no_merged.pkl \
    --station Kontaktieren \
    --output_file Data/filtered_feature_s/top_features_Contacting_cleaned_1_labeled.parquet
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.logger import logger


def load_anomaly_dict(pkl_path: str) -> Dict[str, List[Tuple[datetime, datetime]]]:
    """
    加载异常时间字典
    
    Args:
        pkl_path: pkl文件路径
        
    Returns:
        异常字典，格式为 {station_name: [(start_time, end_time), ...]}
    """
    logger.info(f"从 {pkl_path} 加载异常时间字典")
    
    with open(pkl_path, 'rb') as f:
        anomaly_dict = pickle.load(f)
    
    logger.info(f"成功加载异常字典，包含 {len(anomaly_dict)} 个工作站")
    for station, periods in anomaly_dict.items():
        logger.info(f"  - {station}: {len(periods)} 个异常时间段")
    
    return anomaly_dict


def label_anomalies(df: pd.DataFrame, 
                    anomaly_periods: List[Tuple[datetime, datetime]],
                    timestamp_col: str = 'TimeStamp') -> pd.DataFrame:
    """
    为数据集添加异常标签
    
    Args:
        df: 原始数据DataFrame，必须包含TimeStamp列
        anomaly_periods: 异常时间段列表
        timestamp_col: 时间戳列名
        
    Returns:
        添加了anomaly_label列的DataFrame
    """
    logger.info("开始为数据打标签...")
    
    # 确保TimeStamp是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        logger.info(f"将 {timestamp_col} 转换为datetime类型")
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    
    # 创建标签列，默认为0（正常）
    df['anomaly_label'] = 0
    
    # 遍历每个异常时间段，标记落在其中的数据点
    logger.info(f"处理 {len(anomaly_periods)} 个异常时间段")
    total_anomalies = 0
    
    for idx, (start_time, end_time) in enumerate(anomaly_periods):
        # 找到落在当前异常时间段内的数据点
        mask = (df[timestamp_col] >= start_time) & (df[timestamp_col] <= end_time)
        num_anomalies = mask.sum()
        
        if num_anomalies > 0:
            df.loc[mask, 'anomaly_label'] = 1
            total_anomalies += num_anomalies
            
        if (idx + 1) % 100 == 0:
            logger.info(f"  已处理 {idx + 1}/{len(anomaly_periods)} 个异常时间段")
    
    logger.info(f"标签添加完成，共标记 {total_anomalies} 个异常数据点")
    
    return df


def generate_statistics(df: pd.DataFrame, station_name: str) -> dict:
    """
    生成数据统计信息
    
    Args:
        df: 包含anomaly_label列的DataFrame
        station_name: 工作站名称
        
    Returns:
        统计信息字典
    """
    total_samples = len(df)
    anomaly_samples = (df['anomaly_label'] == 1).sum()
    normal_samples = (df['anomaly_label'] == 0).sum()
    anomaly_ratio = (anomaly_samples / total_samples * 100) if total_samples > 0 else 0
    
    # 时间范围
    time_range_start = df['TimeStamp'].min()
    time_range_end = df['TimeStamp'].max()
    time_span_days = (time_range_end - time_range_start).total_seconds() / 86400
    
    stats = {
        'station': station_name,
        'total_samples': total_samples,
        'normal_samples': normal_samples,
        'anomaly_samples': anomaly_samples,
        'anomaly_ratio': anomaly_ratio,
        'time_range_start': time_range_start,
        'time_range_end': time_range_end,
        'time_span_days': time_span_days,
    }
    
    return stats


def print_statistics(stats: dict):
    """
    打印统计信息
    
    Args:
        stats: 统计信息字典
    """
    logger.info("\n" + "="*60)
    logger.info(f"工作站数据统计 - {stats['station']}")
    logger.info("="*60)
    logger.info(f"总样本数:        {stats['total_samples']:,}")
    logger.info(f"正常样本数:      {stats['normal_samples']:,} ({100 - stats['anomaly_ratio']:.2f}%)")
    logger.info(f"异常样本数:      {stats['anomaly_samples']:,} ({stats['anomaly_ratio']:.2f}%)")
    logger.info(f"异常比例:        {stats['anomaly_ratio']:.4f}%")
    logger.info(f"时间范围:        {stats['time_range_start']} 至 {stats['time_range_end']}")
    logger.info(f"时间跨度:        {stats['time_span_days']:.2f} 天")
    logger.info("="*60 + "\n")


def save_labeled_data(df: pd.DataFrame, output_path: str):
    """
    保存打好标签的数据
    
    Args:
        df: 包含anomaly_label列的DataFrame
        output_path: 输出文件路径
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    logger.info(f"保存标注数据到 {output_path}")
    df.to_parquet(output_path, index=False)
    logger.info(f"数据保存完成，文件大小: {os.path.getsize(output_path) / (1024**3):.2f} GB")


def label_dataset(input_file: str,
                  anomaly_dict_file: str, 
                  station_name: str,
                  output_file: str = None) -> pd.DataFrame:
    """
    主函数：加载数据、打标签、统计并保存
    
    Args:
        input_file: 输入数据文件路径 (parquet)
        anomaly_dict_file: 异常字典文件路径 (pkl)
        station_name: 工作站名称（必须与anomaly_dict中的键匹配）
        output_file: 输出文件路径，如果为None则不保存
        
    Returns:
        打好标签的DataFrame
    """
    # 1. 加载数据
    logger.info(f"从 {input_file} 加载数据")
    df = pd.read_parquet(input_file)
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"列名: {df.columns.tolist()}")
    
    # 2. 加载异常字典
    anomaly_dict = load_anomaly_dict(anomaly_dict_file)
    
    # 3. 检查工作站名称
    if station_name not in anomaly_dict:
        logger.error(f"工作站 '{station_name}' 不在异常字典中")
        logger.error(f"可用的工作站: {list(anomaly_dict.keys())}")
        raise ValueError(f"Invalid station name: {station_name}")
    
    anomaly_periods = anomaly_dict[station_name]
    logger.info(f"工作站 {station_name} 有 {len(anomaly_periods)} 个异常时间段")
    
    # 4. 打标签
    df_labeled = label_anomalies(df, anomaly_periods)
    
    # 5. 生成统计信息
    stats = generate_statistics(df_labeled, station_name)
    print_statistics(stats)
    
    # 6. 保存数据
    if output_file:
        save_labeled_data(df_labeled, output_file)
    
    return df_labeled


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='为数据集添加异常标签')
    parser.add_argument('--input_file', type=str,
                        default='Data/filtered_feature_s/top_features_Energy_Data_ring_cleaned_1.parquet',
                        help='输入数据文件路径')
    parser.add_argument('--anomaly_dict', type=str,
                        default='Data/machine/Anomaly_Data/anomaly_dict_merged.pkl',
                        help='异常字典文件路径')
    parser.add_argument('--station', type=str,
                        default='Ringmontage',
                        choices=['Kontaktieren', 'Pcb', 'Ringmontage'],
                        help='工作站名称')
    parser.add_argument('--output_file', type=str,
                        default=None,
                        help='输出文件路径（可选）')
    return parser.parse_args()


def main():
    """主程序"""
    args = parse_args()
    
    try:
        # 如果没有指定输出文件，自动生成
        if args.output_file is None:
            base_name = os.path.splitext(args.input_file)[0]
            args.output_file = f"{base_name}_labeled.parquet"
        
        # 执行标签生成
        df_labeled = label_dataset(
            input_file=args.input_file,
            anomaly_dict_file=args.anomaly_dict,
            station_name=args.station,
            output_file=args.output_file
        )
        
        logger.info("处理完成！")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

