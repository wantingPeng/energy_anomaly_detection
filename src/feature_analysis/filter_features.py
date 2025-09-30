#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过滤窗口特征，去除与anomaly_label和id相关的列

此脚本提供函数来加载window_features.parquet数据，
并过滤掉所有与anomaly_label和id相关的列。

示例用法:
python -m src.feature_analysis.filter_features \
    --input_file experiments/window_features/window_features.parquet \
    --output_file experiments/window_features/filtered_no_id_anomaly.parquet
"""

import os
import sys
import pandas as pd
import argparse
from pathlib import Path
import time
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.logger import logger


def load_and_filter_features(input_file, exclude_prefixes=None):
    """
    加载特征数据并过滤掉指定前缀的列
    
    Args:
        input_file: 输入文件路径
        exclude_prefixes: 要排除的列前缀列表，默认排除'anomaly_label'和'id'
        
    Returns:
        pd.DataFrame: 过滤后的DataFrame
    """
    if exclude_prefixes is None:
        exclude_prefixes = ['anomaly_label', 'id']
    
    logger.info(f"从 {input_file} 加载数据")
    df = pd.read_parquet(input_file)
    
    original_shape = df.shape
    logger.info(f"原始数据形状: {original_shape}")
    
    # 识别要保留的列
    cols_to_keep = [col for col in df.columns if not any(col.startswith(prefix) for prefix in exclude_prefixes)]
    
    # 保留TimeStamp列(如果存在)
    if 'TimeStamp' in df.columns and 'TimeStamp' not in cols_to_keep:
        cols_to_keep.append('TimeStamp')
    
    # 过滤数据框
    filtered_df = df[cols_to_keep]
    
    logger.info(f"过滤后的数据形状: {filtered_df.shape}")
    logger.info(f"移除的列数量: {original_shape[1] - filtered_df.shape[1]}")
    
    # 打印被移除的列名
    removed_cols = set(df.columns) - set(filtered_df.columns)
    logger.info(f"被移除的列: {sorted(list(removed_cols))}")
    
    return filtered_df


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='过滤窗口特征数据')
    parser.add_argument('--input_file', type=str,
                        default='experiments/window_features/window_features.parquet',
                        help='输入特征文件路径')
    parser.add_argument('--output_file', type=str,
                        default='experiments/window_features/filtered_no_id_anomaly.parquet',
                        help='输出文件路径')
    parser.add_argument('--exclude_prefixes', type=str, nargs='+',
                        default=['anomaly_label', 'id'],
                        help='要排除的列前缀列表')
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"experiments/logs/filter_features_{timestamp}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    try:
        # 加载并过滤数据
        start_time = time.time()
        filtered_df = load_and_filter_features(args.input_file, args.exclude_prefixes)
        logger.info(f"特征过滤完成，耗时 {(time.time() - start_time):.2f} 秒")
        
        # 保存过滤后的数据
        output_dir = os.path.dirname(args.output_file)
        os.makedirs(output_dir, exist_ok=True)
        
        filtered_df.to_parquet(args.output_file)
        logger.info(f"过滤后的特征已保存至 {args.output_file}")
        
        logger.info("处理完成！")
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
