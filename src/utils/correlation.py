import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from src.utils.logger import logger

def load_labeled_data(file_path: str) -> pd.DataFrame:
    """
    加载已标记的数据文件
    
    Args:
        file_path: parquet文件路径
        
    Returns:
        pd.DataFrame: 加载的数据
    """
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_parquet(file_path)
        logger.info(f"Successfully loaded data with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def analyze_correlations_method1(df: pd.DataFrame) -> pd.Series:
    """
    使用第一种方法分析相关性（与原analyze_correlations函数相同）
    
    Args:
        df: 输入数据框
        
    Returns:
        pd.Series: 排序后的相关性值
    """
    try:
        # 定义要排除的列
        exclude_cols = ['window_start', 'window_end', 'segment_id', 'overlap_ratio','component_type']
        
        # 创建分析用的数据框副本
        analysis_df = df.drop(columns=exclude_cols, errors='ignore')
        
        # 计算相关性
        correlations = analysis_df.corr()['anomaly_label'].abs().drop('anomaly_label').sort_values(ascending=False)
        
        logger.info("Successfully calculated correlations using method 1")
        return correlations
    except Exception as e:
        logger.error(f"Error in analyze_correlations_method1: {str(e)}")
        raise


def compare_correlation_methods(df: pd.DataFrame, 
                              output_dir: str = "experiments/correlation_analysis") -> Tuple[pd.Series, None]:
    """
    比较两种相关性分析方法
    
    Args:
        df: 输入数据框
        output_dir: 输出目录
        
    Returns:
        Tuple[pd.Series, None]: 方法1的相关性结果和None（方法2的结果以图形方式保存）
    """
    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 方法1：analyze_correlations
        correlations_m1 = analyze_correlations_method1(df)
        
        # 保存方法1的结果
        output_path_m1 = os.path.join(output_dir, "method1_correlations_ringmontage.csv")
        correlations_m1.to_csv(output_path_m1)
        logger.info(f"Saved method 1 correlations to {output_path_m1}")
        

        return correlations_m1
        
    except Exception as e:
        logger.error(f"Error in compare_correlation_methods: {str(e)}")
        raise

if __name__ == "__main__":
    import os
    
    # 设置输入和输出路径
    input_file = "Data/interim/Energy_labeling_correlations/Ringmontage_labeled.parquet"
    output_dir = "experiments/reports/correlation_analysis4"
    
    # 加载数据
    df = load_labeled_data(input_file)
    
    # 比较两种方法
    correlations = compare_correlation_methods(df, output_dir)
    
    # 打印方法1的结果
    print("\nMethod 1 (analyze_correlations) results:")
    print(correlations.head(20))  # 显示前20个最强相关性
