#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import shutil

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.logger import logger

def setup_logger(script_name):
    """
    配置脚本的日志记录器。
    
    Args:
        script_name (str): 脚本名称
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("experiments/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}_{timestamp}.log"
    
    # 在utils/logger.py中已设置logger，只需更新文件处理程序
    for handler in logger.handlers:
        if isinstance(handler, type(logger.handlers[0])):  # FileHandler类型
            handler.close()
            logger.handlers.remove(handler)
    
    # 添加新的文件处理程序
    import logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"Logger已配置。日志文件: {log_file}")
    return timestamp

def filter_data_by_top_features(
    data_dir="Data/row_energyData_subsample_xgboost/labeled/contact",
    feature_importance_path="experiments/xgboost_anomaly_detection/xgboost_20250706_142250/model_20250706_142251/feature_importance_gain.csv",
    output_dir="Data/pc/statistic_100",
    top_n=40
):
    """
    加载XGBoost特征重要性结果，筛选前N个重要特征，并过滤数据集。
    
    Args:
        data_dir (str): 包含数据文件的目录路径
        feature_importance_path (str): 特征重要性CSV文件路径
        output_dir (str): 输出目录路径
        top_n (int): 要筛选的特征数量
        
    Returns:
        dict: 包含处理结果信息的字典
    """
    # 设置日志
    timestamp = setup_logger("statisticFeatureFromxgboostResult")
    start_time = time.time()
    
    logger.info(f"开始处理。提取前{top_n}个重要特征并过滤数据")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"特征重要性文件: {feature_importance_path}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载特征重要性CSV
        logger.info("加载特征重要性文件...")
        importance_df = pd.read_csv(feature_importance_path)
        
        if 'Feature' not in importance_df.columns or 'Importance' not in importance_df.columns:
            column_names = list(importance_df.columns)
            logger.error(f"特征重要性文件格式不正确。列名: {column_names}")
            raise ValueError(f"特征重要性文件缺少'Feature'或'Importance'列。找到的列: {column_names}")
        
        # 直接取前top_n个特征（假设已排序）
        top_features = importance_df.head(top_n)['Feature'].tolist()
        
        logger.info(f"成功提取前{len(top_features)}个重要特征")
        
        # 保存特征列表到输出目录
        with open(os.path.join(output_dir, "top_features.txt"), "w") as f:
            for feature in top_features:
                f.write(f"{feature}\n")
        
        # 处理每个数据文件
        data_files = ["train.parquet"]
        
        for file in data_files:
            file_path = os.path.join(data_dir, file)
            if not os.path.exists(file_path):
                logger.warning(f"文件不存在，跳过: {file_path}")
                continue
                
            logger.info(f"处理文件: {file}")
            
            # 加载数据
            df = pd.read_parquet(file_path)
            logger.info(f"原始数据形状: {df.shape}")
            
            # 确保目标列(通常为anomaly_label)也被保留
            target_column = "anomaly_label" if "anomaly_label" in df.columns else None
            if target_column:
                if target_column not in top_features:
                    filtered_columns = top_features + [target_column]
                else:
                    filtered_columns = top_features
            else:
                filtered_columns = top_features
                
            # 筛选列
            # 首先检查哪些特征实际存在于数据集中
            existing_features = [col for col in filtered_columns if col in df.columns]
            missing_features = [col for col in filtered_columns if col not in df.columns]
            
            if missing_features:
                logger.warning(f"数据中缺少{len(missing_features)}个特征: {', '.join(missing_features[:5])}...")
            
            # 筛选数据
            filtered_df = df[existing_features].copy()
            logger.info(f"筛选后数据形状: {filtered_df.shape}")
            
            # 保存筛选后的数据
            output_file = os.path.join(output_dir, file)
            filtered_df.to_parquet(output_file, index=False)
            logger.info(f"筛选后的数据已保存到: {output_file}")
        
        # 记录处理时间
        processing_time = time.time() - start_time
        logger.info(f"处理完成。总用时: {processing_time:.2f}秒")
        
        # 创建并保存处理记录
        record = {
            "timestamp": timestamp,
            "data_directory": data_dir,
            "feature_importance_file": feature_importance_path,
            "output_directory": output_dir,
            "top_n_features": top_n,
            "processing_time_seconds": processing_time,
            "top_features": top_features
        }
        
        # 将处理记录保存为MD文件
        with open(os.path.join(output_dir, "record.md"), "w") as f:
            f.write(f"# 数据处理记录\n\n")
            f.write(f"- **处理时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **数据目录**: {data_dir}\n")
            f.write(f"- **特征重要性文件**: {feature_importance_path}\n")
            f.write(f"- **输出目录**: {output_dir}\n")
            f.write(f"- **筛选特征数量**: {top_n}\n")
            f.write(f"- **处理用时**: {processing_time:.2f}秒\n\n")
            f.write(f"## 前{min(10, len(top_features))}个重要特征\n\n")
            for i, feature in enumerate(top_features[:10], 1):
                f.write(f"{i}. {feature}\n")
                
        return record
        
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # 直接定义参数，不通过命令行传入
    data_dir = "Data/row_energyData_subsample_xgboost/labeled/contact"
    feature_importance_path = "experiments/xgboost_anomaly_detection/xgboost_20250706_142250/model_20250706_142251/feature_importance_gain.csv"
    output_dir = "Data/pc/statistic_40"
    top_n = 40
    
    # 调用函数处理数据
    filter_data_by_top_features(
        data_dir=data_dir,
        feature_importance_path=feature_importance_path,
        output_dir=output_dir,
        top_n=top_n
    )
