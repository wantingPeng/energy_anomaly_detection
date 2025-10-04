#!/usr/bin/env python3
"""
处理异常持续时间数据脚本
从Data/row/Anomaly_Data/Duration_of_Anomalies.csv中提取Station、Date、Downtime三列
将日期规范化为 pandas 的 timezone-aware 类型: datetime64[ns, UTC]
并根据 Date 与 Downtime 计算开始时间与结束时间（均为 datetime64[ns, UTC]）
"""

import pandas as pd
from pathlib import Path
import sys

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from utils.logger import logger


def process_anomaly_duration_data():
    """
    处理异常持续时间数据的主函数
    """
    logger.info("开始处理异常持续时间数据")
    
    # 定义文件路径
    input_file = project_root / "Data" / "row" / "Anomaly_Data" / "Duration_of_Anomalies.csv"
    output_dir = project_root / "Data" / "redo_anomaly"
    
    # 确保输出目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查输入文件是否存在
    if not input_file.exists():
        logger.error(f"输入文件不存在: {input_file}")
        return
    
    logger.info(f"读取输入文件: {input_file}")
    
    try:
        # 读取CSV文件，使用分号作为分隔符
        df = pd.read_csv(input_file, sep=';', encoding='utf-8')
        logger.info(f"成功读取数据，共 {len(df)} 行")
        
        # 显示原始列名
        logger.info(f"原始列名: {list(df.columns)}")
        
        # 清理列名（去除前后空格）
        df.columns = df.columns.str.strip()
        
        # 提取需要的三列
        required_columns = ['Station', 'Date', 'Downtime']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"缺少必要的列: {missing_columns}")
            logger.info(f"可用的列: {list(df.columns)}")
            return
        
        # 提取三列数据
        df_filtered = df[required_columns].copy()
        logger.info(f"提取三列数据，共 {len(df_filtered)} 行")
        
        # 去除空值
        initial_count = len(df_filtered)
        df_filtered = df_filtered.dropna()
        logger.info(f"去除空值后，剩余 {len(df_filtered)} 行 (移除了 {initial_count - len(df_filtered)} 行)")
        
        # 标准化为 timezone-aware 的 UTC 时间戳: datetime64[ns, UTC]
        logger.info("开始标准化UTC时间为 timezone-aware 类型 (datetime64[ns, UTC])")
        df_filtered['Date'] = df_filtered['Date'].astype(str).str.strip()
        df_filtered['Date_UTC'] = pd.to_datetime(
            df_filtered['Date'],
            format="%d.%m.%Y, %H:%M:%S",
            utc=True,
            errors='coerce'
        )

        # 解析停机时长为 timedelta
        df_filtered['Downtime'] = df_filtered['Downtime'].astype(str).str.strip()
        df_filtered['Downtime_TD'] = pd.to_timedelta(df_filtered['Downtime'], errors='coerce')

        # 计算开始与结束时间（均保持为 datetime64[ns, UTC]）
        logger.info("开始计算开始时间和结束时间 (timezone-aware)")
        df_filtered['Start_Time'] = df_filtered['Date_UTC']
        df_filtered['End_Time'] = df_filtered['Start_Time'] + df_filtered['Downtime_TD']

        # 去除解析失败的行
        initial_count = len(df_filtered)
        df_filtered = df_filtered.dropna(subset=['Date_UTC', 'Start_Time', 'End_Time', 'Downtime_TD'])
        logger.info(f"去除时间解析/计算失败的行后，剩余 {len(df_filtered)} 行 (移除了 {initial_count - len(df_filtered)} 行)")
        
        # 重新排列列的顺序
        final_columns = ['Station', 'Date_UTC', 'Start_Time', 'End_Time', 'Downtime']
        df_final = df_filtered[final_columns].copy()
        
        # 显示处理结果统计
        logger.info("数据处理完成，结果统计:")
        logger.info(f"总行数: {len(df_final)}")
        logger.info(f"站点数量: {df_final['Station'].nunique()}")
        logger.info(f"时间范围: {df_final['Start_Time'].min()} 到 {df_final['End_Time'].max()}")
        logger.info(f"列类型: {df_final.dtypes.to_dict()}")
        
        # 保存为Parquet格式
        output_file = output_dir / "anomaly_duration_processed.parquet"
        df_final.to_parquet(output_file, index=False)
        logger.info(f"数据已保存到: {output_file}")
        
        # 保存为CSV格式（便于查看）
        output_csv = output_dir / "anomaly_duration_processed.csv"
        df_final.to_csv(output_csv, index=False, encoding='utf-8')
        logger.info(f"数据已保存到: {output_csv}")
        
        # 显示前几行数据作为预览
        logger.info("数据预览:")
        logger.info(f"\n{df_final.head().to_string()}")
        
        # 显示数据信息
        logger.info("数据信息:")
        try:
            import io
            buf = io.StringIO()
            df_final.info(buf=buf)
            logger.info(f"\n{buf.getvalue()}")
        except Exception:
            # 回退：不阻塞流程
            pass
        
        # 显示各站点的数据统计
        station_stats = df_final['Station'].value_counts()
        logger.info("各站点异常数量统计:")
        logger.info(f"\n{station_stats.to_string()}")
        
        logger.info("异常持续时间数据处理完成!")
        
    except Exception as e:
        logger.error(f"处理数据时发生错误: {e}")
        raise


if __name__ == "__main__":
    process_anomaly_duration_data()
