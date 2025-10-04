#!/usr/bin/env python3
"""
从 Data/redo_anomaly/anomaly_duration_processed.csv 构建可快速查表的异常区间字典，并保存为 .pkl。

输出字典结构（兼容现有使用方式）：
    { station_name: [(start_time, end_time), ...] }

时间类型均为 pandas 的 timezone-aware 时间戳：datetime64[ns, UTC]
"""

import argparse
import pickle
from pathlib import Path
import sys
import pandas as pd

# 项目根目录
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from utils.logger import logger

'''
def merge_overlapping_periods(periods):
    """
    合并重叠区间。
    Args:
        periods (list[tuple[pd.Timestamp, pd.Timestamp]]): 已按开始时间排序的区间列表
    Returns:
        list[tuple[pd.Timestamp, pd.Timestamp]]: 合并后的不重叠区间
    """
    if not periods:
        return []

    merged = []
    current_start, current_end = periods[0]

    for start, end in periods[1:]:
        if start <= current_end:
            # 合并区间
            if end > current_end:
                current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end

    merged.append((current_start, current_end))
    return merged
'''

def build_anomaly_lookup(input_csv: Path, output_pkl: Path, merge_overlaps: bool = False) -> dict:
    """
    从处理后的CSV构建异常查表字典。

    Args:
        input_csv (Path): 输入CSV路径（包含 Station, Date_UTC, Start_Time, End_Time, Downtime）
        output_pkl (Path): 输出pkl路径
        merge_overlaps (bool): 是否合并重叠区间

    Returns:
        dict: { station: [(start, end), ...] }
    """
    logger.info(f"读取CSV: {input_csv}")
    if not input_csv.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_csv}")

    # 读取并解析时间列为 timezone-aware
    df = pd.read_csv(input_csv)
    expected_cols = {"Station", "Date_UTC", "Start_Time", "End_Time", "Downtime"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"CSV缺少必要列: {missing}")

    # 标准化列名空格
    df.columns = df.columns.str.strip()
    df["Station"] = df["Station"].astype(str).str.strip()

    # 时间列转为 timezone-aware
    for col in ["Date_UTC", "Start_Time", "End_Time"]:
        df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    before_drop = len(df)
    df = df.dropna(subset=["Station", "Start_Time", "End_Time"])  # 确保关键列有效
    logger.info(f"有效行数: {len(df)} (移除 {before_drop - len(df)})")

    # 构建字典
    anomaly_dict: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    logger.info("按站点构建异常区间列表")

    for station, group in df.groupby("Station"):
        # 排序并转为列表
        group_sorted = group.sort_values("Start_Time")
        periods = list(zip(group_sorted["Start_Time"].tolist(), group_sorted["End_Time"].tolist()))

        '''# 可选：合并重叠区间
        if merge_overlaps:
            periods = merge_overlapping_periods(periods)
'''
        anomaly_dict[station] = periods
        logger.info(f"{station}: 区间数 {len(periods)}")

    # 确保输出目录存在
    output_pkl.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"保存pkl到: {output_pkl}")
    with open(output_pkl, "wb") as f:
        pickle.dump(anomaly_dict, f)

    logger.info(f"完成，站点数: {len(anomaly_dict)}")
    # 预览一个站点
    try:
        any_station = next(iter(anomaly_dict))
        preview_n = min(3, len(anomaly_dict[any_station]))
        logger.info(f"示例站点: {any_station}, 前{preview_n}个区间: {anomaly_dict[any_station][:preview_n]}")
    except StopIteration:
        logger.warning("无可用站点构建区间")

    return anomaly_dict


def parse_args():
    parser = argparse.ArgumentParser(description="构建异常查表字典(.pkl)")
    parser.add_argument(
        "--input",
        type=str,
        default=str(project_root / "Data" / "redo_anomaly" / "anomaly_duration_processed.csv"),
        help="输入CSV路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(project_root / "Data" / "redo_anomaly" / "anomaly_dict_redo.pkl"),
        help="输出pkl路径"
    )
    parser.add_argument(
        "--no-merge-overlaps",
        action="store_true",
        help="不合并重叠区间"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_csv = Path(args.input)
    output_pkl = Path(args.output)
    #merge_overlaps = not args.no-merge-overlaps if hasattr(args, "no-merge-overlaps") else True

    # argparse 将 '-' 转换为 '_'，修正上行逻辑
    #merge_overlaps = not getattr(args, "no_merge_overlaps", False)

    build_anomaly_lookup(input_csv, output_pkl, merge_overlaps=False)


if __name__ == "__main__":
    main()


