import os
import pandas as pd
from glob import glob
from tqdm import tqdm  # 进度条（可选）

def check_segment_id_in_parquets(directory):
    """
    检查指定目录下所有Parquet文件是否每行都有segment_id值
    
    参数:
        directory (str): 包含Parquet文件的根目录路径（支持递归搜索）
    
    返回:
        dict: {
            "valid_files": int,       # 完全合规的文件数量
            "invalid_files": list,   # 有缺失segment_id的文件列表（含详细信息）
            "total_files": int       # 总检查文件数
        }
    """
    # 1. 获取所有Parquet文件路径（递归搜索）
    parquet_files = glob(os.path.join(directory, "**/*.parquet"), recursive=True)
    
    if not parquet_files:
        raise ValueError(f"目录中未找到Parquet文件: {directory}")
    
    # 2. 初始化结果统计
    results = {
        "valid_files": 0,
        "invalid_files": [],
        "total_files": len(parquet_files)
    }
    
    # 3. 逐个检查文件（带进度条）
    for file_path in tqdm(parquet_files, desc="检查文件中"):
        try:
            # 读取Parquet文件
            df = pd.read_parquet(file_path)
            
            # 检查segment_id是否存在且无缺失
            if "segment_id" not in df.columns:
                results["invalid_files"].append({
                    "file": file_path,
                    "issue": "缺少segment_id列",
                    "missing_rows": "全部"
                })
                continue
                
            # 检查缺失值
            missing_mask = df["segment_id"].isna()
            if missing_mask.any():
                missing_count = missing_mask.sum()
                results["invalid_files"].append({
                    "file": file_path,
                    "issue": f"{missing_count}行缺失segment_id",
                    "missing_rows": df[missing_mask].index.tolist()[:10]  # 示例前10行
                })
            else:
                results["valid_files"] += 1
                
        except Exception as e:
            results["invalid_files"].append({
                "file": file_path,
                "issue": f"文件读取失败: {str(e)}",
                "missing_rows": "N/A"
            })
    
    return results


# 使用示例
if __name__ == "__main__":
    report = check_segment_id_in_parquets("Data/processed/lsmt/sliding_window/segment_fixe")
    
    print(f"\n检查完成！结果摘要：")
    print(f"总文件数: {report['total_files']}")
    print(f"合规文件: {report['valid_files']}")
    print(f"问题文件: {len(report['invalid_files'])}")
    
    if report["invalid_files"]:
        print("\n问题文件详情（前5个）：")
        for item in report["invalid_files"][:5]:
            print(f"- 文件: {item['file']}")
            print(f"  问题: {item['issue']}")
            if isinstance(item["missing_rows"], list):
                print(f"  示例缺失行索引: {item['missing_rows']}")