import pandas as pd
import numpy as np
import os
from typing import List, Dict, Tuple
from datetime import datetime
from pathlib import Path
import math  # 添加缺失的math导入
from tqdm import tqdm  # 添加进度条支持

class EnergyDataAnalyzer:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.data_dirs = ['Contacting', 'PCB', 'Ring']
        self.sample_size = 1  # 修改为只取1个样本
        self.report = []
        
        # 验证基础路径是否存在
        if not self.base_path.exists():
            raise ValueError(f"Base path does not exist: {self.base_path}")
        
    def sample_files(self) -> Dict[str, List[Path]]:
        """从每个目录中只采样一个月的文件"""
        print("开始采样文件...")
        sampled_files = {}
        for dir_name in self.data_dirs:
            dir_path = self.base_path / dir_name
            print(f"正在处理目录: {dir_name}")
            if dir_path.exists():
                files = list(dir_path.glob('*.csv'))
                if files:
                    # 只取第一个月的文件
                    first_file = files[0]
                    sampled_files[dir_name] = [first_file]
                    print(f"已选择文件: {first_file.name} 从 {dir_name}")
        return sampled_files
    
    def analyze_data_structure(self, df: pd.DataFrame) -> Dict:
        """分析数据结构"""
        try:
            # 使用更高效的方式检查重复
            duplicate_rows = df.duplicated(keep=False)
            duplicate_count = duplicate_rows.sum()
            
            # 只获取需要的列来检查重复
            id_timestamp_duplicates = df[['ID', 'TimeStamp']].duplicated(keep=False).sum()
            
            # 优化性能的数据结构分析
            structure = {
                'columns': list(df.columns),
                'dtypes': df.dtypes.astype(str).to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'unique_values': {col: df[col].nunique() for col in df.columns},
                'duplicate_values': {col: len(df) - df[col].nunique() for col in df.columns},
                'sample_values': {col: df[col].head(3).tolist() for col in df.columns},
                'duplicate_info': {
                    'total_duplicate_rows': duplicate_count,
                    'id_timestamp_duplicates': id_timestamp_duplicates
                }
            }
            return structure
        except Exception as e:
            print(f"分析数据结构时出错: {str(e)}")
            return {}
    
    def _add_section(self, title: str, level: int = 1):
        """添加Markdown标题"""
        self.report.append(f"{'#' * level} {title}\n")
    
    def _add_list(self, items: List[str]):
        """添加Markdown列表"""
        for item in items:
            self.report.append(f"- {item}\n")
    
    def _add_table(self, data: Dict, headers: List[str]):
        """添加Markdown表格"""
        table = []
        # 添加表头
        table.append("| " + " | ".join(headers) + " |")
        table.append("| " + " | ".join(["---"] * len(headers)) + " |")
        
        # 添加数据行
        for row in data:
            table.append("| " + " | ".join(str(x) for x in row) + " |")
        
        self.report.append("\n".join(table) + "\n")
    
    def generate_report(self) -> str:
        """生成Markdown格式的报告"""
        print("开始生成报告...")
        self.report = []
        
        # 添加报告标题
        self._add_section("能源数据结构分析报告")
        self.report.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 分析每个目录
        sampled_files = self.sample_files()
        for dir_name, files in sampled_files.items():
            print(f"\n分析目录: {dir_name}")
            self._add_section(f"目录分析: {dir_name}", 2)
            
            for file in tqdm(files, desc=f"处理 {dir_name} 中的文件"):
                try:
                    print(f"\n读取文件: {file.name}")
                    df = pd.read_csv(file)
                    print(f"文件大小: {len(df)} 行 x {len(df.columns)} 列")
                    
                    self._add_section(f"文件: {file.name}", 3)
                    
                    # 基本信息
                    self._add_section("基本信息", 4)
                    self._add_list([
                        f"总行数: {len(df)}",
                        f"总列数: {len(df.columns)}",
                        f"时间范围: {df['TimeStamp'].min()} 到 {df['TimeStamp'].max()}"
                    ])
                    
                    # 时间连续性分析
                    self._add_section("时间连续性分析", 4)
                    time_intervals = self.calculate_time_intervals(df['TimeStamp'])
                    self._add_list([
                        f"最小时间间隔: {time_intervals['minInterval']}秒",
                        f"最大时间间隔: {time_intervals['maxInterval']}秒",
                        f"平均时间间隔: {time_intervals['avgInterval']}秒",
                        f"时间间隔标准差: {time_intervals['stdInterval']}秒"
                    ])
                    
                    # 数据结构分析
                    self._add_section("数据结构分析", 4)
                    structure = self.analyze_data_structure(df)
                    
                    # 列信息表格
                    self._add_section("列信息", 5)
                    table_data = []
                    for col in structure['columns']:
                        table_data.append([
                            col,
                            structure['dtypes'][col],
                            structure['missing_values'][col],
                            structure['unique_values'][col],
                            structure['duplicate_values'][col],
                            str(structure['sample_values'][col])
                        ])
                    self._add_table(table_data, ['列名', '数据类型', '缺失值数量', '唯一值数量', '重复值数量', '示例值'])
                    
                    # 添加重复数据信息
                    self._add_section("重复数据分析", 5)
                    self._add_list([
                        f"完全重复的行数: {structure['duplicate_info']['total_duplicate_rows']}",
                        f"ID和TimeStamp同时重复的行数: {structure['duplicate_info']['id_timestamp_duplicates']}"
                    ])
                    
                except Exception as e:
                    print(f"处理文件 {file} 时出错: {str(e)}")
                    continue
        
        return "\n".join(self.report)
    
    def analyze_directory(self) -> None:
        """分析整个目录的数据并生成报告"""
        try:
            print("开始分析目录...")
            report = self.generate_report()
            output_file = "data_structure_report.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n分析完成。报告已保存到 {output_file}")
        except Exception as e:
            print(f"分析过程中发生错误: {str(e)}")

    def calculate_time_intervals(self, timestamps: pd.Series) -> Dict:
        """计算时间间隔指标"""
        # 将时间戳转换为Date对象数组并排序
        dates = timestamps.map(lambda ts: pd.to_datetime(ts).to_pydatetime())
        sorted_dates = sorted(dates)
        
        # 计算时间间隔(秒)
        intervals = []
        for i in range(1, len(sorted_dates)):
            interval = (sorted_dates[i] - sorted_dates[i-1]).total_seconds()
            intervals.append(interval)
        
        # 计算统计指标
        min_interval = min(intervals)
        max_interval = max(intervals)
        avg_interval = sum(intervals) / len(intervals)
        
        # 计算标准差
        variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
        std_interval = math.sqrt(variance)
        
        return {
            'minInterval': min_interval,
            'maxInterval': max_interval,
            'avgInterval': avg_interval,
            'stdInterval': std_interval
        }

if __name__ == "__main__":
    base_path = "/home/wanting/energy_anomaly_detection/Data/row/Energy_Data"
    analyzer = EnergyDataAnalyzer(base_path)
    analyzer.analyze_directory() 