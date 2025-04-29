import os
import logging
from datetime import datetime
from pathlib import Path
import inspect

def setup_logger(log_file: str = None, name: str = "preprocessing_pipeline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] - %(message)s")

        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 文件输出
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger


# === 自动生成 log 文件路径 ===

# 1. 获取调用模块名（如 convert_csv_to_parquet）
caller = inspect.stack()[1].filename
caller_name = Path(caller).stem  # 提取文件名（不带.py）

# 2. 构造日志目录和文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = Path("/home/wanting/energy_anomaly_detection/experiments/logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"{caller_name}_{timestamp}.log"

# ✅ 初始化 logger（自动按脚本命名）
logger = setup_logger(str(log_file), name=caller_name)


