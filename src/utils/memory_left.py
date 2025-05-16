from src.utils.logger import logger
import psutil

def log_memory(prefix=""):
    mem = psutil.virtual_memory()
    used_gb = mem.used / 1024**3
    avail_gb = mem.available / 1024**3
    logger.info(f"[{prefix}] Memory used: {used_gb:.2f} GB | Available: {avail_gb:.2f} GB")
