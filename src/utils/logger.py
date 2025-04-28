import logging

def setup_logger(log_file=None):
    logger = logging.getLogger("preprocessing_pipeline")
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # 控制台输出
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # 如果提供了log文件路径，同时保存到文件
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger
