import yaml
import os

def load_config(config_path: str) -> dict:
    """
    读取 YAML 配置文件，并以字典形式返回。
    如果文件不存在或格式错误，抛出友好的错误提示。
    """
    # 检查文件是否存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            if config is None:
                raise ValueError(f"配置文件 {config_path} 是空的或者内容格式有误。")
    except yaml.YAMLError as e:
        raise ValueError(f"解析 YAML 文件出错：{e}")

    return config