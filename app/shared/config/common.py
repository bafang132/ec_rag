"""
环境变量读取助手：统一类型转换与默认值处理。
"""
import os
from typing import Optional


def env_str(key: str, default: str = "") -> str:
    """读取字符串型环境变量。"""
    value = os.getenv(key)
    return default if value is None or value == "" else value


def env_bool(key: str, default: bool = False) -> bool:
    """读取布尔型环境变量（true/1/yes/on 视为真）。"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


def env_int(key: str, default: int = 0) -> int:
    """读取整型环境变量，非法值回落默认值。"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(key: str, default: float = 0.0) -> float:
    """读取浮点型环境变量，非法值回落默认值。"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def env_list(key: str, default: Optional[list[str]] = None, sep: str = ",") -> list[str]:
    """读取列表型环境变量（分隔符分割，去空白）。"""
    raw = os.getenv(key)
    if raw is None or raw.strip() == "":
        return list(default or [])
    return [item.strip() for item in raw.split(sep) if item.strip()]
