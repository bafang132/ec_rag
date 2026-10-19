"""
路径工具模块：统一推导项目根目录，避免各模块重复计算。
"""
from pathlib import Path

# app/shared/utils/path_util.py -> 项目根为上四级
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# 常用目录
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"
OUTPUT_DIR = PROJECT_ROOT / "output"
SOURCE_DIR = PROJECT_ROOT / "app" / "source"
PROMPT_DIR = SOURCE_DIR / "prompts"
DOC_DIR = SOURCE_DIR / "doc"


def ensure_dir(path: Path) -> Path:
    """确保目录存在并返回。"""
    path.mkdir(parents=True, exist_ok=True)
    return path
