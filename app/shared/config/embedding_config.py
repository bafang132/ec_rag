"""
Embedding 配置模块：支持本地 BGE-M3 / 智谱 API / 开发哈希向量三种提供方。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_int, env_str


@dataclass
class EmbeddingConfig:
    # auto: 优先本地 BGE-M3，不可用则智谱 API，再不可用则哈希向量（仅开发）
    provider: str = env_str("EMBEDDING_PROVIDER", "auto").lower()
    dim: int = env_int("EMBEDDING_DIM", 1024)
    bge_m3_path: str = env_str("BGE_M3_PATH", "")
    bge_m3: str = env_str("BGE_M3", "BAAI/bge-m3")
    bge_device: str = env_str("BGE_DEVICE", "cpu")
    bge_fp16: bool = env_bool("BGE_FP16", False)
    zhipu_api_key: str = env_str("ZHIPU_API_KEY", "")
    zhipu_embedding_model: str = env_str("ZHIPU_EMBEDDING_MODEL", "embedding-2")


embedding_config = EmbeddingConfig()
