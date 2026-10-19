"""
Reranker 配置模块：本地 bge-reranker-large / LLM 兜底重排。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_str


@dataclass
class RerankerConfig:
    # auto: 优先本地 FlagReranker，不可用则 LLM 兜底
    provider: str = env_str("RERANKER_PROVIDER", "auto").lower()
    bge_reranker_large: str = env_str("BGE_RERANKER_LARGE", "BAAI/bge-reranker-large")
    bge_reranker_device: str = env_str("BGE_RERANKER_DEVICE", "cpu")
    bge_reranker_fp16: bool = env_bool("BGE_RERANKER_FP16", False)


reranker_config = RerankerConfig()
