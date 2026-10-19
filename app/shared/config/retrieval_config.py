"""
检索配置模块：候选池、重排缓存与召回策略参数。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_int


@dataclass
class RetrievalConfig:
    default_limit: int = env_int("RETRIEVAL_DEFAULT_LIMIT", 15)
    rerank_cache_max: int = env_int("RERANK_CACHE_MAX", 500)
    rerank_cache_ttl: int = env_int("RERANK_CACHE_TTL", 300)
    # 标量硬过滤：默认关闭。LLM 抽取的平台/类目标签常与索引值不匹配，
    # 硬过滤会把正确文档整批排除导致召回崩溃（实测 HitRate@5 从 100% 跌到 20%）。
    # 关闭后平台/市场/类目仅作语义增强拼接到查询，由重排器负责精度。
    use_scalar_filter: bool = env_bool("RETRIEVAL_USE_SCALAR_FILTER", False)


retrieval_config = RetrievalConfig()
