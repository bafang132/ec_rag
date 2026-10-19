"""
语义检索服务模块：混合向量检索（dense + BM25/Milvus sparse）。

检索策略（基于实测经验）：
- **纯语义召回，不做标量硬过滤**：平台/市场/类目由 LLM 抽取，常与索引标签对不上，
  作为硬过滤会把正确文档整批排除，导致召回崩溃。
- **软召回**：把平台/市场/类目拼接到查询文本（语义增强），让向量与关键词两路都能感知；
  精度交给下游重排器（bge-reranker / LLM 兜底）负责。
"""
from typing import List, Optional

from app.infra.llm import llm_provider
from app.infra.vectorstore import vectorstore_gateway
from app.shared.config.ec_config import UNKNOWN_VALUE
from app.shared.config.retrieval_config import retrieval_config
from app.shared.runtime.logger import logger
from app.shared.utils.ec_utils import normalize_terms

RETRIEVAL_DEFAULT_LIMIT = retrieval_config.default_limit


def normalize_retrieved_chunk(chunk: dict) -> dict:
    """把向量库返回结果归一化为查询链内部统一结构。"""
    return {
        "chunk_id": chunk.get("chunk_id"),
        "item_name": chunk.get("item_name", ""),
        "title": chunk.get("title"),
        "parent_title": chunk.get("parent_title"),
        "part": chunk.get("part"),
        "file_title": chunk.get("file_title"),
        "content": chunk.get("content", ""),
        "score": chunk.get("score", 0.0),
        "type": vectorstore_gateway.backend,
        "url": None,
        "platform": chunk.get("platform", ""),
        "market": chunk.get("market", ""),
        "category": chunk.get("category", ""),
        "doc_type": chunk.get("doc_type", ""),
    }


def build_semantic_query(rewritten_query: str, ec_filters: dict, scene: str = "") -> str:
    """
    构造语义增强查询：改写问题 + 平台/市场/类目标签 + 术语归一化。

    示例："FBA运费怎么算" + 平台:亚马逊,市场:北美,类目:物流仓储
          + 术语:FBA 亚马逊物流（Fulfillment by Amazon）
    """
    parts: list[str] = [rewritten_query]
    filters = ec_filters or {}
    for key, label in (("platform", "平台"), ("market", "市场"), ("category", "类目")):
        value = filters.get(key, "")
        if value and value != UNKNOWN_VALUE:
            parts.append(f"{label}:{value}")
    if scene:
        parts.append(f"场景:{scene}")
    enhanced = " ".join(parts)
    # 术语归一化：补全缩写对应的规范表述，弥合语义鸿沟
    return normalize_terms(enhanced)


def search_ec_chunks(
    *,
    rewritten_query: str,
    ec_filters: Optional[dict] = None,
    scene: str = "",
    limit: int = RETRIEVAL_DEFAULT_LIMIT,
    item_names: Optional[List[str]] = None,
    use_web_search: bool = False,
) -> list[dict]:
    """
    执行跨境电商语义检索（混合向量召回 + 可选联网搜索并发）。

    :param rewritten_query: 改写后的查询
    :param ec_filters: 领域筛选条件（软召回，拼接进语义查询）
    :param scene: 场景描述（语义召回）
    :param limit: 最大返回条数
    :param item_names: 主体名过滤（硬过滤开启时生效）
    :param use_web_search: 是否启用联网搜索（由并行搜索层处理）
    :return: 归一化切片列表
    """
    semantic_query = build_semantic_query(rewritten_query, ec_filters or {}, scene)
    logger.info(f"语义检索查询：{semantic_query[:120]}")

    embedding_result = llm_provider.embed_documents([semantic_query])
    dense_vector = embedding_result["dense"][0]
    sparse_vector = embedding_result["sparse"][0] if embedding_result.get("sparse") else None

    expr = None
    if item_names and retrieval_config.use_scalar_filter:
        expr = f"item_name in {item_names}"

    results = vectorstore_gateway.hybrid_search(
        dense_vector=dense_vector,
        query_text=semantic_query,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=limit,
    )
    logger.info(f"混合检索返回 {len(results)} 个候选（backend={vectorstore_gateway.backend}）")
    return [normalize_retrieved_chunk(chunk) for chunk in results]
