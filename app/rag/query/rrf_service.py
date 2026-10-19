"""
RRF 融合服务模块：多路召回结果（语义 / HyDE / 联网）的 Reciprocal Rank Fusion 融合。
"""
from app.shared.runtime.logger import logger


def reciprocal_rank_fusion(
    param_list: list[tuple[list[dict], float]],
    *,
    k: int = 60,
    top: int = 15,
) -> list[dict]:
    """
    对多路召回结果执行 RRF 融合（k=60 为经典平滑参数）。

    :param param_list: [(该路结果列表, 权重), ...]
    :param k: RRF 平滑参数
    :param top: 最终保留条数
    :return: 融合后按分数降序的文档列表
    """
    score_dict: dict[str, float] = {}
    entity_dict: dict[str, dict] = {}
    for chunks_list, weight in param_list:
        for rank, chunk in enumerate(chunks_list or [], start=1):
            chunk_id = chunk.get("chunk_id")
            if not chunk_id:
                continue
            score_dict[chunk_id] = score_dict.get(chunk_id, 0.0) + (1.0 / (k + rank)) * weight
            entity_dict.setdefault(chunk_id, chunk)

    document_list = []
    for chunk_id, score in score_dict.items():
        document = entity_dict.get(chunk_id, {}).copy()
        document["score"] = score
        document_list.append(document)
    document_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    final_documents = document_list[:top]
    logger.info(
        f"RRF融合完成：输入 {len(param_list)} 路，输出 {len(final_documents)} 条"
    )
    return final_documents


def fuse_retrieval_results(state: dict) -> list[dict]:
    """
    查询链 RRF 融合步骤（node_rrf 业务实现）。

    当前融合：语义检索 + HyDE 检索两路（等权）。
    """
    embedding_chunks = state.get("embedding_chunks", []) or []
    hyde_chunks = state.get("hyde_embedding_chunks", []) or []
    return reciprocal_rank_fusion([
        (embedding_chunks, 1.0),
        (hyde_chunks, 1.0),
    ])
