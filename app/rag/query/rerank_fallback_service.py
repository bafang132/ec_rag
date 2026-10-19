"""
RAG 降级处理模块：检索结果为空/质量过低时提示联网搜索，以及联网结果合并。
"""
from typing import Optional

from app.shared.runtime.logger import logger
from app.shared.utils.sse_utils import SSEEvent, push_to_session

# 触发降级提示的最高分阈值（低于此值视为无效检索）
LOW_SCORE_THRESHOLD = 0.05


def check_rag_fallback_needed(
    reranked_docs: list[dict],
    session_id: str,
    original_query: str,
) -> bool:
    """
    检查检索是否不足以回答问题，是则通过 SSE 推送 fallback 提示。

    触发条件（满足任一）：
    - 检索结果为空；
    - 最高分低于 LOW_SCORE_THRESHOLD。
    """
    if not reranked_docs:
        logger.info("[fallback] 检索结果为空，触发联网搜索提示")
        push_to_session(session_id, SSEEvent.WEB_SEARCH_HINT, {
            "query": original_query,
            "message": "本地知识库暂无相关答案，是否切换为联网搜索模式获取最新信息？",
        })
        return True

    top_score = max(doc.get("score", 0.0) for doc in reranked_docs)
    if top_score < LOW_SCORE_THRESHOLD:
        logger.info(f"[fallback] 最高分 {top_score:.4f} 过低，触发联网搜索提示")
        push_to_session(session_id, SSEEvent.WEB_SEARCH_HINT, {
            "query": original_query,
            "message": "本地知识库匹配度较低，是否切换为联网搜索模式？",
        })
        return True
    return False


def merge_web_search_results(
    reranked_docs: list[dict],
    web_search_docs: list[dict],
    threshold: float = 0.3,
) -> list[dict]:
    """
    将联网搜索结果合并到已有结果（按 score 递减打分，仅质量不足时使用）。
    """
    if not web_search_docs:
        return reranked_docs
    merged = list(reranked_docs or [])
    for i, doc in enumerate(web_search_docs, start=len(merged) + 1):
        merged.append({
            "chunk_id": f"web_{i}",
            "title": doc.get("title", ""),
            "content": doc.get("snippet", ""),
            "text": doc.get("snippet", ""),
            "url": doc.get("url", ""),
            "type": "web",
            "score": max(0.1, 1.0 - i * 0.05),
        })
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    logger.info(f"[merge_web] 合并后共 {len(merged)} 条文档")
    return merged
