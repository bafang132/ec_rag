"""
重排服务模块：多路召回结果整理、Cross-Encoder/LLM 打分与动态截断。

重排设计：
- 统一本地召回与联网搜索结果格式；
- 候选文本超限时先经 LLM 精炼再打分；
- 按相邻分差动态截断 TopK（保质量也控上下文长度）；
- 内存 LRU 缓存相同 query+候选 的打分结果（线程安全）。
"""
import hashlib
import threading
import time
from typing import Optional

from app.infra.llm import llm_provider
from app.shared.config.retrieval_config import retrieval_config
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.shared.model.lm_utils import build_messages, invoke_text

RERANK_MAX_TOPK = 10
RERANK_MIN_TOPK = 5
# 动态截断阈值（针对归一化分数 0-1 设计）：
# 相邻分差 > 0.15（绝对）或跌幅 > 50%（相对）时截断，保留高质量候选。
# 注：旧版设计使用 abs=2.0 / ratio=2.0，在 0-1 归一化分数下永不失灵，
# 动态截断形同虚设；本系统按归一化分数量程修正。
RERANK_GAP_RATIO = 0.5
RERANK_GAP_ABS = 0.15
RERANK_MAX_INPUT_TOKENS = 512
RERANK_SUMMARY_CHAR_RATIO = 1.3
RERANK_MIN_SUMMARY_CHARS = 50

_RERANK_CACHE_MAX = retrieval_config.rerank_cache_max
_RERANK_CACHE_TTL = retrieval_config.rerank_cache_ttl
_rerank_cache: dict[str, tuple[float, list[dict]]] = {}
_rerank_cache_lock = threading.Lock()


def _rerank_cache_key(query: str, chunk_ids: list[str]) -> str:
    raw = f"{query}|{','.join(sorted(chunk_ids))}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _chunk_fingerprint(text: str, index: int) -> str:
    """候选文档的稳定指纹（正文哈希，内容一变即换 key）。"""
    digest = hashlib.md5((text or "").encode("utf-8")).hexdigest()[:12]
    return f"text_{index}_{digest}"


def _rerank_cache_get(query: str, chunk_ids: list[str]) -> Optional[list[dict]]:
    key = _rerank_cache_key(query, chunk_ids)
    with _rerank_cache_lock:
        entry = _rerank_cache.get(key)
        if entry is None:
            return None
        ts, docs = entry
        if time.time() - ts > _RERANK_CACHE_TTL:
            _rerank_cache.pop(key, None)
            return None
        return docs


def _rerank_cache_set(query: str, chunk_ids: list[str], docs: list[dict]) -> None:
    key = _rerank_cache_key(query, chunk_ids)
    with _rerank_cache_lock:
        if len(_rerank_cache) >= _RERANK_CACHE_MAX:
            oldest_key = min(_rerank_cache, key=lambda k: _rerank_cache[k][0])
            _rerank_cache.pop(oldest_key, None)
        _rerank_cache[key] = (time.time(), docs)


def merge_rrf_and_web(rrf_chunks: list[dict], web_search_docs: list[dict]) -> list[dict]:
    """把本地召回与联网搜索结果整理成统一格式（供同一个重排器打分）。"""
    final_chunk_list: list[dict] = []
    skipped = 0
    for chunk in rrf_chunks or []:
        if not isinstance(chunk, dict):
            skipped += 1
            continue
        final_chunk_list.append({
            "item_name": chunk.get("item_name"),
            "chunk_id": chunk.get("chunk_id"),
            "file_title": chunk.get("file_title"),
            "title": chunk.get("title"),
            "text": chunk.get("content"),
            "url": None,
            "type": chunk.get("type", "vector"),
            "score": chunk.get("score", 0.0),
        })
    for doc in web_search_docs or []:
        if not isinstance(doc, dict):
            skipped += 1
            continue
        final_chunk_list.append({
            "title": doc.get("title"),
            "text": doc.get("snippet"),
            "url": doc.get("url"),
            "type": "web",
            "score": 0.0,
        })
    if skipped:
        logger.warning(f"重排输入存在 {skipped} 条非字典数据，已跳过")
    logger.info(
        f"两路数据统一完成：本地 {len(rrf_chunks or [])} 条 + 联网 {len(web_search_docs or [])} 条 "
        f"= {len(final_chunk_list)} 条"
    )
    return final_chunk_list


def summarize_long_rerank_text(question: str, answer: str, limit: int) -> str:
    """候选文本超出重排上下文上限时，先调 LLM 精炼。"""
    try:
        client = llm_provider.chat()
        prompt = load_prompt("rerank_text_refine", question=question, answer=answer, limit=limit)
        refined = invoke_text(client, build_messages("你是文本精简提炼专家", prompt))
        logger.debug(f"重排前文本精炼：{len(answer)} → {len(refined)} 字符")
        return refined
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"重排文本精炼失败，使用截断兜底：{exc}")
        return answer[:limit]


def build_question_pairs(question: str, final_chunk_list: list[dict], reranker) -> list[list[str]]:
    """构造重排模型的问答对列表（超长文本先精炼，预留 64 token 余量防截断）。"""
    tokenizer = reranker.tokenizer
    query_tokens = tokenizer.encode(question, add_special_tokens=False)
    headroom = 64
    pairs: list[list[str]] = []
    for item in final_chunk_list:
        answer = item.get("text") or ""
        answer_for_rerank = answer
        answer_tokens = tokenizer.encode(answer, add_special_tokens=False)
        total_tokens = len(query_tokens) + len(answer_tokens) + headroom
        if total_tokens > RERANK_MAX_INPUT_TOKENS:
            limit = max(
                RERANK_MIN_SUMMARY_CHARS,
                int((RERANK_MAX_INPUT_TOKENS - len(query_tokens) - headroom) / RERANK_SUMMARY_CHAR_RATIO),
            )
            logger.info(f"候选文本超出重排上下文限制，精炼后再打分（token {total_tokens} > {RERANK_MAX_INPUT_TOKENS}）")
            answer_for_rerank = summarize_long_rerank_text(question, answer, limit)
        pairs.append([question, answer_for_rerank])
    return pairs


def score_and_sort_chunks(state: dict, final_chunk_list: list[dict]) -> list[dict]:
    """
    调用重排模型打分并按分数降序排序。

    重排器不可用（初始化失败/返回 None）时降级为按召回分数排序，
    保证问答链路不中断（精度让位于可用性）。
    """
    if not final_chunk_list:
        return []
    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    try:
        reranker = llm_provider.reranker_model()
        pairs = build_question_pairs(rewritten_query, final_chunk_list, reranker)
        score_list = reranker.compute_score(pairs, normalize=True)
        for score, chunk in zip(score_list, final_chunk_list):
            chunk["score"] = round(float(score), 4)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"重排模型不可用，降级按召回分数排序：{exc}")
        for chunk in final_chunk_list:
            chunk.setdefault("score", 0.0)
    final_chunk_list.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return final_chunk_list


def dynamic_topk(chunk_list_score_sorted: list[dict]) -> list[dict]:
    """按相邻分差动态截断 TopK（分差突变处截断，保留高质量候选）。"""
    if not chunk_list_score_sorted:
        return []
    min_topk = RERANK_MIN_TOPK
    max_topk = min(RERANK_MAX_TOPK, len(chunk_list_score_sorted))
    topk = max_topk
    if topk > min_topk:
        for index in range(min_topk - 1, max_topk - 1):
            score_1 = chunk_list_score_sorted[index].get("score", 0.0)
            score_2 = chunk_list_score_sorted[index + 1].get("score", 0.0)
            abs_score = score_1 - score_2
            ratio_score = abs_score / (score_1 + 1e-7)
            if abs_score > RERANK_GAP_ABS or ratio_score > RERANK_GAP_RATIO:
                topk = index + 1
                break
    return chunk_list_score_sorted[:topk]


def rerank_documents(state: dict) -> list[dict]:
    """
    重排主流程（node_rerank 业务实现）。

    流程：合并两路 → 缓存查询 → 打分排序 → 动态截断。
    """
    rrf_chunks = state.get("rrf_chunks", []) or []
    web_docs = state.get("web_search_docs", []) or []
    merged = merge_rrf_and_web(rrf_chunks, web_docs)

    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    chunk_ids = [
        _chunk_fingerprint(c.get("content") or c.get("text") or "", i)
        for i, c in enumerate(merged)
    ]
    cached = _rerank_cache_get(rewritten_query, chunk_ids)
    if cached is not None:
        logger.info(f"Rerank 缓存命中，直接返回 {len(cached)} 条")
        return dynamic_topk(cached)

    sorted_docs = score_and_sort_chunks(state, merged)
    _rerank_cache_set(rewritten_query, chunk_ids, sorted_docs)
    return dynamic_topk(sorted_docs)
