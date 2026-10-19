"""
向量化服务模块：为文档切块批量生成稠密+稀疏向量。

改进点：
1. 语义增强文本引入跨境电商术语归一化（normalize_terms），
   让「用户搜 FBA 费用」能命中「文档写 亚马逊物流运费」的切片；
2. 分批策略可配置（默认 60 条/批，适配智谱 64 条上限）；
3. 批次失败不静默丢数据：记录真实原因，部分失败留 error 级日志。
"""
from app.infra.llm import llm_provider
from app.rag.import_.config import EMBEDDING_BATCH_SIZE, VECTOR_ENHANCE_MAX_TERMS
from app.shared.runtime.logger import logger
from app.shared.utils.ec_utils import normalize_terms


def require_chunks(state: dict) -> list[dict]:
    """校验切片存在。"""
    chunks = state.get("chunks", [])
    if not chunks:
        raise ValueError("chunks 为空，无法向量化")
    return chunks


def build_vector_text(chunk: dict) -> str:
    """
    构造用于向量化的语义增强文本。

    除正文外，把文档标题、平台、市场、类目、文档类型与归一化术语拼进语义，
    使「亚马逊FBA仓储费」这类带条件提问获得更好的语义召回。
    """
    content = chunk.get("content", "")
    segments: list[str] = []
    if chunk.get("article_title"):
        segments.append(f"主题:{chunk['article_title']}")
    if chunk.get("platform") and chunk["platform"] != "未分类":
        segments.append(f"平台:{chunk['platform']}")
    if chunk.get("market") and chunk["market"] != "未分类":
        segments.append(f"市场:{chunk['market']}")
    if chunk.get("category") and chunk["category"] != "未分类":
        segments.append(f"类目:{chunk['category']}")
    if chunk.get("doc_type") and chunk["doc_type"] != "未分类":
        segments.append(f"文档类型:{chunk['doc_type']}")
    # 术语归一化：把正文中的缩写补全为规范表述（增强语义召回）
    enhanced = normalize_terms(content, max_terms=VECTOR_ENHANCE_MAX_TERMS)
    segments.append(f"内容:{enhanced}")
    return ",".join(segments)


def embed_chunks(chunks: list[dict], *, step: int = EMBEDDING_BATCH_SIZE) -> list[dict]:
    """批量为切片生成向量（dense + sparse）。"""
    chunks_vector: list[dict] = []
    failures: list[str] = []
    total = len(chunks)

    for index in range(0, total, step):
        try:
            step_chunks = chunks[index:index + step]
            vector_str_list = [build_vector_text(item) for item in step_chunks]
            result = llm_provider.embed_documents(vector_str_list)
            sparse_list = result.get("sparse")
            for i, chunk in enumerate(step_chunks):
                chunk_new = chunk.copy()
                chunk_new["dense_vector"] = result["dense"][i]
                chunk_new["sparse_vector"] = sparse_list[i] if sparse_list else None
                chunks_vector.append(chunk_new)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"index={index} 批次向量化失败，跳过：{exc}")
            failures.append(f"批次[起始index={index}]:{exc}")
            continue

    if not chunks_vector:
        detail = " | ".join(failures) or "未知原因"
        raise RuntimeError(f"{total} 个切片向量化全部失败：{detail}")
    if len(chunks_vector) < total:
        logger.error(
            f"部分切片向量化失败并已丢弃：成功 {len(chunks_vector)}/{total}，"
            f"详情：{' | '.join(failures)}"
        )
    return chunks_vector


def generate_chunk_embeddings(state: dict) -> dict:
    """向量化主入口（node_embedding 业务实现）。"""
    state["chunks"] = embed_chunks(require_chunks(state))
    return state
