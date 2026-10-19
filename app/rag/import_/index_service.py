"""
入库服务模块：把带向量的切片写入向量库（Milvus / 本地库，经门面自动选择）。

幂等策略：同一主体（item_name）重复导入时先删旧数据，保证当前导入覆盖旧版本。
"""
from app.infra.vectorstore import vectorstore_gateway
from app.rag.import_.config import CHUNK_FIELD_DEFAULTS
from app.rag.import_.version_service import record_version
from app.shared.runtime.logger import logger


def require_chunks(state: dict) -> list[dict]:
    """校验切片存在且已向量化。"""
    chunks = state.get("chunks", [])
    if not chunks:
        raise ValueError("chunks 为空，无法入库")
    missing = [i for i, c in enumerate(chunks) if not c.get("dense_vector")]
    if missing:
        raise ValueError(f"有 {len(missing)} 个切片缺少向量（如切片 {missing[:5]}），无法入库")
    return chunks


def list_all_items(limit: int = 10000) -> list[dict]:
    """列出全部主体（管理页）。"""
    return vectorstore_gateway.list_items(limit=limit)


def remove_old_chunks(item_name: str) -> int:
    """删除指定主体的旧切片。"""
    return vectorstore_gateway.delete_by_item(item_name)


def delete_chunk_by_id(chunk_id: int) -> int:
    """按切片 ID 删除单个切片。"""
    return vectorstore_gateway.delete_by_chunk_id(chunk_id)


def index_chunks(state: dict) -> dict:
    """
    入库主入口（node_index 业务实现）。

    流程：校验 → 补齐必填字段 → 门面写入（自动处理同主体覆盖）。
    """
    chunks = require_chunks(state)
    item_name = state.get("item_name") or state.get("file_title") or ""

    for chunk in chunks:
        chunk["item_name"] = item_name or chunk.get("file_title") or ""
        for field, default in CHUNK_FIELD_DEFAULTS.items():
            if field not in chunk or chunk.get(field) is None:
                chunk[field] = default

    count = vectorstore_gateway.insert_chunks(chunks)
    state["indexed_count"] = count
    logger.info(f"入库完成：主体[{item_name}] 写入 {count} 个切片（backend={vectorstore_gateway.backend}）")

    # 记录文档版本（增量更新依据）：按主体指纹 + 分片数落库
    try:
        record_version(
            item_name=item_name,
            content_hash=state.get("file_hash", "") or "",
            chunk_count=count,
            source=state.get("local_file_path", ""),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"版本记录失败（不影响入库）：{exc}")
    return state
