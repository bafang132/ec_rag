"""
Milvus 向量库实现：混合检索（dense + sparse）+ 增删查管理。

与本地向量库（local_store.py）接口对齐，由上层门面按配置选择。
"""
from typing import Callable, Optional

from app.shared.config.milvus_config import milvus_config
from app.shared.clients import milvus_utils
from app.shared.runtime.logger import logger

# 归一化输出字段（与本地库保持一致）
DEFAULT_OUTPUT_FIELDS = [
    "chunk_id", "item_name", "file_title", "title", "parent_title", "part", "content",
    "platform", "market", "category", "doc_type", "article_title", "author",
    "source_file", "source_path",
]


class MilvusStore:
    """Milvus 向量库（生产混合检索）。"""

    backend = "milvus"

    def __init__(self, collection: Optional[str] = None) -> None:
        self._loaded_collections: set[str] = set()
        # 多租户：传入集合名覆盖默认集合，实现租户级数据隔离
        self._collection_override = collection

    @property
    def chunks_collection(self) -> str:
        return self._collection_override or milvus_config.chunks_collection

    def client(self):
        return milvus_utils.get_milvus_client()

    def ensure_loaded(self, collection_name: Optional[str] = None) -> None:
        """确保集合已加载到查询节点（混合检索前置条件，带进程内缓存）。"""
        collection = collection_name or self.chunks_collection
        if collection in self._loaded_collections:
            return
        client = self.client()
        try:
            state = client.get_load_state(collection_name=collection)
            if "loaded" in str(state).lower():
                self._loaded_collections.add(collection)
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"集合[{collection}]加载状态检查失败，直接尝试加载：{exc}")
        try:
            client.load_collection(collection_name=collection)
            self._loaded_collections.add(collection)
            logger.info(f"集合[{collection}]已加载，可执行混合检索")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"集合[{collection}]加载失败（可能已加载）：{exc}")

    def ensure_collection(self, dim: int = 1024) -> None:
        """确保切块集合存在（首次导入自动创建）。"""
        client = self.client()
        milvus_utils.create_chunks_collection(client, self.chunks_collection, dim=dim)

    def add_chunks(self, chunks: list[dict]) -> int:
        """批量写入切片（自动建集合 + 同主体先删后插，保证覆盖更新）。"""
        if not chunks:
            return 0
        client = self.client()
        dim = len(chunks[0].get("dense_vector") or [])
        self.ensure_collection(dim=dim or milvus_config.embedding_dim)

        item_names = {c.get("item_name") for c in chunks if c.get("item_name")}
        for name in item_names:
            self.delete_by_item(name)

        rows = []
        for chunk in chunks:
            row = {k: v for k, v in chunk.items() if k not in ("chunk_id", "score", "rank")}
            rows.append(row)
        result = client.insert(collection_name=self.chunks_collection, data=rows)
        count = result.get("insert_count", len(rows))
        self._loaded_collections.discard(self.chunks_collection)
        self.ensure_loaded()
        logger.info(f"Milvus 写入 {count} 个切片")
        return count

    def search(
        self,
        dense_vector: list[float],
        query_text: str = "",
        sparse_vector: Optional[dict] = None,
        expr: Optional[str] = None,
        limit: int = 15,
        output_fields: Optional[list[str]] = None,
        dense_weight: float = 0.9,
    ) -> list[dict]:
        """混合检索（dense + sparse，RRF 融合）。"""
        client = self.client()
        self.ensure_loaded()
        reqs = milvus_utils.create_hybrid_search_requests(
            dense_vector, sparse_vector or {}, expr=expr, limit=limit
        )
        resp = milvus_utils.hybrid_search(
            client=client,
            collection_name=self.chunks_collection,
            reqs=reqs,
            ranker_weights=(dense_weight, 1 - dense_weight),
            norm_score=True,
            limit=limit,
            output_fields=output_fields or DEFAULT_OUTPUT_FIELDS,
        )
        results = []
        for hit in (resp[0] if resp else []):
            entity = hit.get("entity", hit)
            item = {k: entity.get(k) for k in (output_fields or DEFAULT_OUTPUT_FIELDS)}
            item["score"] = round(float(hit.get("distance", 0.0)), 6)
            results.append(item)
        return results

    def delete_by_item(self, item_name: str) -> int:
        """按 item_name 删除全部切片。"""
        client = self.client()
        self.ensure_loaded()
        before = len(self.query_all(predicate=lambda m: m.get("item_name") == item_name,
                                    output_fields=["chunk_id"]))
        client.delete(
            collection_name=self.chunks_collection,
            filter=f"item_name == '{item_name}'",
        )
        logger.info(f"Milvus 删除主体[{item_name}]")
        return before

    def delete_by_chunk_id(self, chunk_id: int) -> int:
        """按 chunk_id 删除单个切片。"""
        client = self.client()
        self.ensure_loaded()
        client.delete(
            collection_name=self.chunks_collection,
            filter=f"chunk_id == {int(chunk_id)}",
        )
        return 1

    def query_all(
        self,
        predicate: Optional[Callable[[dict], bool]] = None,
        output_fields: Optional[list[str]] = None,
        limit: int = 10000,
    ) -> list[dict]:
        """全量/条件查询（管理页聚合用；predicate 在 Python 侧二次过滤）。"""
        client = self.client()
        self.ensure_loaded()
        try:
            rows = client.query(
                collection_name=self.chunks_collection,
                filter="chunk_id >= 0",
                output_fields=output_fields or DEFAULT_OUTPUT_FIELDS,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Milvus query_all 失败：{exc}")
            return []
        if predicate:
            rows = [r for r in rows if predicate(r)]
        return rows

    def list_items(self, limit: int = 10000) -> list[dict]:
        """按 item_name 聚合主体列表（管理页）。"""
        rows = self.query_all(limit=limit)
        items: dict[str, dict] = {}
        for meta in rows:
            name = meta.get("item_name") or ""
            if not name:
                continue
            if name not in items:
                items[name] = {
                    "item_name": name,
                    "file_title": meta.get("file_title", ""),
                    "title": meta.get("title", ""),
                    "platform": meta.get("platform", ""),
                    "market": meta.get("market", ""),
                    "category": meta.get("category", ""),
                    "doc_type": meta.get("doc_type", ""),
                    "chunk_count": 0,
                    "char_count": 0,
                }
            items[name]["chunk_count"] += 1
            items[name]["char_count"] += len(meta.get("content") or "")
        return list(items.values())

    def count(self) -> int:
        try:
            return len(self.query_all(output_fields=["chunk_id"]))
        except Exception:  # noqa: BLE001
            return 0

    def is_empty(self) -> bool:
        return self.count() == 0
