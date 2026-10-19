"""
向量库门面模块：统一封装 Milvus / 本地向量库两种后端。

选择策略（VECTORSTORE_PROVIDER）：
- auto    : 配置了 MILVUS_URL 且 pymilvus 可用 → Milvus；否则本地向量库
- milvus  : 强制 Milvus（不可用则启动报错）
- local   : 强制本地向量库

上层业务（导入/检索/管理）只依赖本门面，不关心底层实现。
"""
from typing import Optional

from app.shared.config.common import env_str
from app.shared.config.milvus_config import milvus_config
from app.shared.config.retrieval_config import retrieval_config
from app.shared.runtime.logger import logger
from app.shared.runtime.tenant import DEFAULT_TENANT

_store_instances: dict[str, object] = {}


def _local_persist_dir(tenant: str) -> str:
    """本地向量库路径：默认租户沿用原路径，其余按租户分目录隔离。"""
    if tenant == DEFAULT_TENANT:
        return env_str("LOCAL_VECTOR_DIR", "data/vector_store/ec_local_db")
    return f"data/vector_store/{tenant}/ec_local_db"


def _milvus_collection(tenant: str) -> str:
    """Milvus 集合名：默认租户沿用原集合，其余加租户后缀。"""
    if tenant == DEFAULT_TENANT:
        return None  # None → 使用 milvus_config 默认集合
    return f"ec_chunks_{tenant}"


def _create_store(tenant: str = DEFAULT_TENANT):
    """按配置创建指定租户的向量库实例。"""
    provider = env_str("VECTORSTORE_PROVIDER", "auto").lower()

    if provider == "local":
        from app.infra.vectorstore.local_store import LocalVectorStore

        logger.info(f"向量库：强制使用本地向量库（tenant={tenant}）")
        return LocalVectorStore(persist_dir=_local_persist_dir(tenant))

    if provider == "milvus":
        from app.infra.vectorstore.milvus_store import MilvusStore

        logger.info(f"向量库：强制使用 Milvus（tenant={tenant}）")
        return MilvusStore(collection=_milvus_collection(tenant))

    # auto：Milvus 可用则用，否则本地
    if milvus_config.url:
        try:
            from app.infra.vectorstore.milvus_store import MilvusStore

            store = MilvusStore(collection=_milvus_collection(tenant))
            store.client()  # 触发一次连接验证
            logger.info(f"向量库：auto 模式选择 Milvus（tenant={tenant}）")
            return store
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Milvus 不可用，降级本地向量库：{exc}")

    from app.infra.vectorstore.local_store import LocalVectorStore

    logger.info(f"向量库：auto 模式选择本地向量库（tenant={tenant}）")
    return LocalVectorStore(persist_dir=_local_persist_dir(tenant))


def get_vector_store(tenant: str = DEFAULT_TENANT):
    """获取指定租户的向量库实例（按租户缓存，互不干扰）。"""
    if tenant not in _store_instances:
        _store_instances[tenant] = _create_store(tenant)
    return _store_instances[tenant]


class VectorStoreGateway:
    """
    向量库统一门面（对齐主流向量库网关的接口风格）。

    对外只暴露业务需要的操作：混合检索、写入、按主体/ID 删除、管理查询。
    """

    def __init__(self) -> None:
        self._store = None

    @property
    def store(self):
        # 按当前请求租户解析后端（多租户隔离）；默认租户路径与旧版一致
        from app.shared.runtime.tenant import get_tenant

        return get_vector_store(get_tenant())

    @property
    def backend(self) -> str:
        """当前后端标识（milvus / local）。"""
        return getattr(self.store, "backend", "local")

    @property
    def chunks_collection(self) -> str:
        """集合名（Milvus 场景；本地库返回本地库标识）。"""
        return getattr(self.store, "chunks_collection", "ec_local_db")

    # ---------------------- 检索 ----------------------

    def create_requests(self, dense_vector, sparse_vector, *, expr: str = None, limit: int = 5):
        """构造混合检索请求（仅 Milvus 后端需要；本地库忽略）。"""
        from app.shared.clients import milvus_utils

        return milvus_utils.create_hybrid_search_requests(
            dense_vector=dense_vector, sparse_vector=sparse_vector or {}, expr=expr, limit=limit
        )

    def hybrid_search(
        self,
        *,
        dense_vector: list[float],
        query_text: str = "",
        sparse_vector: Optional[dict] = None,
        expr: Optional[str] = None,
        limit: int = 15,
        output_fields: Optional[list[str]] = None,
        dense_weight: float = 0.9,
    ) -> list[dict]:
        """
        执行混合检索，返回归一化切片列表。

        标量过滤默认关闭（retrieval_config.use_scalar_filter=False）：
        LLM 抽取的平台/类目标签常与索引值不匹配，硬过滤会把正确文档整批排除。
        """
        if not retrieval_config.use_scalar_filter:
            expr = None
        return self.store.search(
            dense_vector=dense_vector,
            query_text=query_text,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit,
            output_fields=output_fields,
            dense_weight=dense_weight,
        )

    # ---------------------- 写入 / 删除 ----------------------

    def insert_chunks(self, chunks: list[dict]) -> int:
        """批量写入切片。"""
        return self.store.add_chunks(chunks)

    def delete_by_item(self, item_name: str) -> int:
        """按主体删除全部切片。"""
        return self.store.delete_by_item(item_name)

    def delete_by_chunk_id(self, chunk_id) -> int:
        """按切片 ID 删除。"""
        return self.store.delete_by_chunk_id(int(chunk_id))

    # ---------------------- 管理查询 ----------------------

    def list_items(self, limit: int = 10000) -> list[dict]:
        """主体聚合列表（管理页）。"""
        return self.store.list_items(limit=limit)

    def get_chunks(self, item_name: str, limit: int = 20) -> list[dict]:
        """查看指定主体的切片详情。"""
        rows = self.store.query_all(
            predicate=lambda m: m.get("item_name") == item_name,
            output_fields=[
                "chunk_id", "content", "title", "file_title", "item_name",
                "platform", "market", "category", "doc_type",
            ],
            limit=limit,
        )
        return [
            {
                "chunk_id": r.get("chunk_id"),
                "content": (r.get("content") or "")[:200],
                "title": r.get("title", ""),
                "file_title": r.get("file_title", ""),
                "item_name": r.get("item_name", ""),
                "platform": r.get("platform", ""),
                "market": r.get("market", ""),
                "category": r.get("category", ""),
                "doc_type": r.get("doc_type", ""),
            }
            for r in rows
        ]

    def count(self) -> int:
        """切片总数。"""
        return self.store.count()

    def ensure_ready(self) -> None:
        """服务启动预热（加载集合/本地库）。"""
        if hasattr(self.store, "ensure_loaded"):
            self.store.ensure_loaded()
        logger.info(f"向量库就绪：backend={self.backend}, chunks={self.count()}")


vectorstore_gateway = VectorStoreGateway()
