"""
Milvus 客户端工具模块：连接管理 + 混合检索（dense + sparse）。

集合 schema 保持通用并扩展跨境电商元数据字段
（platform / market / category / doc_type）。
"""
from typing import Any, Optional

from app.shared.config.milvus_config import milvus_config
from app.shared.runtime.logger import logger

_client = None

# 跨境电商元数据字段（导入时显式声明，供过滤检索）
EC_METADATA_FIELDS = [
    "platform", "market", "category", "doc_type",
    "article_title", "author", "source_file", "source_path",
]


def get_milvus_client():
    """获取 Milvus 客户端单例（延迟连接）。"""
    global _client
    if _client is not None:
        return _client
    if not milvus_config.url:
        raise ValueError("未配置 MILVUS_URL，无法连接 Milvus")
    from pymilvus import MilvusClient

    _client = MilvusClient(uri=milvus_config.url, db_name=milvus_config.db_name or "default")
    logger.success(f"Milvus 连接成功：{milvus_config.url}")
    return _client


def create_hybrid_search_requests(dense_vector, sparse_vector, *, expr: str = None, limit: int = 5):
    """
    构造 Milvus 混合检索请求（dense + sparse 两路）。

    :param dense_vector: 稠密向量（list）
    :param sparse_vector: 稀疏向量（dict[idx, val]）
    :param expr: 过滤表达式
    :param limit: 单路返回条数
    :return: AnnSearchRequest / RRFRanker 组合的请求列表
    """
    from pymilvus import AnnSearchRequest, RRFRanker

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "IP", "params": {}},
        limit=limit,
        expr=expr,
    )
    if sparse_vector:
        sparse_req = AnnSearchRequest(
            data=[sparse_vector],
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}},
            limit=limit,
            expr=expr,
        )
        return [dense_req, sparse_req]
    return [dense_req]


def hybrid_search(*, client, collection_name: str, reqs: list, ranker_weights=(0.9, 0.1),
                  norm_score: bool = False, limit: int = 5,
                  output_fields: Optional[list[str]] = None, search_params: dict | None = None):
    """
    执行 Milvus 混合检索（RRF 融合多路结果）。

    :return: Milvus 原始检索结果（二维列表）
    """
    from pymilvus import RRFRanker

    ranker = RRFRanker(k=60)
    results = client.hybrid_search(
        collection_name=collection_name,
        reqs=reqs,
        ranker=ranker,
        limit=limit,
        output_fields=output_fields,
        **(search_params or {}),
    )
    return results


def create_chunks_collection(client, collection_name: str, dim: int = 1024) -> None:
    """
    创建知识切块集合（含跨境电商元数据字段 + 混合向量索引）。

    已存在时跳过（Milvus 不支持给已有集合追加字段）。
    """
    from pymilvus import DataType

    if client.has_collection(collection_name=collection_name):
        logger.info(f"集合[{collection_name}]已存在，跳过创建")
        return

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="part", datatype=DataType.INT16)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    for field_name in EC_METADATA_FIELDS:
        schema.add_field(field_name=field_name, datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense_vector", index_type="AUTOINDEX",
                           metric_type="IP", params={"nlist": 128})
    index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX",
                           metric_type="IP", params={"inverted_index_algo": "DAAT_MAXSCORE"})
    client.create_collection(collection_name=collection_name, schema=schema, index_params=index_params)
    logger.success(f"集合[{collection_name}]创建成功（dim={dim}）")
