"""
Milvus 配置模块：向量库地址与集合名。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class MilvusConfig:
    url: str = env_str("MILVUS_URL", "")
    chunks_collection: str = env_str("CHUNKS_COLLECTION", "ec_kb_chunks")
    item_name_collection: str = env_str("ITEM_NAME_COLLECTION", "ec_kb_item_names")
    db_name: str = env_str("MILVUS_DB", "default")
    embedding_dim: int = 1024


milvus_config = MilvusConfig()
