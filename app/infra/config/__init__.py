"""
基础设施配置聚合模块：统一暴露各组件配置，接口风格与各网关保持一致。
"""
from app.shared.config.embedding_config import embedding_config
from app.shared.config.lm_config import lm_config
from app.shared.config.milvus_config import milvus_config
from app.shared.config.mineru_config import mineru_config
from app.shared.config.minio_config import minio_config
from app.shared.config.mongo_config import mongo_config
from app.shared.config.reranker_config import reranker_config
from app.shared.config.retrieval_config import retrieval_config
from app.shared.config.settings_config import settings


class InfraConfig:
    """基础设施配置聚合。"""

    llm = lm_config
    embedding = embedding_config
    reranker = reranker_config
    retrieval = retrieval_config
    milvus = milvus_config
    minio = minio_config
    mineru = mineru_config
    mongo = mongo_config
    settings = settings


infra_config = InfraConfig()
