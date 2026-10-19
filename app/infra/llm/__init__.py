"""
LLM 提供者模块：统一封装聊天模型、Embedding 与 Reranker 的访问方式。

接口与各 LLM 提供方保持一致：
- chat(model=None, json_mode=False)      -> LLM 客户端（invoke/stream）
- vision_chat()                          -> 视觉模型客户端
- embedding_model()                      -> Embedding 提供方
- reranker_model()                       -> Reranker 实例
- embed_documents(texts)                 -> {"dense": [...], "sparse": [...] 或 None}
"""
from typing import Optional

from app.infra.config import infra_config
from app.shared.model.embedding_utils import generate_embeddings, get_embedding_provider
from app.shared.model.lm_utils import get_llm_client
from app.shared.model.reranker_utils import get_reranker_model
from app.shared.runtime.logger import logger


class LLMProvider:
    """LLM / Embedding / Reranker 统一访问门面。"""

    def chat(self, model: Optional[str] = None, json_mode: bool = False):
        """获取聊天模型客户端（invoke/stream 接口）。"""
        return get_llm_client(model=model, json_mode=json_mode)

    def vision_chat(self):
        """获取视觉模型客户端（文档图片摘要）。"""
        return get_llm_client(model=infra_config.llm.vl_model)

    def embedding_model(self):
        """获取 Embedding 提供方实例。"""
        return get_embedding_provider()

    def reranker_model(self):
        """获取 Reranker 实例（compute_score/tokenizer 接口）。"""
        return get_reranker_model()

    def embed_documents(self, texts: list[str]) -> dict:
        """为文本列表生成向量（dense + sparse，稀疏可能为 None）。"""
        return generate_embeddings(texts)

    @property
    def embedding_dim(self) -> int:
        """当前 Embedding 提供方的向量维度。"""
        try:
            return get_embedding_provider().dim
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"获取 Embedding 维度失败，使用配置值：{exc}")
            return infra_config.embedding.dim


llm_provider = LLMProvider()
