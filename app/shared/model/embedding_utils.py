"""
Embedding 工具模块：多提供方统一接口。

提供方（EMBEDDING_PROVIDER 控制）：
- bge   : 本地 BGE-M3（FlagEmbedding），dense(1024) + sparse 混合向量，检索质量最佳
- zhipu : 智谱 embedding-2 API，仅 dense 向量（Milvus 场景自动合成 hashing 稀疏向量）
- hash  : 开发/测试用确定性哈希向量（离线联调，生产禁用）
- auto  : bge 可用 → zhipu 可用 → hash，逐级降级

统一返回格式：{"dense": [[...], ...], "sparse": [{idx: val}, ...] or None}
"""
import hashlib
import math
import re
from typing import Optional

import numpy as np

from app.shared.config.embedding_config import embedding_config
from app.shared.model.device_utils import resolve_device, resolve_fp16
from app.shared.runtime.logger import logger

# 简单中英分词（用于 hashing 稀疏向量与 BM25 关键词打分）
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    """中英混合分词：英文数字按词、中文按字。"""
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.lower())


class BaseEmbeddingProvider:
    """Embedding 提供方统一接口。"""

    name = "base"
    supports_sparse = False

    def embed_documents(self, texts: list[str]) -> dict:
        raise NotImplementedError

    def embed_query(self, text: str) -> dict:
        return self.embed_documents([text])

    @property
    def dim(self) -> int:
        return embedding_config.dim


class BGEM3Provider(BaseEmbeddingProvider):
    """本地 BGE-M3：dense + sparse 混合向量（FlagEmbedding）。"""

    name = "bge-m3"
    supports_sparse = True

    def __init__(self) -> None:
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction

        model_name = embedding_config.bge_m3_path or embedding_config.bge_m3
        device = resolve_device(embedding_config.bge_device, default="cpu")
        use_fp16 = resolve_fp16(embedding_config.bge_fp16, device)
        logger.info(
            f"初始化 BGE-M3：model={model_name}, device={device}, fp16={use_fp16}"
        )
        self._model = BGEM3EmbeddingFunction(
            model_name=model_name,
            device=device,
            use_fp16=use_fp16,
            normalize_embeddings=True,
        )
        self._dim = 1024
        logger.success("BGE-M3 初始化成功（dense 1024 + sparse，L2 归一化）")

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> dict:
        if not texts:
            return {"dense": [], "sparse": []}
        embeddings = self._model.encode_documents(texts)
        sparse_list: list[dict] = []
        for i in range(len(texts)):
            indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]: embeddings["sparse"].indptr[i + 1]
            ].tolist()
            data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]: embeddings["sparse"].indptr[i + 1]
            ].tolist()
            sparse_list.append({int(k): float(v) for k, v in zip(indices, data)})
        return {
            "dense": [emb.tolist() for emb in embeddings["dense"]],
            "sparse": sparse_list,
        }


class ZhipuEmbeddingProvider(BaseEmbeddingProvider):
    """智谱 embedding-2 API（OpenAI 兼容协议），仅 dense 向量。"""

    name = "zhipu-embedding-2"
    supports_sparse = False

    def __init__(self) -> None:
        from openai import OpenAI

        if not embedding_config.zhipu_api_key:
            raise ValueError("智谱 Embedding 需要 ZHIPU_API_KEY")
        self._client = OpenAI(
            api_key=embedding_config.zhipu_api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
        )
        self._model = embedding_config.zhipu_embedding_model
        self._dim = 1024
        logger.success(f"智谱 Embedding 初始化成功：model={self._model}")

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> dict:
        if not texts:
            return {"dense": [], "sparse": None}
        # 智谱单次请求条数限制 64，分批调用避免限流（分批调用实践结论）
        dense_all: list[list[float]] = []
        batch_size = 60
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            dense_all.extend([item.embedding for item in resp.data])
        return {"dense": dense_all, "sparse": None}


class HashEmbeddingProvider(BaseEmbeddingProvider):
    """
    开发测试用确定性哈希向量：同一文本永远得到同一向量，
    离线联调检索链路时无需任何 API/模型。生产环境禁用。
    """

    name = "hash-dev"
    supports_sparse = False

    def __init__(self, dim: Optional[int] = None) -> None:
        self._dim = dim or embedding_config.dim
        logger.warning(
            "使用哈希向量提供方（仅限开发联调）：向量无语义，检索结果不具备真实语义相关性"
        )

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> dict:
        dense = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            tokens = tokenize(text)
            if not tokens:
                dense.append(vec.tolist())
                continue
            # token 哈希到维度桶 + 词频权重，再 L2 归一化
            counts: dict[int, float] = {}
            for tok in tokens:
                bucket = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
            for bucket, weight in counts.items():
                vec[bucket] = weight / norm
            dense.append(vec.tolist())
        return {"dense": dense, "sparse": None}


def hash_sparse_vector(text: str, dim: int = 30000) -> dict:
    """
    基于分词 hashing 的稀疏向量（dense-only 提供方在 Milvus 混合检索场景下使用）。

    :param text: 输入文本
    :param dim: 稀疏向量维度（词表大小）
    :return: {索引: 权重} 字典
    """
    counts: dict[int, float] = {}
    for tok in tokenize(text):
        idx = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % dim
        counts[idx] = counts.get(idx, 0.0) + 1.0
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {k: v / norm for k, v in counts.items()}


# ===================== 提供方工厂（单例 + 自动降级） =====================

_provider_instance: Optional[BaseEmbeddingProvider] = None


def _try_build(provider: str) -> Optional[BaseEmbeddingProvider]:
    """尝试实例化指定提供方，失败返回 None。"""
    try:
        if provider == "bge":
            return BGEM3Provider()
        if provider == "zhipu":
            return ZhipuEmbeddingProvider()
        if provider == "hash":
            return HashEmbeddingProvider()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Embedding 提供方 [{provider}] 初始化失败：{exc}")
    return None


def get_embedding_provider() -> BaseEmbeddingProvider:
    """
    获取 Embedding 提供方单例（auto 模式逐级降级）。

    :return: 可用的 Embedding 提供方
    """
    global _provider_instance
    if _provider_instance is not None:
        return _provider_instance

    configured = embedding_config.provider
    if configured in ("bge", "zhipu", "hash"):
        provider = _try_build(configured)
        if provider is None:
            raise RuntimeError(
                f"Embedding 提供方 [{configured}] 初始化失败，请检查配置与依赖"
            )
        _provider_instance = provider
        return provider

    # auto：bge -> zhipu -> hash
    for candidate in ("bge", "zhipu", "hash"):
        provider = _try_build(candidate)
        if provider is not None:
            logger.info(f"Embedding auto 模式选择提供方：{provider.name}")
            _provider_instance = provider
            return provider
    raise RuntimeError("无可用 Embedding 提供方（bge/zhipu/hash 均初始化失败）")


def generate_embeddings(texts: list[str]) -> dict:
    """
    为文本列表生成向量（统一入口）。

    :param texts: 待向量化文本列表
    :return: {"dense": [[...]], "sparse": [{...}] 或 None}
    """
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("参数 texts 必须是非空字符串列表")
    if any(not isinstance(t, str) for t in texts):
        raise ValueError("texts 中存在非字符串内容")
    provider = get_embedding_provider()
    result = provider.embed_documents(texts)
    # dense-only 提供方统一补 None，保证下游结构一致
    if result.get("sparse") is None:
        result["sparse"] = None
    return result


if __name__ == "__main__":
    out = generate_embeddings(["亚马逊FBA运费怎么计算？", "欧洲VAT税率是多少？"])
    print(f"provider={get_embedding_provider().name}, dim={len(out['dense'][0])}, "
          f"sparse={'有' if out['sparse'] else '无'}")
