"""
pytest 公共 fixtures：隔离的本地向量库、Stub Embedding/LLM/Reranker。

设计目标：测试不依赖 Milvus / BGE-M3 / 真实 LLM API，
在任意环境（含 CI）可完整运行检索、导入、问答链路测试。
"""
import sys
from pathlib import Path

import pytest

# 保证项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 测试环境：强制本地向量库 + 哈希向量（开发提供方），隔离持久化目录
import os

os.environ["VECTORSTORE_PROVIDER"] = "local"
os.environ["LOCAL_VECTOR_DIR"] = str(PROJECT_ROOT / "data" / "vector_store" / "test_db")
os.environ["EMBEDDING_PROVIDER"] = "hash"


@pytest.fixture()
def clean_vector_store(tmp_path):
    """每个测试独立的本地向量库实例（不影响默认库）。"""
    from app.infra.vectorstore.local_store import LocalVectorStore

    store = LocalVectorStore(persist_dir=tmp_path / "vs", dim=64)
    yield store
    # 清理：tmp_path 由 pytest 自动回收


@pytest.fixture()
def clean_gateway(monkeypatch, tmp_path):
    """指向临时目录的向量库门面（隔离测试，默认租户）。"""
    import app.infra.vectorstore as vs_package
    from app.infra.vectorstore.local_store import LocalVectorStore

    store = LocalVectorStore(persist_dir=tmp_path / "gateway_db", dim=64)
    # 多租户网关按租户缓存实例：把默认租户指向临时库
    monkeypatch.setattr(vs_package, "_store_instances", {"default": store}, raising=False)
    yield vs_package.vectorstore_gateway


class StubEmbeddingProvider:
    """确定性哈希向量提供方（小维度，加速测试）。"""

    name = "stub-hash"
    supports_sparse = False

    def __init__(self, dim: int = 64):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_documents(self, texts: list[str]) -> dict:
        import hashlib
        import math

        import numpy as np

        dense = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            tokens = text.lower().split()
            if not tokens:
                dense.append(vec.tolist())
                continue
            counts: dict[int, float] = {}
            for tok in tokens:
                bucket = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16) % self._dim
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
            for bucket, weight in counts.items():
                vec[bucket] = weight / norm
            dense.append(vec.tolist())
        return {"dense": dense, "sparse": None}


class StubLLMClient:
    """脚本化 LLM 客户端：按调用顺序返回预设响应。"""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls: list[object] = []

    def invoke(self, messages, **kwargs):
        self.calls.append(messages)
        if self.responses:
            content = self.responses.pop(0)
        else:
            content = "【测试答案】基于知识库的回答。"

        class _Resp:
            pass

        resp = _Resp()
        resp.content = content
        return resp

    def stream(self, messages, **kwargs):
        content = self.invoke(messages, **kwargs).content
        for ch in content:
            yield type("_Chunk", (), {"content": ch})()


class StubReranker:
    """基于词重叠的确定性重排器。"""

    name = "stub-reranker"

    def __init__(self):
        self._tokenizer = _StubTokenizer()

    @property
    def tokenizer(self):
        return self._tokenizer

    def compute_score(self, pairs: list[list[str]], normalize: bool = True) -> list[float]:
        scores = []
        for query, doc in pairs:
            q = set(query.lower().split())
            d = set((doc or "").lower().split())
            if not q or not d:
                scores.append(0.0)
                continue
            overlap = len(q & d) / max(1, len(q))
            scores.append(min(1.0, overlap * 2))
        return scores


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return list(text or "")


@pytest.fixture()
def stub_embedding(monkeypatch):
    """把 embedding 提供方替换为 Stub（64 维）。"""
    from app.shared.model import embedding_utils

    provider = StubEmbeddingProvider(dim=64)
    monkeypatch.setattr(embedding_utils, "_provider_instance", provider, raising=False)
    monkeypatch.setattr(embedding_utils, "get_embedding_provider", lambda: provider)
    return provider


@pytest.fixture()
def stub_llm(monkeypatch):
    """把 LLM 客户端替换为脚本化 Stub（同时 patch 门面与工具层）。"""
    from app.infra.llm import llm_provider
    from app.shared.model import lm_utils

    client = StubLLMClient()
    monkeypatch.setattr(lm_utils, "get_llm_client", lambda *a, **k: client)
    # 门面层持有模块级引用，直接 patch 提供者对象的 chat 方法
    monkeypatch.setattr(llm_provider, "chat", lambda *a, **k: client)
    monkeypatch.setattr(llm_provider, "vision_chat", lambda *a, **k: client)
    return client


@pytest.fixture()
def stub_reranker(monkeypatch):
    """把 Reranker 替换为 Stub（同时 patch 门面与工具层）。"""
    from app.infra.llm import llm_provider
    from app.shared.model import reranker_utils

    reranker = StubReranker()
    monkeypatch.setattr(reranker_utils, "_reranker_instance", reranker, raising=False)
    monkeypatch.setattr(reranker_utils, "get_reranker_model", lambda: reranker)
    monkeypatch.setattr(llm_provider, "reranker_model", lambda: reranker)
    return reranker


@pytest.fixture()
def isolated_history(monkeypatch, tmp_path):
    """隔离的历史存储（临时 JSON 文件；同时 patch 模块级引用）。"""
    import app.infra.persistence as persistence_module
    from app.shared.clients import history_store as hs_module

    monkeypatch.setattr(hs_module, "_LOCAL_HISTORY_PATH", tmp_path / "history.json", raising=False)
    monkeypatch.setattr(hs_module, "_store_instance", None, raising=False)
    monkeypatch.setattr(hs_module, "get_history_store", hs_module.LocalHistoryStore)
    # 仓储模块持有模块级引用，需同步 patch 模块属性（不是实例属性）
    monkeypatch.setattr(persistence_module, "get_history_store", hs_module.LocalHistoryStore)
    return hs_module.LocalHistoryStore()
