"""
性能测试：本地向量库检索延迟、批量写入吞吐、分块吞吐量。

阈值宽松（CI 环境差异大），主要验证没有量级异常（如 O(n²) 查找、重复加载）。
"""
import time

import pytest

from tests.conftest import StubEmbeddingProvider


class TestLocalStorePerformance:
    """本地向量库性能。"""

    def _prepare(self, store, n: int = 300):
        provider = StubEmbeddingProvider(dim=64)
        chunks = []
        for i in range(n):
            content = f"亚马逊FBA仓储费规则第{i}条：月度仓储费按立方英尺收取，IPI分数与滞销清理策略"
            chunks.append({
                "item_name": f"文档{i}",
                "content": content,
                "title": f"标题{i}",
                "file_title": f"文档{i}.md",
                "dense_vector": provider.embed_documents([content])["dense"][0],
            })
        store.add_chunks(chunks)

    def test_bulk_insert(self, clean_vector_store):
        store = clean_vector_store
        provider = StubEmbeddingProvider(dim=64)
        chunks = [{
            "item_name": f"D{i}",
            "content": f"content {i}",
            "dense_vector": provider.embed_documents([f"content {i}"])["dense"][0],
        } for i in range(300)]
        t0 = time.time()
        count = store.add_chunks(chunks)
        elapsed = time.time() - t0
        assert count == 300
        assert elapsed < 30, f"批量写入 300 条耗时 {elapsed:.1f}s，异常"

    def test_search_latency(self, clean_vector_store):
        store = clean_vector_store
        self._prepare(store, n=300)
        provider = StubEmbeddingProvider(dim=64)
        q = provider.embed_documents(["FBA仓储费 IPI"])["dense"][0]

        # 预热（含持久化）
        store.search(dense_vector=q, query_text="FBA仓储费 IPI", limit=10)

        t0 = time.time()
        for _ in range(20):
            results = store.search(dense_vector=q, query_text="FBA仓储费 IPI", limit=10)
        avg_ms = (time.time() - t0) / 20 * 1000
        assert len(results) == 10
        assert avg_ms < 2000, f"平均检索耗时 {avg_ms:.0f}ms，异常"

    def test_persist_and_reload(self, tmp_path):
        from app.infra.vectorstore.local_store import LocalVectorStore

        db = tmp_path / "perf_db"
        store = LocalVectorStore(persist_dir=db, dim=64)
        self._prepare(store, n=200)

        t0 = time.time()
        store2 = LocalVectorStore(persist_dir=db, dim=64)
        reload_s = time.time() - t0
        assert store2.count() == 200
        assert reload_s < 10, f"重载 200 条耗时 {reload_s:.1f}s，异常"


class TestSplitPerformance:
    """分块性能。"""

    def test_large_document_split(self):
        from app.shared.utils.text_split import split_text

        # 100KB 文档
        large = ("亚马逊FBA仓储费规则：月度仓储费按立方英尺每月收取，旺季费率上调。\n\n" * 800)
        t0 = time.time()
        chunks = split_text(large, chunk_size=600, chunk_overlap=50)
        elapsed = time.time() - t0
        assert len(chunks) > 10
        assert elapsed < 10, f"100KB 文档分块耗时 {elapsed:.1f}s，异常"

    def test_split_result_sizes(self):
        from app.shared.utils.text_split import split_text

        large = ("一二三四五六七八九十" * 2000)
        chunks = split_text(large, chunk_size=500, chunk_overlap=50)
        # 绝大多数块应在阈值附近（允许硬切边界）
        sizes = [len(c) for c in chunks]
        assert max(sizes) <= 520
        assert sum(sizes) > len(large) * 0.9  # 内容基本保留（去空白后）


class TestEmbeddingPerformance:
    """哈希向量提供方吞吐（开发环境基准）。"""

    def test_hash_embedding_batch(self):
        provider = StubEmbeddingProvider(dim=1024)
        texts = [f"亚马逊FBA规则测试文本 {i}" for i in range(64)]
        t0 = time.time()
        result = provider.embed_documents(texts)
        elapsed = time.time() - t0
        assert len(result["dense"]) == 64
        assert elapsed < 5, f"64 条哈希向量耗时 {elapsed:.1f}s，异常"

    def test_embedding_deterministic(self):
        """相同文本必须得到相同向量（哈希提供方的确定性）。"""
        provider = StubEmbeddingProvider(dim=64)
        v1 = provider.embed_documents(["测试文本"])["dense"][0]
        v2 = provider.embed_documents(["测试文本"])["dense"][0]
        assert v1 == v2
