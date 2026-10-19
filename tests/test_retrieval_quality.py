"""
检索质量测试：本地向量库混合检索、RRF 融合、重排服务。

验证点：
1. 本地向量库 写入/检索/删除/聚合 全链路；
2. 稠密 + BM25 两路融合下语义相关文档排前；
3. RRF 融合算法数学正确性；
4. 重排服务打分、排序、动态截断与缓存。
"""
import math

import pytest

from tests.conftest import StubEmbeddingProvider


def _make_chunks(store, items: list[tuple]):
    """把 (item_name, content[, extra_meta]) 元组列表写入向量库。"""
    provider = StubEmbeddingProvider(dim=64)
    chunks = []
    for item in items:
        item_name, content = item[0], item[1]
        extra = item[2] if len(item) > 2 else {}
        emb = provider.embed_documents([content])["dense"][0]
        chunk = {
            "item_name": item_name,
            "content": content,
            "title": item_name,
            "file_title": f"{item_name}.md",
            "dense_vector": emb,
            **extra,
        }
        chunks.append(chunk)
    store.add_chunks(chunks)
    return chunks


class TestLocalVectorStore:
    """本地向量库基础能力。"""

    def test_add_and_count(self, clean_vector_store):
        store = clean_vector_store
        assert store.count() == 0
        assert store.is_empty()
        _make_chunks(store, [
            ("FBA仓储", "亚马逊FBA月度仓储费按立方英尺收取，旺季上调", {"platform": "亚马逊"}),
            ("VAT", "欧洲VAT税率为20%，需注册税号申报", {"platform": "亚马逊"}),
        ])
        assert store.count() == 2
        assert not store.is_empty()

    def test_hybrid_search_ranks_relevant_first(self, clean_vector_store):
        store = clean_vector_store
        _make_chunks(store, [
            ("FBA仓储", "FBA monthly storage fee per cubic foot peak season surcharge"),
            ("VAT指南", "VAT tax rate europe declaration tax number"),
            ("Listing", "listing title bullet points keywords optimization"),
        ])
        provider = StubEmbeddingProvider(dim=64)
        q = provider.embed_documents(["FBA storage fee"])["dense"][0]
        results = store.search(dense_vector=q, query_text="FBA storage fee", limit=2)
        assert len(results) == 2
        assert results[0]["item_name"] == "FBA仓储"
        assert results[0]["score"] >= results[1]["score"]

    def test_delete_by_item(self, clean_vector_store):
        store = clean_vector_store
        _make_chunks(store, [("A", "content a"), ("B", "content b")])
        assert store.delete_by_item("A") == 1
        assert store.count() == 1
        assert store.delete_by_item("不存在") == 0

    def test_delete_all_leaves_clean_state(self, clean_vector_store):
        store = clean_vector_store
        _make_chunks(store, [("A", "content a"), ("B", "content b")])
        store.delete_by_item("A")
        store.delete_by_item("B")
        assert store.count() == 0
        assert store.is_empty()
        # 空库检索返回空
        provider = StubEmbeddingProvider(dim=64)
        q = provider.embed_documents(["任意"])["dense"][0]
        assert store.search(dense_vector=q, query_text="任意", limit=5) == []

    def test_persistence_roundtrip(self, tmp_path):
        from app.infra.vectorstore.local_store import LocalVectorStore

        store1 = LocalVectorStore(persist_dir=tmp_path / "persist_db", dim=64)
        _make_chunks(store1, [("持久化测试", "persistence roundtrip content")])
        assert store1.count() == 1

        # 模拟重启：重新加载
        store2 = LocalVectorStore(persist_dir=tmp_path / "persist_db", dim=64)
        assert store2.count() == 1
        assert store2.query_all()[0]["item_name"] == "持久化测试"

    def test_list_items_aggregation(self, clean_vector_store):
        store = clean_vector_store
        _make_chunks(store, [
            ("FBA", "chunk one", {"platform": "亚马逊", "category": "物流仓储"}),
            ("FBA", "chunk two", {"platform": "亚马逊", "category": "物流仓储"}),
            ("VAT", "chunk three", {"platform": "亚马逊", "category": "税务与财务"}),
        ])
        items = {i["item_name"]: i for i in store.list_items()}
        assert items["FBA"]["chunk_count"] == 2
        assert items["FBA"]["platform"] == "亚马逊"
        assert items["VAT"]["category"] == "税务与财务"

    def test_expr_scalar_filter(self, clean_vector_store):
        store = clean_vector_store
        _make_chunks(store, [
            ("A", "amazon fba logistics", {"platform": "亚马逊"}),
            ("B", "shopee southeast asia", {"platform": "Shopee"}),
        ])
        provider = StubEmbeddingProvider(dim=64)
        q = provider.embed_documents(["amazon"])["dense"][0]
        results = store.search(dense_vector=q, expr="platform == '亚马逊'", limit=5)
        assert all(r["platform"] == "亚马逊" for r in results)
        assert len(results) == 1

    def test_dimension_mismatch_rejected(self, clean_vector_store):
        store = clean_vector_store  # dim=64
        bad = [{"item_name": "X", "content": "x", "dense_vector": [0.1] * 32}]
        assert store.add_chunks(bad) == 0
        assert store.count() == 0


class TestRRFFusion:
    """RRF 融合算法。"""

    def test_basic_fusion_order(self):
        from app.rag.query.rrf_service import reciprocal_rank_fusion

        list1 = [{"chunk_id": "a", "content": "x"}, {"chunk_id": "b", "content": "y"}]
        list2 = [{"chunk_id": "b", "content": "y"}, {"chunk_id": "c", "content": "z"}]
        fused = reciprocal_rank_fusion([(list1, 1.0), (list2, 1.0)], top=3)
        # b 在两路都排前，应第一
        assert fused[0]["chunk_id"] == "b"
        ids = [d["chunk_id"] for d in fused]
        assert set(ids) == {"a", "b", "c"}

    def test_rrf_score_formula(self):
        from app.rag.query.rrf_service import reciprocal_rank_fusion

        list1 = [{"chunk_id": "a", "content": "x"}]
        list2 = [{"chunk_id": "a", "content": "x"}]
        fused = reciprocal_rank_fusion([(list1, 1.0), (list2, 1.0)], k=60, top=1)
        # 两路都排第 1：2 * 1/(60+1)
        assert math.isclose(fused[0]["score"], 2 * (1 / 61), rel_tol=1e-6)

    def test_empty_inputs(self):
        from app.rag.query.rrf_service import reciprocal_rank_fusion

        assert reciprocal_rank_fusion([([], 1.0), ([], 1.0)]) == []


class TestRerankService:
    """重排服务（Stub Reranker）。"""

    def _state(self, query="FBA仓储费"):
        return {"rewritten_query": query, "original_query": query, "session_id": "t"}

    def test_merge_and_score(self, stub_reranker):
        from app.rag.query.rerank_service import rerank_documents

        state = self._state("FBA storage fee")
        state["rrf_chunks"] = [
            {"chunk_id": "1", "item_name": "FBA", "title": "FBA仓储",
             "content": "FBA storage fee monthly charge", "score": 0.5},
            {"chunk_id": "2", "item_name": "VAT", "title": "VAT",
             "content": "tax rate declaration", "score": 0.3},
        ]
        state["web_search_docs"] = []
        docs = rerank_documents(state)
        assert len(docs) >= 2
        assert docs[0]["title"] == "FBA仓储"  # 词重叠更高
        assert docs[0]["score"] >= docs[1]["score"]

    def test_dynamic_topk_truncation(self, stub_reranker):
        from app.rag.query.rerank_service import dynamic_topk

        # 前 5 个高分（0.9→0.7），第 6 个断崖下降到 0.1 → 截断到 5
        docs = [{"score": s} for s in (0.9, 0.85, 0.8, 0.75, 0.7, 0.1)]
        topk = dynamic_topk(docs)
        assert len(topk) == 5

    def test_dynamic_topk_keeps_all_when_flat(self, stub_reranker):
        from app.rag.query.rerank_service import dynamic_topk

        # 分数平缓下降，无断崖 → 保留全部
        docs = [{"score": s} for s in (0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6)]
        topk = dynamic_topk(docs)
        assert len(topk) == 7

    def test_cache_hit(self, stub_reranker):
        from app.rag.query import rerank_service

        state = self._state("缓存测试 query")
        state["rrf_chunks"] = [{"chunk_id": "1", "title": "T", "content": "缓存测试 query content"}]
        state["web_search_docs"] = []
        first = rerank_service.rerank_documents(state)
        # 相同输入再次调用应命中缓存（结果一致）
        second = rerank_service.rerank_documents(state)
        assert [d.get("score") for d in first] == [d.get("score") for d in second]
