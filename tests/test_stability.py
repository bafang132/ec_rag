"""
稳定性测试：空输入、异常输入、损坏数据、错误路径不崩溃。

验证点：
1. 空查询/空文档/空检索结果的降级行为；
2. 模型输出非 JSON 时的容错；
3. 本地向量库元数据损坏后的自恢复；
4. SSE 队列自动创建与清理；
5. 全链路图在 stub 提供方下可完整跑通。
"""
import json

import pytest


class TestEmptyAndInvalidInputs:
    """空/非法输入降级。"""

    def test_empty_query_state_raises_clear_error(self):
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.intent_service import confirm_ec_intent

        state = create_query_default_state(session_id="", original_query="")
        with pytest.raises(ValueError, match="session_id 和 original_query 不能为空"):
            confirm_ec_intent(state)

    def test_answer_generation_empty_docs(self, isolated_history, stub_llm):
        from app.process.query.agent.state import create_query_default_state
        from app.rag.query.answer_output_service import produce_answer

        state = create_query_default_state(
            session_id="s_empty", original_query="测试问题", is_stream=False
        )
        state["rewritten_query"] = "测试问题"
        state["reranked_docs"] = []
        result = produce_answer(state)
        assert result["answer"]  # 兜底生成，不崩溃

    def test_malformed_json_from_model(self, isolated_history, stub_llm):
        from app.shared.utils.ec_utils import safe_json_loads

        assert safe_json_loads("完全不是JSON") is None
        assert safe_json_loads("") is None
        # 带围栏的 JSON 可解析
        assert safe_json_loads('```json\n{"a": 1}\n```') == {"a": 1}
        # 前后有多余文本
        assert safe_json_loads('前缀文字 {"a": 1} 后缀文字') == {"a": 1}

    def test_rerank_skips_non_dict(self):
        from app.rag.query.rerank_service import merge_rrf_and_web

        merged = merge_rrf_and_web(
            [{"chunk_id": "1", "content": "x"}, "非法字符串", None],
            [{"title": "t", "snippet": "s"}],
        )
        assert len(merged) == 2  # 非字典被跳过
        assert merged[-1]["type"] == "web"

    def test_rerank_empty_all(self, stub_reranker):
        from app.rag.query.rerank_service import rerank_documents

        state = {"rewritten_query": "q", "rrf_chunks": [], "web_search_docs": []}
        assert rerank_documents(state) == []


class TestVectorStoreCorruption:
    """本地向量库损坏自恢复。"""

    def test_corrupted_meta_resets(self, tmp_path):
        from app.infra.vectorstore.local_store import LocalVectorStore

        db = tmp_path / "corrupt_db"
        db.mkdir()
        (db / "meta.json").write_text("{这不是合法JSON", encoding="utf-8")
        (db / "vectors.npy").write_bytes(b"not a numpy file")

        store = LocalVectorStore(persist_dir=db, dim=64)
        assert store.count() == 0  # 损坏后重置为空库而非崩溃
        assert store.search(dense_vector=[0.1] * 64, query_text="x") == []

    def test_meta_vector_mismatch_resets(self, tmp_path):
        import numpy as np

        from app.infra.vectorstore.local_store import LocalVectorStore

        db = tmp_path / "mismatch_db"
        db.mkdir()
        (db / "meta.json").write_text(json.dumps(
            [{"chunk_id": 1, "item_name": "A", "content": "x"}]
        ), encoding="utf-8")
        np.save(db / "vectors.npy", np.zeros((5, 64), dtype=np.float32))  # 数量不一致

        store = LocalVectorStore(persist_dir=db, dim=64)
        assert store.count() == 0  # 维度不一致，安全重置

    def test_concurrent_read_write(self, clean_vector_store):
        """并发写入不丢数据（线程安全）。"""
        import threading

        store = clean_vector_store
        provider_dummy = [0.1] * 64

        def writer(n):
            for i in range(5):
                store.add_chunks([{
                    "item_name": f"T{n}", "content": f"content {n}-{i}",
                    "dense_vector": provider_dummy,
                }])

        threads = [threading.Thread(target=writer, args=(n,)) for n in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert store.count() == 20


class TestSSEQueues:
    """SSE 队列管理。"""

    def test_auto_create_on_push(self):
        from app.shared.utils.sse_utils import (
            create_sse_queue,
            push_to_session,
            remove_sse_queue,
        )

        sid = "test_sse_auto"
        create_sse_queue(sid)
        push_to_session(sid, "delta", {"delta": "x"})
        remove_sse_queue(sid)
        # 队列删除后推送应自动重建（不丢事件）
        push_to_session(sid, "final", {"answer": "y"})
        remove_sse_queue(sid)

    def test_pack_format(self):
        from app.shared.utils.sse_utils import _sse_pack

        packed = _sse_pack("delta", {"delta": "中文"})
        assert packed.startswith("event: delta\n")
        assert "中文" in packed
        assert packed.endswith("\n\n")


class TestEndToEndGraph:
    """端到端：导入图 + 查询图在 stub 提供方下完整跑通。"""

    def test_import_graph_markdown(self, tmp_path, stub_embedding, stub_llm, stub_reranker,
                                   clean_gateway, isolated_history):
        from app.process.import_.agent.main_graph import kb_import_app
        from app.process.import_.agent.state import create_default_state

        # 构造测试 Markdown（含标题结构）
        md = tmp_path / "测试知识文档.md"
        md.write_text(
            "# 亚马逊FBA仓储规则\n\n"
            "月度仓储费按立方英尺每月收取，旺季费率上调约2-3倍。"
            "长期仓储费针对库龄超过365天的库存收取。\n\n"
            "## IPI与仓储容量\n\n"
            "IPI分数低于500将面临仓储容量限制，建议定期清理滞销库存。\n\n"
            "## 滞销清理策略\n\n"
            "降价促销、站外Deal、批量移除与线下清货渠道是常用手段。",
            encoding="utf-8",
        )
        state = create_default_state(
            task_id="e2e_import",
            local_file_path=str(md),
            local_dir=str(tmp_path / "out"),
            item_name="测试知识文档",
        )
        final = None
        for event in kb_import_app.stream(state):
            final = event
        assert final is not None
        # 取最后一个节点的状态
        last_state = list(final.values())[-1]
        assert last_state["indexed_count"] >= 1
        assert len(last_state["chunks"]) >= 1
        # 元数据写入
        assert last_state["ec_metadata"]["platform"] == "亚马逊"
        assert last_state["ec_metadata"]["category"] == "物流仓储"
        # 向量库可检索
        assert clean_gateway.count() >= 1

    def test_query_graph_end_to_end(self, clean_gateway, stub_embedding, stub_llm,
                                    stub_reranker, isolated_history):
        from app.process.query.agent.main_graph import query_app
        from app.process.query.agent.state import create_query_default_state
        from tests.conftest import StubEmbeddingProvider

        # 预置知识
        provider = StubEmbeddingProvider(dim=64)
        chunks = [{
            "item_name": "FBA仓储规则",
            "content": "月度仓储费按立方英尺每月收取，旺季费率上调。长期仓储费针对库龄超过365天的库存。",
            "title": "FBA仓储规则",
            "file_title": "亚马逊FBA入仓与仓储规则.md",
            "platform": "亚马逊",
            "category": "物流仓储",
            "dense_vector": provider.embed_documents(["月度仓储费按立方英尺每月收取"])["dense"][0],
        }]
        clean_gateway.insert_chunks(chunks)

        # 意图识别 stub：返回结构化 JSON；答案 stub：返回文本
        stub_llm.responses = [
            json.dumps({
                "rewritten_query": "亚马逊FBA月度仓储费如何计算",
                "platform": "亚马逊", "market": "北美",
                "category": "物流仓储", "doc_type": "实操教程",
                "needs_clarification": False,
            }, ensure_ascii=False),
            "【测试答案】FBA月度仓储费按立方英尺每月收取，旺季费率上调。来源：《FBA仓储规则》",
        ]

        state = create_query_default_state(
            session_id="e2e_query",
            original_query="亚马逊FBA仓储费怎么算？",
            is_stream=False,
            role="seller",
        )
        result = query_app.invoke(state)
        assert result["answer"]
        assert "仓储费" in result["answer"]
        assert result["ec_filters"]["platform"] == "亚马逊"
        # 检索链路有结果
        assert result["embedding_chunks"] or result["reranked_docs"]

    def test_query_graph_greeting_shortcut(self, isolated_history):
        from app.process.query.agent.main_graph import query_app
        from app.process.query.agent.state import create_query_default_state

        state = create_query_default_state(
            session_id="e2e_greet", original_query="你好", is_stream=False
        )
        result = query_app.invoke(state)
        assert result["answer"]  # 寒暄直达答案
        assert "跨境电商" in result["answer"]
