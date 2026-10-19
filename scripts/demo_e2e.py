"""
端到端演示脚本：导入全部示例知识文档 → 检索 → 问答 → 测评（抽样）。

用途：验证系统在「无 LLM API / 无 Milvus / 无 BGE 模型」的最小环境下，
全链路（导入图 + 查询图 + 测评器）可跑通，且各降级路径工作正常。

运行：python scripts/demo_e2e.py
"""
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

# 强制最小环境：本地向量库 + 哈希向量
os.environ["VECTORSTORE_PROVIDER"] = "local"
os.environ["EMBEDDING_PROVIDER"] = "hash"

from app.shared.runtime.logger import logger  # noqa: E402

# ---------------- Stub LLM：模拟「模型不可用」→ 验证规则兜底路径 ----------------

class _StubResp:
    def __init__(self, content):
        self.content = content


class _StubChat:
    """元数据抽取返回不可解析文本（走规则兜底）；答案生成返回固定文本。"""

    def invoke(self, messages, **kwargs):
        text = str(messages)
        if "亚马逊FBA入仓与仓储规则" in text or "元数据" in text:
            return _StubResp("（模型不可用，返回非 JSON 文本）")
        return _StubResp(
            "【结论】根据知识库规则：FBA 月度仓储费按立方英尺每月收取，旺季费率上调。\n\n"
            "【操作步骤】1. 清理滞销库存提升 IPI；2. 分批发货避免集中入库；3. 库龄临近 365 天评估移除。\n\n"
            "【风险提示】暂无特别风险提示。\n\n"
            "【参考来源】- 《亚马逊FBA入仓与仓储规则》"
        )

    def stream(self, messages, **kwargs):
        yield _StubResp(self.invoke(messages).content)


class _StubLLMProvider:
    def chat(self, *args, **kwargs):
        return _StubChat()

    def vision_chat(self, *args, **kwargs):
        return _StubChat()

    def reranker_model(self):
        return _StubReranker()

    def embed_documents(self, texts):
        from app.shared.model.embedding_utils import generate_embeddings

        return generate_embeddings(texts)


class _StubReranker:
    """词重叠重排器（与 tests/conftest.py 的 StubReranker 同策略）。"""

    name = "demo-stub"

    @property
    def tokenizer(self):
        return _StubTokenizer()

    def compute_score(self, pairs, normalize=True):
        scores = []
        for query, doc in pairs:
            q = set(query.lower().split())
            d = set((doc or "").lower().split())
            overlap = len(q & d) / max(1, len(q)) if q and d else 0.0
            scores.append(min(1.0, overlap * 2))
        return scores


class _StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return list(text or "")


def patch_llm():
    import app.infra.llm as llm_mod

    llm_mod.llm_provider = _StubLLMProvider()


def main() -> None:
    patch_llm()

    from app.process.import_.agent.main_graph import kb_import_app
    from app.process.import_.agent.state import create_default_state

    doc_dir = PROJECT_ROOT / "app" / "source" / "doc"
    docs = sorted(f for f in os.listdir(doc_dir) if f.endswith(".md"))
    print(f"========== 第一步：导入 {len(docs)} 篇知识文档 ==========")
    total_chunks = 0
    for doc in docs:
        path = str(doc_dir / doc)
        state = create_default_state(
            task_id=f"demo_{doc}",
            local_file_path=path,
            local_dir=str(PROJECT_ROOT / "output" / "demo"),
            item_name=doc[:-3],
        )
        final = None
        for event in kb_import_app.stream(state):
            final = event
        last = list(final.values())[-1]
        total_chunks += last.get("indexed_count", 0)
        meta = last.get("ec_metadata", {})
        print(f"  OK {doc[:-3][:22]:24s} 切片x{last.get('indexed_count'):<3d} "
              f"平台:{meta.get('platform')}  类目:{meta.get('category')}")
    print(f"导入完成：{len(docs)} 篇文档，共 {total_chunks} 个切片\n")

    # ---------------- 第二步：问答 ----------------
    print("========== 第二步：问答链路（stub LLM） ==========")
    from app.process.query.agent.main_graph import query_app
    from app.process.query.agent.state import create_query_default_state

    questions = [
        "亚马逊FBA仓储费怎么计算？滞销库存太多怎么办？",
        "欧洲VAT税率一般多少？什么情况下必须注册税号？",
        "跟卖被投诉侵权会封号吗？",
        "你好",
    ]
    for q in questions:
        state = create_query_default_state(
            session_id=f"demo_{hash(q) % 10000}",
            original_query=q,
            is_stream=False,
            role="seller",
        )
        result = query_app.invoke(state)
        answer = result.get("answer", "")
        risk = result.get("risk_level", "none")
        filters = result.get("ec_filters", {})
        print(f"\n  问题: {q}")
        print(f"  筛选: 平台={filters.get('platform')} 类目={filters.get('category')} 风险={risk}")
        print(f"  答案: {answer[:150].replace(chr(10), ' ')}...")
    print()

    # ---------------- 第三步：测评（offline 抽样 3 条 + 启发式评委） ----------------
    print("========== 第三步：测评（offline 模式，抽样 3 条，启发式评委） ==========")
    from evaluation.runner import run

    report = run(mode="offline", use_llm=False, n=3)
    overall = report.get("overall", {})
    print(f"  综合得分: {overall.get('overall_score')}  等级: {overall.get('grade')}")
    print(f"  检索分: {overall.get('retrieval_score')}  生成分: {overall.get('generation_score')}")
    for case in report.get("per_case", []):
        hit = "Y" if case.get("retrieved") else "N"
        print(f"    [{hit}] {case['id']} {case['query'][:30]} -> "
              f"{(case.get('retrieved') or ['-'])[0][:20]}")
    print("\n========== 演示完成 ==========")


if __name__ == "__main__":
    main()
