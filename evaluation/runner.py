"""
评估执行器：加载测试集 → 调用项目系统获取「检索结果 + 答案」→ 计算指标 → 汇总。

三种运行模式：
  - offline : 直接 import 项目模块，调用 query_app 图（最贴近生产代码路径）
  - online  : 通过 HTTP 调用已启动的 /query 接口（需查询服务在运行）
  - selftest: 用合成检索结果验证指标引擎（零外部依赖，秒级）

compare 模式：offline 模式下同时跑一份「纯 LLM 基线」（无 RAG 上下文），
用于回答「RAG 到底有没有比直接问 LLM 更好」。
"""
import json
import sys
import time
import traceback
from pathlib import Path

# 自引导：把 evaluation 目录与项目根加入 sys.path，
# 使 `from metrics import ...`（本包内部绝对导入）在任何调用方式下都可解析
# （python -m evaluation.run_eval / 被 query_server 懒导入 / 被其它脚本 import）。
_EVAL_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _EVAL_DIR.parent
for _p in (str(_EVAL_DIR), str(_PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metrics import evaluate_retrieval, evaluate_generation, objective_generation, compute_overall  # noqa: E402
from judge import judge  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTSET = Path(__file__).resolve().parent / "testset" / "ec_eval.jsonl"


def load_testset(path: Path = TESTSET) -> list[dict]:
    """加载 JSONL 测试集。"""
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _extract_items(docs: list[dict]) -> list[str]:
    """从重排文档列表中提取有序文档标识（优先 item_name，兜底 file_title）。"""
    out = []
    for d in docs or []:
        idn = d.get("item_name") or d.get("file_title") or ""
        if idn:
            out.append(idn)
    return out


def _docs_to_context(docs: list[dict]) -> str:
    """检索上下文拼接（供评委使用）。"""
    parts = []
    for i, d in enumerate(docs or [], 1):
        content = d.get("content") or d.get("text") or ""
        parts.append(f"[片段{i}] {content}")
    return "\n".join(parts)


# ===================== 系统接入 =====================

def run_offline(case: dict) -> dict:
    """离线模式：调用 query_app 图，捕获 reranked_docs 与 answer。"""
    from app.process.query.agent.main_graph import query_app
    from app.process.query.agent.state import create_query_default_state

    state = create_query_default_state(
        session_id=f"eval_{case['id']}",
        original_query=case["query"],
        is_stream=False,
        role=case.get("role", "seller"),
    )
    result = query_app.invoke(state)
    docs = result.get("reranked_docs", []) or []
    answer = result.get("answer", "") or ""

    # rerank_service 会把文档重建为精简结构，丢失 item_name；
    # 用重排前的 embedding_chunks/rrf_chunks 建立「标题 -> item」映射还原顺序。
    emb = result.get("embedding_chunks", []) or []
    rrf = result.get("rrf_chunks", []) or []
    title2item = {}
    for d in emb + rrf:
        nm = d.get("item_name") or d.get("file_title")
        t = d.get("title")
        if nm and t:
            title2item.setdefault(t, nm)
    retrieved = []
    for d in docs:
        nm = d.get("item_name") or d.get("file_title") or title2item.get(d.get("title"))
        if nm:
            retrieved.append(nm)
    if not retrieved:
        retrieved = [d.get("item_name") or d.get("file_title")
                     for d in emb + rrf if (d.get("item_name") or d.get("file_title"))]

    # 文档级去重保序：同一文档多个 chunk 只保留排名最靠前的一次，
    # 避免重复 chunk 挤占 top-k 名额导致 Recall/Precision 被低估。
    seen = set()
    retrieved = [x for x in retrieved if not (x in seen or seen.add(x))]

    return {
        "retrieved": retrieved,
        "answer": answer,
        "context": _docs_to_context(docs),
        "item_names": result.get("item_names", []),
        "ec_filters": result.get("ec_filters", {}),
    }


def run_online(case: dict, base_url: str = "http://localhost:8001") -> dict:
    """在线模式：调用 /query 接口（仅取答案用于生成质量评估）。"""
    import urllib.request

    payload = json.dumps({
        "query": case["query"],
        "is_stream": False,
        "use_web_search": False,
        "role": case.get("role", "seller"),
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/query",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    answer = data.get("answer", "") or ""
    return {
        "retrieved": [],
        "answer": answer,
        "context": "",
        "item_names": [],
        "ec_filters": {},
        "online": True,
    }


def run_selftest(case: dict) -> dict:
    """自测模式：合成检索结果验证指标计算（不依赖项目/服务）。"""
    relevant = case.get("relevant_items", [])
    noise = ["噪声文档A", "噪声文档B", "噪声文档C"]
    retrieved = list(relevant[:2]) + noise + list(relevant[2:])
    answer = "【自测合成答案】" + "；".join(case.get("expected_topics", [])) + \
             "。可参考：" + (case.get("reference", "") or "")
    return {
        "retrieved": retrieved,
        "answer": answer,
        "context": (case.get("reference", "") or "") + " " + "；".join(case.get("expected_topics", [])),
        "item_names": relevant,
        "ec_filters": {"category": case.get("category", "")},
        "selftest": True,
    }


def run_baseline(case: dict) -> dict:
    """纯 LLM 基线：问题直接抛给生成模型，不使用检索上下文。"""
    from app.infra.llm import llm_provider
    from app.shared.model.lm_utils import build_messages, invoke_text

    client = llm_provider.chat()
    messages = build_messages(
        "你是一位专业的跨境电商知识专家，请直接根据你的知识回答下面的问题，"
        "不要引用外部文档，给出实用、安全的建议。保持回答简洁、可落地。",
        case["query"],
    )
    answer = invoke_text(client, messages)
    return {
        "retrieved": [],
        "answer": answer,
        "context": "",
        "item_names": [],
        "ec_filters": {},
        "baseline": True,
    }


# ===================== 主流程 =====================

def run(mode: str = "offline", use_llm: bool = True, base_url: str = "http://localhost:8001",
        testset_path: Path | None = None, n: int | None = None,
        progress_cb=None, compare: bool = False, third_party: bool = False) -> dict:
    """执行评估并返回报告字典。

    third_party=True 时，在自研评测之外额外调用 RAGAS / TruLens（需安装可选依赖，
    未安装则 report["third_party"] 标记 unavailable，不影响主流程）。
    """
    cases = load_testset(testset_path or TESTSET)
    if n is not None and n > 0:
        cases = cases[:n]
    retrieval_cases = []
    judgements = []
    objectives = []
    baseline_judgements = []
    baseline_objectives = []
    per_case = []

    runner = {
        "offline": run_offline,
        "online": lambda c: run_online(c, base_url),
        "selftest": run_selftest,
    }.get(mode, run_offline)

    run_baseline_for_case = run_baseline if (mode == "offline" and compare) else None

    for case in cases:
        rec = {"id": case["id"], "query": case["query"], "role": case.get("role", "seller"),
               "relevant": case.get("relevant_items", [])}
        t0 = time.time()
        try:
            sys_out = runner(case)
            rec["latency_ms"] = round((time.time() - t0) * 1000, 1)
            rec["retrieved"] = sys_out.get("retrieved", [])
            rec["answer"] = sys_out.get("answer", "")
            rec["context"] = sys_out.get("context", "")
            rec["item_names"] = sys_out.get("item_names", [])
            rec["ec_filters"] = sys_out.get("ec_filters", {})
            rec["status"] = "ok" if rec["answer"] else "empty_answer"

            retrieval_cases.append({
                "relevant": case.get("relevant_items", []),
                "retrieved": sys_out.get("retrieved", []),
            })
            if sys_out.get("answer"):
                j = judge(case, sys_out.get("context", ""), sys_out["answer"], use_llm=use_llm)
                rec["judgement"] = j
                judgements.append(j)
                objectives.append(objective_generation(case, sys_out.get("answer", "")))
            else:
                rec["judgement"] = None

            if run_baseline_for_case:
                try:
                    base_out = run_baseline_for_case(case)
                    base_j = judge(case, "", base_out.get("answer", ""), use_llm=use_llm)
                    rec["baseline"] = {"answer": base_out.get("answer", ""), "judgement": base_j}
                    baseline_judgements.append(base_j)
                    baseline_objectives.append(objective_generation(case, base_out.get("answer", "")))
                except Exception as be:
                    rec["baseline"] = {"answer": "", "error": str(be)}
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = str(e)
            rec["trace"] = traceback.format_exc()
            # 计入「零分」以便聚合反映真实失败
            retrieval_cases.append({"relevant": case.get("relevant_items", []), "retrieved": []})
        per_case.append(rec)
        if progress_cb:
            progress_cb(len(per_case), len(cases), case.get("query", ""))

    retrieval = evaluate_retrieval(retrieval_cases, ks=(1, 3, 5, 10))
    generation = evaluate_generation(judgements, objectives) if judgements else \
        {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_relevance": 0.0,
         "completeness": 0.0, "safety": 0.0, "score": 0.0, "objective_score": None, "n_cases": 0}
    overall = compute_overall(retrieval, generation)

    report = {
        "meta": {
            "mode": mode,
            "use_llm_judge": use_llm,
            "n_cases": len(cases),
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "retrieval": retrieval,
        "generation": generation,
        "overall": overall,
        "per_case": per_case,
    }

    if baseline_judgements:
        baseline_generation = evaluate_generation(baseline_judgements, baseline_objectives)
        report["baseline"] = {
            "generation": baseline_generation,
            "delta": {
                "faithfulness": round(generation["faithfulness"] - baseline_generation["faithfulness"], 4),
                "answer_relevancy": round(generation["answer_relevancy"] - baseline_generation["answer_relevancy"], 4),
                "context_relevance": round(generation["context_relevance"] - baseline_generation["context_relevance"], 4),
                "completeness": round(generation["completeness"] - baseline_generation["completeness"], 4),
                "safety": round(generation["safety"] - baseline_generation["safety"], 4),
                "objective_score": round((generation.get("objective_score") or 0) - (baseline_generation.get("objective_score") or 0), 4),
            },
            "n_cases": len(baseline_judgements),
        }

    # ---------------- 第三方评测（可选） ----------------
    if third_party:
        try:
            from evaluation.third_party import available, evaluate

            if not available():
                report["third_party"] = {
                    "available": False,
                    "note": "未安装 ragas / trulens；可执行 pip install -e '.[eval]' 后启用",
                }
            else:
                tp_records = [
                    {"question": c.get("query", ""),
                     "answer": rec["answer"],
                     "context": rec.get("context", "")}
                    for c, rec in zip(cases, per_case)
                    if rec.get("answer")
                ]
                report["third_party"] = evaluate(tp_records)
        except Exception as exc:  # noqa: BLE001
            report["third_party"] = {"available": False, "error": repr(exc)}

    return report


if __name__ == "__main__":
    import sys

    m = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    print(json.dumps(run(m), ensure_ascii=False, indent=2))
