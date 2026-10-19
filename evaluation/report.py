"""
评估报告生成模块：把评估结果渲染为 Markdown / JSON 报告并落盘。
"""
import json
import time
from pathlib import Path


def render_markdown(report: dict) -> str:
    """把评估报告渲染为 Markdown 文本。"""
    meta = report.get("meta", {})
    ret = report.get("retrieval", {})
    gen = report.get("generation", {})
    overall = report.get("overall", {})

    lines: list[str] = []
    lines.append("# 跨境电商 RAG 系统测评报告")
    lines.append("")
    lines.append(f"- 运行模式：{meta.get('mode', '')}")
    lines.append(f"- LLM 评委：{'启用' if meta.get('use_llm_judge') else '关闭（启发式）'}")
    lines.append(f"- 用例数：{meta.get('n_cases', 0)}")
    lines.append(f"- 生成时间：{meta.get('generated_at', '')}")
    lines.append("")

    lines.append("## 一、综合评分")
    lines.append("")
    lines.append(f"**总分：{overall.get('overall_score', 0)} ｜ 等级：{overall.get('grade', '-')}**")
    lines.append("")
    lines.append(f"- 检索分（权重 0.40）：{overall.get('retrieval_score', 0)}")
    lines.append(f"- 生成分（权重 0.60）：{overall.get('generation_score', 0)}")
    lines.append(f"- 安全性（否决观察项）：{overall.get('safety', '-')}")
    lines.append("")

    lines.append("## 二、检索指标")
    lines.append("")
    lines.append("| 指标 | 值 |")
    lines.append("|---|---|")
    for key in ("hit_rate@1", "hit_rate@3", "hit_rate@5", "hit_rate@10",
                "recall@5", "precision@5", "ndcg@5", "mrr"):
        if key in ret:
            lines.append(f"| {key} | {ret[key]} |")
    lines.append("")

    lines.append("## 三、生成质量")
    lines.append("")
    lines.append("| 维度 | 得分（0-1） |")
    lines.append("|---|---|")
    for key in ("faithfulness", "answer_relevancy", "context_relevance",
                "completeness", "safety", "llm_score", "topic_coverage",
                "ref_f1", "objective_score", "score"):
        if gen.get(key) is not None:
            lines.append(f"| {key} | {gen[key]} |")
    lines.append("")

    if report.get("baseline"):
        base = report["baseline"]
        delta = base.get("delta", {})
        lines.append("## 四、纯 LLM 基线对比（无 RAG 上下文）")
        lines.append("")
        lines.append("| 维度 | RAG | 基线 | 差值 |")
        lines.append("|---|---|---|---|")
        for key in ("faithfulness", "answer_relevancy", "context_relevance",
                    "completeness", "safety", "objective_score"):
            rag_val = gen.get(key)
            base_gen = base.get("generation", {})
            base_val = base_gen.get(key)
            d = delta.get(key)
            if rag_val is not None or base_val is not None:
                lines.append(f"| {key} | {rag_val} | {base_val} | {d:+} |" if d is not None
                             else f"| {key} | {rag_val} | {base_val} | - |")
        lines.append("")

    lines.append("## 五、逐用例明细")
    lines.append("")
    lines.append("| ID | 问题 | 命中@5 | 评委分 | 耗时(ms) |")
    lines.append("|---|---|---|---|---|")
    for c in report.get("per_case", []):
        j = c.get("judgement") or {}
        judge_desc = ",".join(f"{k[:4]}:{j.get(k)}" for k in
                              ("faithfulness", "answer_relevancy", "completeness", "safety")
                              if j.get(k) is not None)
        hit = "✓" if c.get("retrieved") else "✗"
        lines.append(f"| {c.get('id', '')} | {str(c.get('query', ''))[:30]} | {hit} | "
                     f"{judge_desc or '-'} | {c.get('latency_ms', '-')} |")
    lines.append("")
    return "\n".join(lines)


def save_report(report: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    """
    保存报告（Markdown + JSON）。

    :return: (md 路径, json 路径)
    """
    out_dir = Path(out_dir) if out_dir else (Path(__file__).resolve().parent / "reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    md_path = out_dir / f"eval_report_{ts}.md"
    json_path = out_dir / f"eval_report_{ts}.json"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return md_path, json_path
