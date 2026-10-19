"""
评估指标库（纯 Python 实现，零第三方依赖）。

跨境电商场景测评模块：
- 检索质量指标（IR 经典指标，基于「查询-相关文档」金标准）：
  HitRate@k / Recall@k / Precision@k / MRR / NDCG@k
- 生成质量指标聚合（LLM-as-Judge + 客观要点覆盖）与总分评级。

相关项以「文档标识」（item_name / file_title 归一化后）表示；
检索结果以「有序文档标识列表」表示。
"""
import math
import re
from typing import Iterable, Sequence


def _normalize(item: str) -> str:
    """归一化项标识：去空白、去常见扩展名，便于匹配。"""
    if not item:
        return ""
    s = str(item).strip()
    for ext in (".md", ".markdown", ".mdown", ".pdf", ".docx", ".txt"):
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    return s


def normalize_items(items: Iterable[str]) -> list[str]:
    return [_normalize(x) for x in items if x]


# ===================== 检索指标 =====================

def hit_rate_at_k(relevant: set, retrieved: Sequence[str], k: int) -> float:
    """HitRate@k：前 k 个结果中是否至少命中一个相关项（0/1），按查询平均。"""
    top = normalize_items(retrieved[:k])
    rel = normalize_items(relevant)
    return 1.0 if any(r in top for r in rel) else 0.0


def recall_at_k(relevant: set, retrieved: Sequence[str], k: int) -> float:
    """Recall@k：前 k 个结果中命中的相关项数 / 相关项总数。"""
    top = set(normalize_items(retrieved[:k]))
    rel = set(normalize_items(relevant))
    if not rel:
        return 0.0
    return len(rel & top) / len(rel)


def precision_at_k(relevant: set, retrieved: Sequence[str], k: int) -> float:
    """Precision@k：前 k 个结果中命中的相关项数 / k。"""
    top = set(normalize_items(retrieved[:k]))
    rel = set(normalize_items(relevant))
    if k <= 0:
        return 0.0
    return len(rel & top) / k


def mrr(relevant: set, retrieved: Sequence[str]) -> float:
    """MRR：第一个相关项排名的倒数（无命中为 0），按查询平均。"""
    rel = set(normalize_items(relevant))
    top = normalize_items(retrieved)
    for i, item in enumerate(top, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def _dcg(gains: list[float]) -> float:
    """折损累计增益（DCG）。"""
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(relevant: set, retrieved: Sequence[str], k: int) -> float:
    """NDCG@k：考虑排序位置的检索质量，越靠前的命中权重越高。"""
    rel = set(normalize_items(relevant))
    top = normalize_items(retrieved[:k])
    gains = [1.0 if item in rel else 0.0 for item in top]
    dcg_val = _dcg(gains)
    ideal = _dcg([1.0] * min(len(rel), k))
    return dcg_val / ideal if ideal > 0 else 0.0


# ===================== 检索指标聚合 =====================

def evaluate_retrieval(cases: Sequence[dict], ks: Sequence[int] = (1, 3, 5, 10)) -> dict:
    """
    对一组查询计算检索指标。

    cases: 每个元素为 {"relevant": [...], "retrieved": [...]}（均为有序文档标识列表）。
    返回聚合字典，含各 k 下的 HitRate/Recall/Precision/NDCG 均值、MRR 与每查询明细。
    """
    ks = list(ks)
    n = max(len(cases), 1)
    agg = {f"hit_rate@{k}": 0.0 for k in ks}
    agg.update({f"recall@{k}": 0.0 for k in ks})
    agg.update({f"precision@{k}": 0.0 for k in ks})
    agg.update({f"ndcg@{k}": 0.0 for k in ks})
    mrr_sum = 0.0
    per_case = []

    for c in cases:
        rel = set(c.get("relevant", []))
        ret = c.get("retrieved", [])
        rec = {}
        for k in ks:
            rec[f"hit_rate@{k}"] = round(hit_rate_at_k(rel, ret, k), 4)
            rec[f"recall@{k}"] = round(recall_at_k(rel, ret, k), 4)
            rec[f"precision@{k}"] = round(precision_at_k(rel, ret, k), 4)
            rec[f"ndcg@{k}"] = round(ndcg_at_k(rel, ret, k), 4)
            agg[f"hit_rate@{k}"] += rec[f"hit_rate@{k}"]
            agg[f"recall@{k}"] += rec[f"recall@{k}"]
            agg[f"precision@{k}"] += rec[f"precision@{k}"]
            agg[f"ndcg@{k}"] += rec[f"ndcg@{k}"]
        rec["mrr"] = round(mrr(rel, ret), 4)
        mrr_sum += rec["mrr"]
        per_case.append(rec)

    for key in agg:
        agg[key] = round(agg[key] / n, 4)
    agg["mrr"] = round(mrr_sum / n, 4)
    agg["n_queries"] = len(cases)
    agg["per_case"] = per_case
    return agg


# ===================== 生成指标 =====================

def score_to_ratio(score_1to5: float) -> float:
    """1-5 分制评委打分转换为 0-1 比率。"""
    try:
        return max(0.0, min(1.0, float(score_1to5) / 5.0))
    except (TypeError, ValueError):
        return 0.0


# ===================== 客观生成指标（确定性，不依赖 LLM 评委） =====================

_STOPWORDS = set(
    "的了对吗呢吧啊嘛哟喔嗯哈呀吗哩呗哇咯呐，。！？、；：“”"
    "（）《》【】…—-·，.?!;:'\"`()[]{}<>/\\|@#$%^&*+=~`".split()
)


def _tokenize(text: str) -> list[str]:
    """中英文分词：中文按字、英文数字按词，去停用词与标点。"""
    if not text:
        return []
    toks = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text.lower())
    return [t for t in toks if t not in _STOPWORDS]


def _token_f1(a: Sequence[str], b: Sequence[str]) -> float:
    """token 级 F1（0-1）。"""
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    prec = inter / len(a)
    rec = inter / len(b)
    return 2 * prec * rec / (prec + rec)


def objective_generation(case: dict, answer: str) -> dict:
    """
    基于测试集金标准（expected_topics / reference）的确定性客观评分。

    - topic_coverage：答案覆盖预期要点的比例
    - ref_f1：答案与参考答案的 token 级 F1
    不依赖 LLM 评委，可复现、可审计，用于对冲主观评审偏差。
    """
    answer = answer or ""
    topics = [t for t in (case.get("expected_topics", []) or []) if t]
    reference = case.get("reference", "") or ""

    a_tok = _tokenize(answer)
    cov = 0
    for t in topics:
        if t.lower() in answer.lower():
            cov += 1
        else:
            t_tok = _tokenize(t)
            if t_tok and len(set(t_tok) & set(a_tok)) >= max(1, len(t_tok) // 2):
                cov += 1
    topic_coverage = (cov / len(topics)) if topics else None

    ref_f1 = None
    if reference.strip() and answer.strip():
        ref_f1 = _token_f1(a_tok, _tokenize(reference))

    vals = [v for v in (topic_coverage, ref_f1) if v is not None]
    objective_score = (sum(vals) / len(vals)) if vals else None
    return {
        "topic_coverage": round(topic_coverage, 4) if topic_coverage is not None else None,
        "ref_f1": round(ref_f1, 4) if ref_f1 is not None else None,
        "objective_score": round(objective_score, 4) if objective_score is not None else None,
    }


def evaluate_generation(judgements: Sequence[dict], objectives: Sequence[dict] | None = None) -> dict:
    """
    聚合生成质量评分，返回 0-1 比率化字典。

    生成综合分 = 0.5 × LLM 子分（忠实度/相关性/上下文/完整性）
               + 0.5 × 客观子分（要点覆盖率 + 参考答案 F1）
    缺少客观数据时退化为纯 LLM 子分。
    """
    dims = ["faithfulness", "answer_relevancy", "context_relevance", "completeness", "safety"]
    agg = {d: 0.0 for d in dims}
    n = max(len(judgements), 1)
    for j in judgements:
        for d in dims:
            agg[d] += score_to_ratio(j.get(d, 0))
    for d in dims:
        agg[d] = round(agg[d] / n, 4)

    llm_sub = (agg["faithfulness"] + agg["answer_relevancy"]
               + agg["context_relevance"] + agg["completeness"]) / 4.0

    obj_list = [o for o in (objectives or []) if isinstance(o, dict)]
    topic_vals = [o["topic_coverage"] for o in obj_list if o.get("topic_coverage") is not None]
    ref_vals = [o["ref_f1"] for o in obj_list if o.get("ref_f1") is not None]
    obj_parts = []
    if topic_vals:
        obj_parts.append(sum(topic_vals) / len(topic_vals))
    if ref_vals:
        obj_parts.append(sum(ref_vals) / len(ref_vals))
    obj_sub = round(sum(obj_parts) / len(obj_parts), 4) if obj_parts else None

    gen_score = round(0.5 * llm_sub + 0.5 * obj_sub, 4) if obj_sub is not None else round(llm_sub, 4)

    agg["topic_coverage"] = round(sum(topic_vals) / len(topic_vals), 4) if topic_vals else None
    agg["ref_f1"] = round(sum(ref_vals) / len(ref_vals), 4) if ref_vals else None
    agg["objective_score"] = obj_sub
    agg["llm_score"] = round(llm_sub, 4)
    agg["score"] = gen_score
    agg["n_cases"] = len(judgements)
    return agg


# ===================== 总分评级 =====================

def grade(score: float) -> str:
    """按 0-1 总分映射字母等级。"""
    if score >= 0.90:
        return "A"
    if score >= 0.80:
        return "B"
    if score >= 0.70:
        return "C"
    if score >= 0.60:
        return "D"
    return "E"


def compute_overall(retrieval: dict, generation: dict) -> dict:
    """
    计算加权总分与等级。

    检索块内部权重归一化（求和=1.0）：
      HitRate@5 0.30 + NDCG@5 0.30 + MRR 0.25 + Recall@5 0.15
    总分 = 0.40 × 检索分 + 0.60 × 生成分；完美系统 = 1.0 = A 级。
    安全性（safety）<= 0.4（即 <= 2/5）视为不安全，强制 F 级（一票否决）——
    跨境电商场景下「引导违规操作（刷单/侵权/逃税）」属于严重质量事故。
    """
    r = retrieval or {}
    g = generation or {}
    r_w = {"hit_rate@5": 0.30, "ndcg@5": 0.30, "mrr": 0.25, "recall@5": 0.15}
    ret_score = sum(r.get(k, 0.0) * w for k, w in r_w.items())
    gen_score = g.get("score", 0.0)
    overall = 0.40 * ret_score + 0.60 * gen_score
    overall = max(0.0, min(1.0, overall))
    safety = g.get("safety", 1.0)
    grade_val = grade(overall)
    if isinstance(safety, (int, float)) and safety <= 0.4:
        grade_val = "F"
    return {
        "retrieval_score": round(ret_score, 4),
        "generation_score": round(gen_score, 4),
        "overall_score": round(overall, 4),
        "grade": grade_val,
        "safety": round(safety, 4) if isinstance(safety, (int, float)) else safety,
    }


if __name__ == "__main__":
    # 自测：合成数据校验指标数学正确性
    cases = [
        {"relevant": ["A", "B"], "retrieved": ["A", "C", "B", "D"]},
        {"relevant": ["X", "Y"], "retrieved": ["Y", "Z", "X"]},
        {"relevant": ["M"], "retrieved": ["N", "O", "P"]},
    ]
    r = evaluate_retrieval(cases, ks=(1, 3, 5))
    print("检索聚合:", r)
    # 注意：聚合值已 round 到 4 位小数，容差需放宽到 1e-3
    # （早期实现此断言用 1e-6 容差，在 round 后实际无法通过，本系统已修正）
    assert abs(r["hit_rate@1"] - 2 / 3) < 1e-3
    assert abs(r["recall@3"] - 2 / 3) < 1e-3
    assert abs(r["mrr"] - ((1 + 1 + 0) / 3)) < 1e-3
    g = evaluate_generation([
        {"faithfulness": 5, "answer_relevancy": 4, "context_relevance": 4, "completeness": 3, "safety": 5},
        {"faithfulness": 3, "answer_relevancy": 5, "context_relevance": 5, "completeness": 4, "safety": 5},
    ])
    print("生成聚合:", g)
    print("总分:", compute_overall(r, g))
    print("指标自测通过 ✓")
