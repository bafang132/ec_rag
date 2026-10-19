"""
第三方评测对接模块（可选依赖，零强制安装）。

支持业界标准 RAG 评测框架：RAGAS 与 TruLens。
- 仅在对应库已安装时启用；未安装时 available() 返回 False，调用方优雅降级到自研测评。
- 复用项目自身的 OpenAI 兼容 LLM 配置（OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL）
  作为评测模型，无需额外密钥。

路线图条目「RAGAS / TruLens 等第三方评测对接」的落地点。
"""
from typing import Optional

# 尝试导入第三方框架，导入失败即视为不可用（不抛出，保持零强制依赖）。
_RAGAS_AVAILABLE = False
_TRULENS_AVAILABLE = False
try:
    import ragas  # noqa: F401

    _RAGAS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _RAGAS_AVAILABLE = False
try:
    import trulens_eval  # noqa: F401

    _TRULENS_AVAILABLE = True
except Exception:  # noqa: BLE001
    _TRULENS_AVAILABLE = False


def available() -> bool:
    """是否安装了任一第三方评测框架。"""
    return _RAGAS_AVAILABLE or _TRULENS_AVAILABLE


def frameworks() -> list[str]:
    """返回已安装的框架名列表。"""
    out = []
    if _RAGAS_AVAILABLE:
        out.append("ragas")
    if _TRULENS_AVAILABLE:
        out.append("trulens")
    return out


def _build_ragas_llm():
    """用项目自身的 OpenAI 兼容配置构造 ragas 评测 LLM（按需懒导入）。"""
    from app.shared.config.common import env_str
    from app.shared.config.lm_config import lm_config

    base_url = lm_config.base_url or env_str("OPENAI_BASE_URL", "")
    api_key = lm_config.api_key or env_str("OPENAI_API_KEY", "")
    model = lm_config.model or env_str("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("未配置 LLM 密钥（OPENAI_API_KEY），无法运行 RAGAS 评测")

    # ragas 依赖 langchain；未安装则直接报错由上层降级。
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    llm = ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url or None,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def evaluate_with_ragas(records: list[dict]) -> dict:
    """
    用 RAGAS 计算标准 RAG 质量指标。

    :param records: 每条形如 {"question": str, "context": str, "answer": str}
    :return: 指标字典（faithfulness / answer_relevancy / context_precision / context_recall）
    """
    if not _RAGAS_AVAILABLE:
        raise RuntimeError("ragas 未安装")
    if not records:
        return {}

    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    llm = _build_ragas_llm()
    dataset = Dataset.from_dict({
        "question": [r.get("question", "") for r in records],
        "answer": [r.get("answer", "") for r in records],
        "contexts": [[r.get("context", "")] for r in records],
    })

    result = ragas_evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
    )
    # ragas 返回的是 Dataset；转为普通字典
    try:
        return {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    except Exception:  # noqa: BLE001
        return dict(result)


def evaluate(records: list[dict]) -> dict:
    """
    统一入口：优先 RAGAS（指标更全），失败再降级说明。

    返回结构：
      {"framework": "ragas", "metrics": {...}} 或 {"framework": None, "error": "..."}
    """
    if _RAGAS_AVAILABLE:
        try:
            metrics = evaluate_with_ragas(records)
            return {"framework": "ragas", "metrics": metrics, "available": True}
        except Exception as exc:  # noqa: BLE001
            return {"framework": "ragas", "available": True,
                    "error": f"RAGAS 评测失败：{exc}"}
    return {
        "framework": None,
        "available": available(),
        "frameworks_installed": frameworks(),
        "error": "未安装 ragas / trulens，无法运行第三方评测（pip install -e '.[eval]' 后启用）",
    }


def evaluate_with_trulens(query_fn, questions: list[str]) -> Optional[dict]:
    """TruLens 应用级评测占位（需传入可调用 query_fn(question)->answer）。

    设计为可选增强：安装 trulens 且提供 query_fn 时记录反馈闭环。
    """
    if not _TRULENS_AVAILABLE:
        return None
    # TruLens 集成需与具体链路绑定，这里仅返回可用性，详细埋点可在应用层扩展。
    return {"framework": "trulens", "available": True, "n_questions": len(questions)}
