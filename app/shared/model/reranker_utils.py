"""
Reranker 工具模块：双实现自动选择。

- FlagRerankerProvider：本地 bge-reranker-large（FlagEmbedding，Cross-Encoder），精度最佳
- LLMRerankerProvider：LLM 兜底重排（本地模型不可用时保证链路完整），
  单次批量打分，输出与 FlagReranker 一致的归一化分数

统一接口：compute_score(pairs: list[[query, doc]], normalize=True) -> list[float]
"""
from typing import Optional

from app.shared.config.reranker_config import reranker_config
from app.shared.model.device_utils import resolve_device, resolve_fp16
from app.shared.runtime.logger import logger

_MAX_LLM_RERANK_PAIRS = 12


class BaseReranker:
    """Reranker 统一接口。"""

    name = "base"

    def compute_score(self, pairs: list[list[str]], normalize: bool = True) -> list[float]:
        raise NotImplementedError

    @property
    def tokenizer(self):
        """提供 tokenizer.encode(text, add_special_tokens=False) 以估算上下文长度。"""
        raise NotImplementedError


class _SimpleTokenizer:
    """无模型场景的字符级 token 估算器（中文 1 字 ≈ 1 token）。"""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return list(text or "")


class FlagRerankerProvider(BaseReranker):
    """本地 bge-reranker-large（FlagEmbedding）。"""

    name = "bge-reranker-large"

    def __init__(self) -> None:
        from FlagEmbedding import FlagReranker

        device = resolve_device(reranker_config.bge_reranker_device, default="cpu")
        use_fp16 = resolve_fp16(reranker_config.bge_reranker_fp16, device)
        logger.info(
            f"初始化 bge-reranker-large：model={reranker_config.bge_reranker_large}, "
            f"device={device}, fp16={use_fp16}"
        )
        self._model = FlagReranker(
            model_name_or_path=reranker_config.bge_reranker_large,
            device=device,
            use_fp16=use_fp16,
        )
        logger.success("bge-reranker-large 初始化成功")

    def compute_score(self, pairs: list[list[str]], normalize: bool = True) -> list[float]:
        if not pairs:
            return []
        scores = self._model.compute_score(pairs, normalize=normalize)
        try:
            return [float(s) for s in scores]
        except TypeError:
            return [float(scores)]

    @property
    def tokenizer(self):
        return self._model.tokenizer


class LLMRerankerProvider(BaseReranker):
    """
    LLM 兜底重排：让大模型对 (问题, 候选) 对批量打 0-10 分。

    单次调用完成整批评分（截断到 _MAX_LLM_RERANK_PAIRS 对以内），
    分数 sigmoid 归一化到 0-1，保证与 FlagReranker 输出可比。
    """

    name = "llm-reranker"

    _PROMPT = """你是检索相关性评估专家。请对下面每一条【候选内容】与【用户问题】的相关性打 0-10 的整数分数。
10=完全回答问题，0=完全无关。只输出 JSON 数组，元素个数与候选一致，不要多余文字。

【用户问题】
{query}

【候选内容】
{candidates}

只输出如下格式：
[{{"idx":1,"score":8}}, ...]"""

    def __init__(self) -> None:
        from app.shared.model.lm_utils import get_llm_client, invoke_text

        self._get_client = get_llm_client
        self._invoke_text = invoke_text
        logger.info("LLM 兜底重排初始化（本地 reranker 不可用）")

    def compute_score(self, pairs: list[list[str]], normalize: bool = True) -> list[float]:
        if not pairs:
            return []
        pairs = pairs[:_MAX_LLM_RERANK_PAIRS]
        query = pairs[0][0]
        candidates = "\n".join(
            f"{i}. {doc[:600]}" for i, (_, doc) in enumerate(pairs, start=1)
        )
        prompt = self._PROMPT.format(query=query, candidates=candidates)
        try:
            client = self._get_client()
            raw = self._invoke_text(client, prompt)
            scores = self._parse_scores(raw, len(pairs))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"LLM 重排失败，使用等分兜底：{exc}")
            scores = [5.0] * len(pairs)

        if normalize:
            scores = [self._sigmoid(s) for s in scores]
        return scores

    @staticmethod
    def _parse_scores(raw: str, expected: int) -> list[float]:
        import json
        import re

        match = re.search(r"\[.*\]", raw or "", re.DOTALL)
        if not match:
            return [5.0] * expected
        try:
            items = json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            return [5.0] * expected
        by_idx = {int(it.get("idx", 0)): float(it.get("score", 5)) for it in items if isinstance(it, dict)}
        return [min(10.0, max(0.0, by_idx.get(i, 5.0))) for i in range(1, expected + 1)]

    @staticmethod
    def _sigmoid(x: float) -> float:
        import math

        # 以 5 分为中点做平滑映射，保证排序与原分一致且落在 0-1
        return 1.0 / (1.0 + math.exp(-(x - 5.0)))

    @property
    def tokenizer(self):
        return _SimpleTokenizer()


# ===================== 工厂（单例 + 自动降级） =====================

_reranker_instance: Optional[BaseReranker] = None


def _try_build(provider: str) -> Optional[BaseReranker]:
    try:
        if provider == "bge":
            return FlagRerankerProvider()
        if provider == "llm":
            return LLMRerankerProvider()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Reranker 提供方 [{provider}] 初始化失败：{exc}")
    return None


def get_reranker_model() -> BaseReranker:
    """
    获取 Reranker 单例（auto 模式：bge 不可用则 LLM 兜底）。

    :return: 可用的 Reranker（compute_score / tokenizer 接口）
    """
    global _reranker_instance
    if _reranker_instance is not None:
        return _reranker_instance

    configured = reranker_config.provider
    if configured in ("bge", "llm"):
        reranker = _try_build(configured)
        if reranker is None:
            raise RuntimeError(f"Reranker 提供方 [{configured}] 初始化失败")
        _reranker_instance = reranker
        return reranker

    # auto：bge -> llm
    reranker = _try_build("bge")
    if reranker is None:
        logger.info("本地 bge-reranker 不可用，降级 LLM 兜底重排")
        reranker = _try_build("llm")
    if reranker is None:
        raise RuntimeError("无可用 Reranker（bge/llm 均初始化失败）")
    _reranker_instance = reranker
    return reranker


if __name__ == "__main__":
    r = get_reranker_model()
    print(f"reranker={r.name}")
    print(r.compute_score([["亚马逊FBA运费", "FBA 运费按重量与尺寸计算"], ["亚马逊FBA运费", "Listing 标题怎么写"]]))
