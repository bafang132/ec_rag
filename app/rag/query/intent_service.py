"""
跨境电商意图识别服务模块：问题改写 + 领域筛选条件抽取 + 合规风险检测。

在一次模型调用中同时完成（采用单次调用设计，降低链路耗时）：
1. 查询改写：结合历史对话补全指代/省略，并做术语归一化；
2. 领域抽取：平台 / 市场 / 类目 / 文档类型；
3. 澄清判断：问题过于模糊且历史未澄清过时温和追问；
4. 合规风险：命中高风险规则时附加合规提醒（账号/知识产权/税务/禁售品/平台处罚）。
"""
import re
from typing import Optional

from app.infra.llm import llm_provider
from app.infra.persistence import history_repository
from app.rag.query.web_search_service import is_web_search_configured
from app.shared.config.ec_config import (
    CATEGORIES,
    DOC_TYPES,
    MARKETS,
    PLATFORMS,
    UNKNOWN_VALUE,
)
from app.shared.model.lm_utils import build_messages, invoke_text
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.shared.utils.ec_utils import (
    detect_compliance_risk,
    guess_category,
    guess_market,
    guess_platform,
    normalize_category,
    normalize_market,
    normalize_platform,
    normalize_enum,
    safe_json_loads,
)

QUERY_HISTORY_LIMIT = 10

# 寒暄/闲聊关键词：命中后直接友好回复，跳过 RAG 检索
# （实测教训："你好"被送进向量库检索会返回无关内容并被 LLM 强行作答）
GREETING_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^你好[呀啊吗呢]?$"),
     "你好！我是跨境电商智能问答助手，可以帮你解答亚马逊/速卖通/eBay/TikTok Shop 等平台的"
     "物流、售后、Listing、广告、账号、税务问题。请告诉我你遇到的具体情况～"),
    (re.compile(r"^(您好|hi|hello|hey|哈喽|嗨)[!！,。.\s]*$"),
     "你好！我是跨境电商问答助手，支持平台规则、FBA 物流、VAT 税务、广告投放等问答。"
     "请直接描述你的问题～"),
    (re.compile(r"^(早上好|上午好|中午好|下午好|晚上好)[!！,。.\s]*$"),
     "你好呀！今天有什么跨境电商的问题想讨论？"),
    (re.compile(r"^(再见|拜拜|bye|byebye|88)[!！,。.\s]*$"),
     "好的，再见！有跨境问题随时来找我～"),
    (re.compile(r"^(谢谢|感谢|thx|thanks|辛苦了)[!！,。.\s]*$"),
     "不客气！还有其他跨境运营问题想问吗？"),
    (re.compile(r"^(你是谁|你叫什么|你是做什么的|你能做什么|你能帮我什么)[?？!！,。.\s]*$"),
     "我是跨境电商 RAG 智能问答助手，基于平台规则与政策知识库回答物流、售后、Listing、"
     "广告、账号合规、税务等问题，并会标注信息来源。请告诉我你的具体问题～"),
    (re.compile(r"^(在吗|在么|有人吗)[?？!！,。.\s]*$"),
     "在的！有什么跨境电商问题想和我讨论？"),
    (re.compile(r"^(好的|知道了|明白了|了解)[!！,。.\s]*$"),
     "好的，有什么新问题随时告诉我～"),
]


def detect_greeting_query(query: str) -> Optional[str]:
    """前置寒暄/闲聊检测：命中则直接返回友好回复，跳过检索。"""
    if not query:
        return None
    text = query.strip().lower()
    for pattern, reply in GREETING_PATTERNS:
        if pattern.match(text):
            logger.info(f"命中寒暄/闲聊模板，直接友好回复: query={query[:20]}")
            return reply
    return None


def validate_query_identity(state: dict) -> tuple[str, str]:
    """校验查询状态的核心字段。"""
    original_query = state.get("original_query")
    session_id = state.get("session_id")
    if not original_query or not session_id:
        raise ValueError("session_id 和 original_query 不能为空")
    return original_query, session_id


def load_query_history(session_id: str) -> list[dict]:
    """读取会话最近历史（用于改写时理解上下文）。"""
    return history_repository.list_recent(session_id, limit=QUERY_HISTORY_LIMIT)


def build_history_text(history_messages: list[dict]) -> str:
    """将历史消息拼接为适合 Prompt 的纯文本。"""
    if not history_messages:
        return "没有历史聊天记录!"
    lines: list[str] = []
    for msg in history_messages:
        content = msg.get("rewritten_query") if msg.get("role") == "user" else msg.get("text")
        lines.append(f"角色:{msg.get('role', '')},内容:{content}")
    return "\n".join(lines)


def rewrite_and_extract_intent(history_messages: list[dict], original_query: str) -> dict:
    """
    单次模型调用：查询改写 + 领域筛选条件抽取 + 澄清判断。

    :return: 结构化结果字典；模型不可用时退化为规则兜底（保证链路不中断）
    """
    try:
        client = llm_provider.chat(json_mode=True)
        prompt = load_prompt(
            "ec_query_rewrite_and_intent",
            history_text=build_history_text(history_messages),
            query=original_query,
        )
        system = (
            "你是专业的跨境电商问答助手，擅长理解卖家问题并精准匹配知识库筛选条件。"
            "你熟悉亚马逊、速卖通、eBay、Shopee、TikTok Shop 等平台的规则与术语。"
        )
        raw = invoke_text(client, build_messages(system, prompt))
        result = safe_json_loads(raw) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"意图识别 LLM 调用失败，使用规则兜底：{exc}")
        result = {}

    # 兜底默认值
    defaults = {
        "rewritten_query": original_query,
        "platform": UNKNOWN_VALUE,
        "market": UNKNOWN_VALUE,
        "category": UNKNOWN_VALUE,
        "doc_type": UNKNOWN_VALUE,
        "needs_clarification": False,
        "clarify_message": "",
    }
    for k, v in defaults.items():
        result.setdefault(k, v)

    # 枚举规范化（模型输出非法值时归一化）
    result["platform"] = _normalize_or_fallback(
        result.get("platform"), PLATFORMS, guess_platform, original_query
    )
    result["market"] = _normalize_or_fallback(
        result.get("market"), MARKETS, guess_market, original_query
    )
    result["category"] = _normalize_or_fallback(
        result.get("category"), CATEGORIES, guess_category, original_query
    )
    result["doc_type"] = normalize_enum(result.get("doc_type") or "", DOC_TYPES)

    # 合规风险检测（规则引擎，不依赖模型）
    risk_info = detect_compliance_risk(original_query)
    result["risk_level"] = risk_info["risk_level"]
    result["risk_types"] = risk_info["risk_types"]
    result["risk_warning"] = risk_info["risk_warning"]
    return result


def _normalize_or_fallback(value: str, candidates: list[str], guess_fn, query: str) -> str:
    """模型值归一化失败时，用关键词规则从原问题推断。"""
    normalized = normalize_enum(value or "", candidates)
    if normalized == UNKNOWN_VALUE:
        return guess_fn(query)
    return normalized


def apply_intent_result(state: dict, result: dict) -> None:
    """把意图识别结果写回查询状态。"""
    state["rewritten_query"] = result.get("rewritten_query", "")
    state["ec_filters"] = {
        "platform": result.get("platform", UNKNOWN_VALUE),
        "market": result.get("market", UNKNOWN_VALUE),
        "category": result.get("category", UNKNOWN_VALUE),
        "doc_type": result.get("doc_type", UNKNOWN_VALUE),
        "needs_clarification": bool(result.get("needs_clarification")),
    }
    state["risk_level"] = result.get("risk_level", "none")
    state["risk_types"] = result.get("risk_types", [])
    state["risk_warning"] = result.get("risk_warning", "")

    # 需要澄清时直接产出澄清答案，跳过后续检索
    if result.get("needs_clarification") and result.get("clarify_message"):
        state["answer"] = result["clarify_message"]


def save_user_message(state: dict) -> None:
    """保存用户消息与意图识别结果到历史。"""
    history_repository.save_message(
        session_id=state["session_id"],
        role="user",
        text=state["original_query"],
        rewritten_query=state.get("rewritten_query", ""),
        item_names=[],
    )


def confirm_ec_intent(state: dict) -> dict:
    """
    意图识别主流程（node_ec_intent 业务实现）。

    寒暄直达答案；否则 历史加载 → 改写+抽取 → 风险检测 → 写回状态 → 保存历史。
    """
    original_query, session_id = validate_query_identity(state)

    # 1) 寒暄/闲聊前置检测
    greeting_reply = detect_greeting_query(original_query)
    if greeting_reply:
        state["answer"] = greeting_reply
        state["ec_filters"] = {
            "platform": UNKNOWN_VALUE, "market": UNKNOWN_VALUE,
            "category": UNKNOWN_VALUE, "doc_type": UNKNOWN_VALUE,
            "needs_clarification": True,
        }
        state["risk_level"] = "none"
        state["risk_types"] = []
        state["risk_warning"] = ""
        return state

    # 2) 历史上下文（时间正序，旧→新）
    history_messages = load_query_history(session_id)
    state["history"] = list(reversed(history_messages))

    # 3) 改写 + 领域抽取 + 合规风险
    result = rewrite_and_extract_intent(state["history"], original_query)
    apply_intent_result(state, result)

    # 4) 联网搜索可用性标记（前端可据此展示开关）
    state["web_search_available"] = is_web_search_configured()

    # 5) 用户消息落库
    save_user_message(state)
    return state
