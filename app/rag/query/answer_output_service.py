"""
答案生成服务模块：Prompt 组装、模型生成、引用溯源与结果回写。

多角色回答模板（role）：
- seller  : 新手卖家——结论先行 + 操作步骤 + 风险提示
- operator: 成熟运营——策略视角 + 数据指标 + 优化路径
- expert  : 风控专家——引用来源 + 矛盾点识别 + 合规审查

合规增强：命中合规风险的提问会自动把合规提醒拼入上下文，
并在 seller/operator 模板中强化【风险提示】栏目。
"""
import re

from app.infra.llm import llm_provider
from app.infra.persistence import history_repository
from app.rag.query.rerank_fallback_service import check_rag_fallback_needed
from app.shared.config.ec_config import UNKNOWN_VALUE
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.shared.model.lm_utils import build_messages, invoke_text, stream_text
from app.shared.utils.sse_utils import SSEEvent, push_to_session
from app.shared.utils.task_utils import set_task_result

# 角色 -> 提示词模板映射
ROLE_PROMPT_MAP = {
    "seller": "seller_answer_out",
    "operator": "operator_answer_out",
    "expert": "expert_answer_out",
}

DEFAULT_ROLE = "seller"


def try_return_existing_answer(state: dict) -> bool:
    """优先返回状态中已存在的答案（寒暄/澄清分支），避免重复调用模型。"""
    answer = state.get("answer")
    if not answer:
        return False
    if state.get("is_stream"):
        # 一次性推送整段答案，避免逐字符延迟
        push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
    set_task_result(state["session_id"], "answer", answer)
    return True


def validate_generation_inputs(state: dict) -> tuple[list[dict], str, list[dict]]:
    """校验答案生成阶段输入（检索为空时降级生成，不抛异常）。"""
    reranked_docs = state.get("reranked_docs", []) or []
    rewritten_query = state.get("rewritten_query") or state.get("original_query") or ""
    if not rewritten_query:
        raise ValueError("rewritten_query 和 original_query 均为空，无法生成答案")
    if not reranked_docs:
        logger.warning("reranked_docs 为空，将基于问题本身与合规提醒兜底回答")
    history = state.get("history", []) or []
    return reranked_docs, rewritten_query, history


def build_history_text(history_messages: list[dict]) -> str:
    """历史消息拼接为 Prompt 用纯文本。"""
    if not history_messages:
        return "没有历史聊天记录!"
    lines: list[str] = []
    for msg in history_messages:
        content = msg.get("rewritten_query") if msg.get("role") == "user" else msg.get("text")
        lines.append(f"角色:{msg.get('role', '')},内容:{content}")
    return "\n".join(lines)


def build_context_text(reranked_docs: list[dict]) -> str:
    """
    构造带来源标注的上下文（引用溯源的关键：每块标注文档标题+平台+匹配度）。

    格式：第N块: 标题:《xxx》(平台:亚马逊) 匹配度:0.83 来源:向量查询/联网搜索
    """
    context_chunk_list = []
    for number, chunk in enumerate(reranked_docs, start=1):
        source = "联网搜索" if chunk.get("type") == "web" else "向量查询"
        platform = chunk.get("platform") or ""
        platform_desc = f"(平台:{platform})" if platform else ""
        context_chunk_list.append(
            f"第{number}块: 标题:《{chunk.get('title') or chunk.get('file_title') or '未知文档'}》"
            f"{platform_desc} 匹配度:{chunk.get('score', 0)} 来源:{source}\n"
            f"内容:{chunk.get('content') or chunk.get('text', '')}"
        )
    return "\n\n".join(context_chunk_list)


def build_answer_prompt(
    reranked_docs: list[dict],
    rewritten_query: str,
    history: list[dict],
    state: dict,
) -> str:
    """按角色构建答案生成 Prompt（含合规提醒注入）。"""
    context_text = build_context_text(reranked_docs)
    history_text = build_history_text(history)

    # 合规风险注入：命中高风险领域时把提醒拼入上下文
    risk_info = ""
    risk_level = state.get("risk_level", "none")
    risk_warning = state.get("risk_warning", "")
    if risk_level != "none" and risk_warning:
        risk_info = f"\n\n【合规风险提醒】\n{risk_warning}\n请在【风险提示】部分重点解读上述合规要求。"

    role = state.get("role") or DEFAULT_ROLE
    prompt_name = ROLE_PROMPT_MAP.get(role, ROLE_PROMPT_MAP[DEFAULT_ROLE])
    return load_prompt(
        prompt_name,
        context=context_text + risk_info,
        history=history_text,
        question=rewritten_query,
    )


def generate_answer(state: dict, prompt: str) -> str:
    """
    调用大模型生成答案（流式/非流式）。

    LLM 不可用（未配置密钥/网络故障）时降级为「检索预览模式」：
    直接展示检索到的知识片段与来源，保证系统在最小环境下仍可用。
    """
    is_stream = state.get("is_stream", False)
    session_id = state.get("session_id")
    messages = build_messages("你是专业的跨境电商知识问答助手", prompt)

    try:
        client = llm_provider.chat()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"LLM 客户端不可用，进入检索预览模式：{exc}")
        return _fallback_preview_answer(state, messages)

    try:
        final_result = ""
        if is_stream:
            for delta in stream_text(client, messages):
                final_result += delta
                push_to_session(session_id, SSEEvent.DELTA, {"delta": delta})
        else:
            final_result = invoke_text(client, messages)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"LLM 生成失败，降级检索预览模式：{exc}")
        return _fallback_preview_answer(state, messages)

    if not final_result.strip():
        logger.warning("LLM 返回空答案，降级检索预览模式")
        return _fallback_preview_answer(state, messages)

    set_task_result(session_id, "answer", final_result)
    state["answer"] = final_result
    return final_result


def _fallback_preview_answer(state: dict, messages) -> str:
    """检索预览降级：无 LLM 时直接展示检索片段 + 来源 + 配置指引。"""
    docs = state.get("reranked_docs", []) or []
    if not docs:
        answer = (
            "本地知识库未检索到相关内容，且大模型服务当前不可用。\n\n"
            "建议：1）检查 .env 中 OPENAI_API_KEY / OPENAI_BASE_URL 配置；"
            "2）确认知识库已导入相关文档。"
        )
    else:
        lines = [
            "⚠️ 大模型服务当前不可用，以下为检索到的相关知识片段（预览模式）。",
            "配置 OPENAI_API_KEY 后将由大模型生成完整回答。\n",
        ]
        for i, doc in enumerate(docs[:5], start=1):
            title = doc.get("title") or doc.get("file_title") or "未知文档"
            content = (doc.get("content") or doc.get("text") or "")[:300]
            lines.append(f"【片段{i}】《{title}》（匹配度 {doc.get('score', 0)}）\n{content}\n")
        lines.append("【参考来源】" + "、".join(
            f"《{d.get('title') or d.get('file_title')}》" for d in docs[:5]
        ))
        answer = "\n".join(lines)
    set_task_result(state["session_id"], "answer", answer)
    state["answer"] = answer
    if state.get("is_stream"):
        push_to_session(state["session_id"], SSEEvent.DELTA, {"delta": answer})
    return answer


def extract_image_urls(reranked_docs: list[dict]) -> list[str]:
    """从重排文档中提取可用于前端展示的图片链接。"""
    image_urls: list[str] = []
    reg = re.compile(r"!\[.*?\]\((.*?)\)")
    for doc in reranked_docs:
        url = doc.get("url")
        text = doc.get("text") or doc.get("content", "")
        if url and url.endswith((".png", ".jpg", ".gif", ".jpeg", ".svg")) and url not in image_urls:
            image_urls.append(url)
        if text:
            for image_url in reg.findall(text):
                if image_url not in image_urls:
                    image_urls.append(image_url)
    return image_urls


def save_assistant_message(state: dict) -> None:
    """助手回答写入历史记录。"""
    history_repository.save_message(
        session_id=state["session_id"],
        role="assistant",
        text=state.get("answer"),
        rewritten_query=state.get("rewritten_query") or state.get("original_query"),
        item_names=state.get("item_names", []),
        image_urls=state.get("image_urls", []),
    )


def produce_answer(state: dict) -> dict:
    """
    答案生成主流程（node_answer_output 业务实现）。

    已有答案（寒暄/澄清）直接返回；否则组装 Prompt → 生成 → 提取图片 →
    检索不足检测 → 保存历史。
    """
    if not try_return_existing_answer(state):
        reranked_docs, rewritten_query, history = validate_generation_inputs(state)
        prompt = build_answer_prompt(reranked_docs, rewritten_query, history, state)
        generate_answer(state, prompt)
        state["image_urls"] = extract_image_urls(reranked_docs)
        check_rag_fallback_needed(
            reranked_docs, state.get("session_id", ""), state.get("original_query", "")
        )
    save_assistant_message(state)
    return state
