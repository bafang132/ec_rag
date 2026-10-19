"""
跨境电商元数据抽取服务模块：从导入文档抽取平台/市场/类目/文档类型等结构化元数据，
回写到每个切片，供检索阶段做语义增强与过滤召回。

设计范式：模型识别 + 关键词兜底 + 归一化到受控枚举。
"""
from pathlib import Path

from app.rag.import_.config import (
    METADATA_CONTEXT_CHUNK_K,
    METADATA_CONTEXT_TOTAL_MAX_CHARS,
)
from app.shared.config.ec_config import (
    CATEGORIES,
    DOC_TYPES,
    MARKETS,
    PLATFORMS,
    UNKNOWN_VALUE,
)
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger
from app.shared.utils.ec_utils import (
    CATEGORY_KEYWORDS,
    MARKET_KEYWORDS,
    PLATFORM_KEYWORDS,
    match_enum_by_keywords,
    normalize_enum,
    safe_json_loads,
)
from app.infra.llm import llm_provider
from app.shared.model.lm_utils import build_messages, invoke_text

# 作者常见前缀，用于从正文首行推断作者
import re

AUTHOR_PATTERNS = [
    re.compile(r"作\s*者\s*[:：]\s*(.+)"),
    re.compile(r"文\s*[:：／/]\s*(.+)"),
    re.compile(r"编\s*者\s*[:：]\s*(.+)"),
    re.compile(r"撰\s*稿\s*[:：]\s*(.+)"),
    re.compile(r"来源\s*[:：]\s*(.+)"),
]

# 文档类型关键词规则（dict 形式，供 match_enum_by_keywords 使用）
DOC_TYPE_KEYWORD_RULES_DICT: dict[str, list[str]] = {
    "平台政策": ["政策", "条款", "协议", "规定", "官方", "公告"],
    "规则解读": ["解读", "解析", "分析", "详解", "指南", "手册"],
    "实操教程": ["教程", "步骤", "如何", "怎么", "操作", "实战", "方法", "流程"],
    "案例复盘": ["案例", "复盘", "实例", "经历", "分享", "实录"],
    "术语百科": ["术语", "百科", "名词", "词汇", "缩写", "含义", "是什么"],
    "工具模板": ["模板", "工具", "表格", "清单", "checklist"],
}


def validate_metadata_input(state: dict) -> tuple[list[dict], str]:
    """校验切片与文件标题。"""
    chunks = state.get("chunks", [])
    file_title = state.get("file_title") or ""
    if not chunks:
        raise ValueError("chunks 没有内容，无法抽取元数据")
    if not file_title:
        logger.warning("file_title 为空，使用默认值")
        file_title = "default_title"
        state["file_title"] = file_title
    return chunks, file_title


def build_metadata_context(chunks: list[dict]) -> str:
    """取前若干个切片拼成抽取上下文（截断到最大字符数）。"""
    parts = []
    for index, item in enumerate(chunks[:METADATA_CONTEXT_CHUNK_K], start=1):
        parts.append(f"切片:{index},标题:{item.get('title', '')},内容:{item.get('content', '')}")
    return "\n".join(parts)[:METADATA_CONTEXT_TOTAL_MAX_CHARS]


def extract_metadata_with_llm(context: str, file_title: str) -> dict:
    """
    调用大模型一次性抽取跨境电商结构化元数据（失败返回空字典，走规则兜底）。
    """
    try:
        client = llm_provider.chat(json_mode=True)
        system_prompt = load_prompt("ec_metadata_system", file_title=file_title)
        user_prompt = load_prompt(
            "ec_metadata_extract", file_title=file_title, context=context
        )
        raw = invoke_text(client, build_messages(system_prompt, user_prompt))
        return safe_json_loads(raw) or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"元数据模型抽取失败，将使用规则兜底：{exc}")
        return {}


def _match_by_priority(texts: list[str], rules: list[tuple[str, list[str]]]) -> str:
    """按文本优先级依次匹配关键词规则（文件名优先，语义密度更高）。"""
    for text in texts:
        if not text:
            continue
        for value, keywords in rules:
            for kw in keywords:
                if kw in text:
                    return value
    return ""


def guess_author(text: str) -> str:
    """从正文前 20 行按「作者:xxx」等前缀推断作者。"""
    for line in text.split("\n")[:20]:
        line = line.strip().lstrip("#").strip()
        if not line:
            continue
        for pattern in AUTHOR_PATTERNS:
            match = pattern.match(line)
            if match:
                author = match.group(1).strip()
                if author and len(author) <= 30:
                    return author
    return ""


def _match_by_priority(texts: list[str], rules: dict[str, list[str]]) -> str:
    """
    按文本优先级依次匹配关键词规则，返回首个命中结果。

    文件名通常由用户显式命名，语义密度高于正文，因此排在前面优先匹配——
    避免正文中提到的边缘工具（如清货渠道"站外Deal"）覆盖文档真实主题。
    """
    for text in texts:
        if not text:
            continue
        matched = match_enum_by_keywords(text, rules)
        if matched != UNKNOWN_VALUE:
            return matched
    return UNKNOWN_VALUE


def build_fallback_metadata(md_content: str, file_title: str) -> dict:
    """基于关键词与文件名的规则兜底抽取（模型不可用时保证元数据可用）。"""
    # 文件名权重最高，其次取正文前 3000 字（长文档尾部噪声较大）
    priority_texts = [file_title, md_content[:3000]]
    return {
        "article_title": file_title,
        "author": guess_author(md_content[:1000]),
        "platform": _match_by_priority(priority_texts, PLATFORM_KEYWORDS),
        "market": _match_by_priority(priority_texts, MARKET_KEYWORDS),
        "category": _match_by_priority(priority_texts, CATEGORY_KEYWORDS),
        "doc_type": _match_by_priority(priority_texts, DOC_TYPE_KEYWORD_RULES_DICT),
    }


def normalize_metadata(raw_metadata: dict, state: dict, file_title: str) -> dict:
    """合并模型抽取与规则兜底，输出合法完整的元数据。"""
    md_content = state.get("md_content", "")
    fallback = build_fallback_metadata(md_content, file_title)

    def pick(model_value, fallback_value, candidates, default=UNKNOWN_VALUE):
        """
        模型值优先，但只做严格枚举归一；非法/缺失时直接用规则兜底值。

        注意：不能对模型值调用带关键词猜测的 normalize_platform 等函数——
        模型输出的垃圾值（如"乱填的类目"）可能因含关键字（"类目"）被误判。
        """
        strict = normalize_enum(model_value or "", candidates)
        if strict != default:
            return strict
        return fallback_value or default

    platform = pick(raw_metadata.get("platform"), fallback["platform"], PLATFORMS)
    market = pick(raw_metadata.get("market"), fallback["market"], MARKETS)
    category = pick(raw_metadata.get("category"), fallback["category"], CATEGORIES)
    doc_type = pick(raw_metadata.get("doc_type"), fallback["doc_type"], DOC_TYPES)

    article_title = (
        str(raw_metadata.get("article_title") or "").strip()
        or fallback["article_title"] or file_title
    )
    author = str(raw_metadata.get("author") or "").strip() or fallback["author"]

    local_file_path = state.get("local_file_path", "")
    source_file = Path(local_file_path).name if local_file_path else f"{file_title}.md"

    metadata = {
        "article_title": article_title,
        "author": author or UNKNOWN_VALUE,
        "platform": platform,
        "market": market,
        "category": category,
        "doc_type": doc_type,
        "source_file": source_file,
        "source_path": local_file_path or state.get("md_path", ""),
    }
    logger.info(f"元数据抽取完成：platform={platform}, market={market}, "
                f"category={category}, doc_type={doc_type}")
    return metadata


def apply_metadata(chunks: list[dict], metadata: dict) -> list[dict]:
    """把元数据回写到每个切片。"""
    for chunk in chunks:
        chunk.update(metadata)
    return chunks


def extract_and_apply_metadata(state: dict) -> dict:
    """
    元数据抽取主入口（node_metadata_extract 业务实现）。

    流程：模型抽取 → 规则兜底 → 归一化 → 回写切片。
    """
    chunks, file_title = validate_metadata_input(state)
    context = build_metadata_context(chunks)
    raw_metadata = extract_metadata_with_llm(context, file_title)
    metadata = normalize_metadata(raw_metadata, state, file_title)
    state["ec_metadata"] = metadata
    state["chunks"] = apply_metadata(chunks, metadata)
    return state
