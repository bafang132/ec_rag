"""
文档切块服务模块：Markdown 按标题语义粗切 + 超长切分/过短合并精修。

切分策略：
1. 按 Markdown 标题（H1-H6）做语义级粗切，保证一个主题块落在一起；
2. 超长章节用 C 加速分块器（加载自检失败自动降级纯 Python）二次切分；
3. 过短章节按「同父标题」合并，减少碎片化提升检索效果。
"""
import json
import re
from pathlib import Path
from typing import Any

from app.rag.import_.config import CHUNK_MAX_SIZE, CHUNK_OVERLAP, CHUNK_SIZE
from app.shared.runtime.logger import logger
from app.shared.utils.c_accel import c_split
from app.shared.utils.text_split import split_text


def load_markdown_content(state: dict) -> tuple[str, str]:
    """从状态读取 Markdown 正文与文件标题（缺失时从 md_path 重读）。"""
    md_content = state.get("md_content", "")
    file_title = state.get("file_title", "")
    md_path = state.get("md_path", "")

    if not md_content:
        logger.warning("state 中无 md_content，尝试从 md_path 重读")
        if md_path:
            md_content = Path(md_path).read_text(encoding="utf-8")
            state["md_content"] = md_content
        if not md_content:
            raise ValueError("md_content 无数据且 md_path 重读失败，终止执行")

    if not file_title:
        file_title = Path(md_path).stem if md_path else "default"
        state["file_title"] = file_title

    md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
    return md_content, file_title


def split_by_titles(md_content: str, file_title: str) -> list[dict]:
    """按 Markdown 标题做语义级粗切（代码块内的 # 不视为标题）。"""
    reg = re.compile(r"^\s*#{1,6}\s.+")
    lines = md_content.split("\n")
    chunks: list[dict] = []
    current_title = None
    current_lines: list[str] = []
    is_code_block = False
    title_count = 0

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("```") or line.startswith("~~~"):
            is_code_block = not is_code_block
            current_lines.append(line)
            continue
        if reg.match(line) and not is_code_block:
            if current_title and len(current_lines) > 1:
                chunks.append({
                    "content": "\n".join(current_lines),
                    "title": current_title,
                    "file_title": file_title,
                })
            current_title = line
            current_lines = [current_title]
            title_count += 1
        else:
            current_lines.append(line)

    if current_title:
        chunks.append({
            "content": "\n".join(current_lines),
            "title": current_title,
            "file_title": file_title,
        })
    if title_count == 0:
        # 无标题文档：整体作为一个章节，交给精修阶段处理
        chunks.append({"content": md_content, "title": "default", "file_title": file_title})
    logger.info(f"标题语义粗切完成：{len(chunks)} 个章节")
    return chunks


def _split_long_section(section: dict[str, Any], max_length: int = CHUNK_MAX_SIZE) -> list[dict[str, Any]]:
    """
    超长章节二次切分：C 加速分块器（自动降级纯 Python），保留标题上下文。
    """
    content = section.get("content", "") or ""
    if len(content) <= max_length:
        return [section]

    content = content.replace("\r\n", "\n").replace("\r", "\n")
    title = section.get("title", "") or ""
    prefix = f"{title}\n\n" if title else ""
    available_len = max_length - len(prefix)
    if available_len <= 0:
        logger.warning(f"章节标题过长，无法切分：{title[:20]}...")
        return [section]

    # 清理正文开头重复的标题
    body = content
    if title and body.lstrip().startswith(title):
        body = body[body.find(title) + len(title):].lstrip()

    # C 加速分块（自检失败自动降级纯 Python 递归切分）
    sub_texts = c_split(body, chunk_size=available_len, chunk_overlap=CHUNK_OVERLAP)
    if not sub_texts:
        sub_texts = split_text(body, chunk_size=available_len, chunk_overlap=CHUNK_OVERLAP)

    sub_sections: list[dict[str, Any]] = []
    for idx, text in enumerate(sub_texts, start=1):
        text = text.strip()
        if not text:
            continue
        sub_sections.append({
            "title": f"{title}-{idx}" if title else f"chunk-{idx}",
            "content": (prefix + text).strip(),
            "parent_title": title,
            "part": idx,
            "file_title": section.get("file_title"),
        })
    logger.debug(f"超长章节切分完成：{title} → {len(sub_sections)} 个子块")
    return sub_sections


def _merge_short_sections(
    sections: list[dict[str, Any]],
    min_length: int = CHUNK_SIZE,
    max_length: int = CHUNK_MAX_SIZE,
) -> list[dict[str, Any]]:
    """
    过短章节合并：仅合并「同父标题」且长度不足阈值的相邻块，避免跨章节合并。
    """
    if not sections:
        return []
    merged: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for sec in sections:
        if current is None:
            current = sec
            continue
        current_content = current.get("content", "")
        is_current_short = len(current_content) < min_length
        is_same_parent = current.get("parent_title") == sec.get("parent_title")

        if is_current_short and is_same_parent:
            parent_title = sec.get("parent_title", "")
            next_content = sec["content"]
            if parent_title and next_content.startswith(parent_title):
                next_content = next_content[len(parent_title):].lstrip()
            merged_content = current_content + "\n\n" + next_content
            if max_length > 0 and len(merged_content) > max_length:
                merged.append(current)
                current = sec
                continue
            current["content"] = merged_content
            if "part" in sec:
                current["part"] = sec["part"]
        else:
            merged.append(current)
            current = sec

    if current is not None:
        merged.append(current)
    return merged


def refine_chunks(
    sections: list[dict],
    max_len: int = CHUNK_MAX_SIZE,
    min_len: int = CHUNK_SIZE,
) -> list[dict]:
    """Chunk 精细化：长切短合 + 父标题兜底。"""
    if not max_len or max_len <= 0:
        logger.warning(f"Chunk 最大长度配置无效（{max_len}），跳过精细化")
        return sections

    refined: list[dict] = []
    for sec in sections:
        refined.extend(_split_long_section(sec, max_len))
    logger.info(f"超长章节切分完成：{len(refined)} 个初始子块")

    final = _merge_short_sections(refined, min_length=min_len, max_length=max_len)
    logger.info(f"过短章节合并完成：最终 {len(final)} 个 Chunk")

    for sec in final:
        if not isinstance(sec, dict):
            continue
        if "part" not in sec:
            sec["part"] = 0
        if not sec.get("parent_title"):
            sec["parent_title"] = sec.get("title") or ""
    return final


def backup_chunks(chunks: list[dict], md_path: str) -> None:
    """切分结果落一份 chunks.json，方便排查切片质量。"""
    try:
        chunks_json_path = Path(md_path).parent / "chunks.json"
        chunks_json_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=4), encoding="utf-8")
    except OSError as exc:
        logger.warning(f"chunks.json 备份失败：{exc}")


def split_document(state: dict) -> dict:
    """
    文档切分主入口（node_document_split 业务实现）。

    流程：读取 Markdown → 标题粗切 → 精细化（长切短合）→ 备份 → 写回 state。
    """
    md_content, file_title = load_markdown_content(state)
    chunks = split_by_titles(md_content, file_title)
    chunks = refine_chunks(chunks, max_len=CHUNK_MAX_SIZE, min_len=CHUNK_SIZE)
    if state.get("md_path"):
        backup_chunks(chunks, state["md_path"])
    state["chunks"] = chunks
    logger.info(f"文档切分完成：{file_title} → {len(chunks)} 个 Chunk")
    return state
