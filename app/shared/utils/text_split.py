"""
纯 Python 递归文本切分器（中文语义优先）。

不依赖 langchain，保证任何环境可用；优先在语义边界（段落 > 换行 > 中文标点 > 英文标点 > 空格）
切分，尽量保持块内语义完整。C 加速路径见 c_accel.py（加载自检通过后自动启用）。
"""
from typing import List

# 分割符优先级（从粗到细）
DEFAULT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "]


def _find_split_point(text: str, separators: List[str], target: int) -> int:
    """
    在 target 位置之前寻找最靠后的语义分割点。

    :return: 分割位置（分隔符之后）；找不到返回 -1
    """
    best_pos = -1
    for sep in separators:
        pos = text.rfind(sep, 0, target)
        if pos != -1 and pos + len(sep) > best_pos:
            best_pos = pos + len(sep)
    return best_pos


def split_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50,
               separators: List[str] | None = None) -> List[str]:
    """
    把长文本递归切分为不超过 chunk_size 的块（允许重叠 chunk_overlap 字符）。

    :param text: 待切分文本
    :param chunk_size: 单块最大字符数
    :param chunk_overlap: 相邻块重叠字符数（保语义连续）
    :param separators: 分割符优先级列表
    :return: 切分后的文本块列表（已去空白，无空块）
    """
    if not text or not text.strip():
        return []
    if chunk_size <= 0:
        return [text.strip()]
    separators = separators or DEFAULT_SEPARATORS

    chunks: List[str] = []
    remaining = text
    guard = 0
    max_iterations = max(1, (len(text) // max(1, chunk_size // 2)) + 10)

    while remaining.strip():
        guard += 1
        if guard > max_iterations * 4:
            # 防御性退出：异常输入（如超长无分隔符文本）下避免死循环
            tail = remaining.strip()
            if tail:
                chunks.append(tail)
            break

        if len(remaining) <= chunk_size:
            block = remaining.strip()
            if block:
                chunks.append(block)
            break

        split_pos = _find_split_point(remaining, separators, chunk_size)
        if split_pos <= 0:
            split_pos = chunk_size  # 找不到语义边界，硬切

        block = remaining[:split_pos].strip()
        if block:
            chunks.append(block)

        # 下一块起点：保留 overlap 字符重叠
        next_start = max(0, split_pos - max(0, chunk_overlap))
        if next_start >= len(remaining):
            break
        remaining = remaining[next_start:]

    return chunks


if __name__ == "__main__":
    demo = (
        "# 亚马逊 FBA 仓储规则\n\n"
        "FBA 卖家需要遵守平台的仓储容量限制。IPI 分数低于 500 的账户将面临仓储限制，"
        "超出限制的入库计划可能被拒收。建议定期清理滞销库存，避免产生长期仓储费。\n\n"
        "旺季期间（如 Prime Day、黑五），平台会临时上调仓储费，卖家应提前 30 天备货。"
    )
    for i, c in enumerate(split_text(demo, chunk_size=80, chunk_overlap=10), 1):
        print(f"--- chunk {i} ({len(c)} chars) ---")
        print(c)
