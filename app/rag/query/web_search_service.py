"""
联网搜索服务模块：知识库不足时补充外部证据（可选能力）。

默认关闭；配置 MCP_DASHSCOPE_BASE_URL 后启用 DashScope WebSearch MCP。
未配置时所有接口安全降级，不影响主链路。
"""
from typing import Optional

from app.shared.config.common import env_str
from app.shared.runtime.logger import logger

_WEB_SEARCH_TIMEOUT = 15


def is_web_search_configured() -> bool:
    """联网搜索是否可用（配置了 MCP 地址且总开关开启）。"""
    from app.shared.config.ec_config import WEB_SEARCH_ENABLED

    return bool(WEB_SEARCH_ENABLED and env_str("MCP_DASHSCOPE_BASE_URL", ""))


def web_search(query: str, limit: int = 5) -> list[dict]:
    """
    执行联网搜索，返回统一格式的文档列表。

    :param query: 检索查询
    :param limit: 返回条数
    :return: [{title, snippet, url}]；未配置或失败返回空列表
    """
    if not is_web_search_configured():
        logger.info("联网搜索未配置，跳过")
        return []
    try:
        return _search_via_dashscope_mcp(query, limit)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"联网搜索失败：{exc}")
        return []


def _search_via_dashscope_mcp(query: str, limit: int) -> list[dict]:
    """经 DashScope WebSearch MCP 执行搜索（HTTP JSON-RPC 简化实现）。"""
    import requests

    base_url = env_str("MCP_DASHSCOPE_BASE_URL", "")
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "web_search",
            "arguments": {"query": query, "count": limit},
        },
    }
    resp = requests.post(base_url, json=payload, timeout=_WEB_SEARCH_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    content = data.get("result", {}).get("content", [])
    docs: list[dict] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            # MCP 文本结果通常是 JSON 字符串，尽力解析
            import json

            try:
                parsed = json.loads(item.get("text", "[]"))
                for entry in parsed.get("results", parsed if isinstance(parsed, list) else []):
                    docs.append({
                        "title": entry.get("title", ""),
                        "snippet": entry.get("snippet", "") or entry.get("abstract", ""),
                        "url": entry.get("url", ""),
                    })
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue
    logger.info(f"联网搜索完成：{len(docs)} 条结果")
    return docs[:limit]
