"""
输出解析器（原生实现，不依赖 langchain）。

与 lm_utils 的 invoke_with_parser 配合使用，兼容两种 LLM 客户端：
- LangChain ChatOpenAI（安装 langchain 时自动优先）
- 原生 OpenAI SDK 客户端（默认，零额外依赖）
"""
import json
import re
from typing import Any


class StrOutputParser:
    """把模型响应对象解析为纯文本。"""

    def invoke(self, response: Any) -> str:
        content = getattr(response, "content", None)
        if content is None:
            return str(response)
        return str(content)


class JsonOutputParser:
    """把模型响应文本解析为 JSON 对象（兼容围栏与多余文本）。"""

    def invoke(self, response: Any) -> dict:
        text = getattr(response, "content", None)
        if text is None:
            text = str(response)
        text = str(text).strip()
        fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        try:
            data = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            brace = re.search(r"\{.*\}", text, re.DOTALL)
            if not brace:
                raise ValueError(f"模型输出不是合法 JSON：{text[:200]}")
            data = json.loads(brace.group(0))
        if not isinstance(data, dict):
            raise ValueError(f"模型输出 JSON 不是对象：{text[:200]}")
        return data
