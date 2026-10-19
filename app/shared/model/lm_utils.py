"""
LLM 客户端工具模块：OpenAI 兼容协议，双实现自动选择。

优先使用 LangChain ChatOpenAI（技术栈一致，安装 langchain-openai 后生效）；
未安装时使用原生 openai SDK 客户端，接口完全兼容（invoke/stream），
保证任何环境（含离线测试）都能运行问答链路。
"""
from typing import Iterator, List, Optional, Union

from app.shared.config.lm_config import lm_config
from app.shared.model.parsers import JsonOutputParser, StrOutputParser
from app.shared.runtime.logger import logger

_DEFAULT_LLM_MODEL = "qwen-flash"
_DEFAULT_TEMPERATURE = 0.1
_llm_client_cache: dict[tuple[str, bool], object] = {}


# ===================== 原生 OpenAI SDK 客户端 =====================

class _NativeResponse:
    """模拟 LangChain 响应对象（.content）。"""

    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class _NativeChunk:
    """模拟 LangChain 流式块对象（.content）。"""

    __slots__ = ("content",)

    def __init__(self, content: str):
        self.content = content


class NativeChatClient:
    """
    原生 OpenAI 兼容客户端（零 langchain 依赖）。

    接口与 LangChain ChatOpenAI 对齐：
    - invoke(messages_or_prompt) -> _NativeResponse(.content)
    - stream(prompt) -> Iterator[_NativeChunk(.content)]
    """

    def __init__(self, model: str, temperature: float, api_key: str, base_url: str,
                 timeout: int, max_retries: int):
        from openai import OpenAI

        self.model = model
        self.temperature = temperature
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    @staticmethod
    def _to_api_messages(messages) -> List[dict]:
        """把 str / dict / (type, content) 对象统一转为 OpenAI messages 格式。"""
        if isinstance(messages, str):
            return [{"role": "user", "content": messages}]
        api_messages: List[dict] = []
        for m in messages:
            if isinstance(m, dict):
                api_messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            else:
                role = getattr(m, "type", None) or getattr(m, "role", None) or "user"
                content = getattr(m, "content", str(m))
                # LangChain 消息类型 system/human/ai -> OpenAI role
                role_map = {"system": "system", "human": "user", "ai": "assistant"}
                api_messages.append({"role": role_map.get(role, role), "content": content})
        return api_messages

    def invoke(self, messages, **kwargs):
        api_messages = self._to_api_messages(messages)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=self.temperature,
            **kwargs,
        )
        return _NativeResponse(resp.choices[0].message.content or "")

    def stream(self, messages, **kwargs) -> Iterator[_NativeChunk]:
        api_messages = self._to_api_messages(messages)
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=api_messages,
            temperature=self.temperature,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            text = getattr(delta, "content", None) if delta is not None else None
            if text:
                yield _NativeChunk(text)


# ===================== 客户端工厂 =====================

def get_llm_client(model: Optional[str] = None, json_mode: bool = False):
    """
    获取带全局缓存的 LLM 客户端实例。

    优先 LangChain ChatOpenAI，不可用时原生 openai SDK 客户端（接口一致）。

    :param model: 模型名；为空用配置的默认模型
    :param json_mode: 是否要求 JSON 输出（结构化抽取场景）
    :return: LLM 客户端（invoke/stream 接口）
    """
    target_model = model or lm_config.llm_model or _DEFAULT_LLM_MODEL
    cache_key = (target_model, json_mode)
    if cache_key in _llm_client_cache:
        logger.debug(f"[LLM客户端] 缓存命中：模型={target_model}，JSON模式={json_mode}")
        return _llm_client_cache[cache_key]

    if not lm_config.api_key:
        raise ValueError("[LLM客户端] 配置缺失：请在 .env 配置 OPENAI_API_KEY（大模型API密钥）")
    if not lm_config.base_url:
        raise ValueError("[LLM客户端] 配置缺失：请在 .env 配置 OPENAI_BASE_URL（API接口基础地址）")

    # ---- 优先 LangChain 实现（技术栈一致） ----
    try:
        from langchain_openai import ChatOpenAI

        model_kwargs = {}
        if json_mode:
            model_kwargs["response_format"] = {"type": "json_object"}
        client = ChatOpenAI(
            model=target_model,
            temperature=lm_config.llm_temperature or _DEFAULT_TEMPERATURE,
            api_key=lm_config.api_key,
            base_url=lm_config.base_url,
            timeout=lm_config.llm_timeout,
            max_retries=lm_config.llm_max_retries,
            model_kwargs=model_kwargs,
        )
        logger.info(f"[LLM客户端] LangChain ChatOpenAI 初始化成功：模型={target_model}")
        _llm_client_cache[cache_key] = client
        return client
    except ImportError:
        logger.info("[LLM客户端] 未安装 langchain-openai，使用原生 openai SDK 客户端")
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"[LLM客户端] LangChain 初始化失败，回退原生客户端：{exc}")

    # ---- 原生实现 ----
    client = NativeChatClient(
        model=target_model,
        temperature=lm_config.llm_temperature or _DEFAULT_TEMPERATURE,
        api_key=lm_config.api_key,
        base_url=lm_config.base_url,
        timeout=lm_config.llm_timeout,
        max_retries=lm_config.llm_max_retries,
    )
    logger.info(f"[LLM客户端] 原生 openai SDK 客户端初始化成功：模型={target_model}")
    _llm_client_cache[cache_key] = client
    return client


# ===================== 统一调用助手（兼容两种客户端） =====================

def invoke_text(client, messages, **kwargs) -> str:
    """调用 LLM 并返回纯文本。"""
    resp = client.invoke(messages, **kwargs)
    return getattr(resp, "content", str(resp))


def invoke_json(client, messages, **kwargs) -> dict:
    """调用 LLM 并解析为 JSON 对象（失败抛 ValueError）。"""
    resp = client.invoke(messages, **kwargs)
    return JsonOutputParser().invoke(resp)


def stream_text(client, messages, **kwargs) -> Iterator[str]:
    """流式调用 LLM，逐步产出文本增量。"""
    for chunk in client.stream(messages, **kwargs):
        text = getattr(chunk, "content", None)
        if text:
            yield text


def build_messages(system: str, user: str) -> List[dict]:
    """构造 OpenAI 格式的 system/user 消息对。"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


if __name__ == "__main__":
    # 本地连通性测试：python -m app.shared.model.lm_utils
    try:
        c = get_llm_client()
        print(invoke_text(c, build_messages("你是助手", "用一句话介绍你自己")))
    except Exception as e:
        print(f"LLM 调用失败（请检查 .env 配置）：{e}")
