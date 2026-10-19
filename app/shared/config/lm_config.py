"""
LLM 配置模块：OpenAI 兼容接口的模型与调用参数。
"""
from dataclasses import dataclass

from app.shared.config.common import env_float, env_int, env_str


@dataclass
class LMConfig:
    api_key: str = env_str("OPENAI_API_KEY", "")
    base_url: str = env_str("OPENAI_BASE_URL", "")
    llm_model: str = env_str("LLM_DEFAULT_MODEL", "qwen-flash")
    vl_model: str = env_str("VL_MODEL", "qwen3-vl-flash")
    llm_temperature: float = env_float("LLM_DEFAULT_TEMPERATURE", 0.1)
    llm_timeout: int = env_int("LLM_TIMEOUT", 60)
    llm_max_retries: int = env_int("LLM_MAX_RETRIES", 2)


lm_config = LMConfig()
