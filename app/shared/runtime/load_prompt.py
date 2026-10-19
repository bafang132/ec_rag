"""
提示词加载模块：从 app/source/prompts 读取 .prompt 模板并渲染占位符。

模板中的 JSON 花括号需以双写形式 {{ }} 转义，传参渲染时由 str.format 还原。
"""
from pathlib import Path

from app.shared.runtime.logger import logger
from app.shared.utils.path_util import PROJECT_ROOT


def load_prompt(name: str, **kwargs) -> str:
    """
    加载提示词并渲染变量占位符。

    :param name: 提示词文件名（不带 .prompt 后缀，如 ec_query_rewrite_and_intent）
    :param kwargs: 需渲染的变量键值对
    :return: 渲染后的最终提示词字符串
    """
    prompt_path = PROJECT_ROOT / "app" / "source" / "prompts" / f"{name}.prompt"

    if not prompt_path.exists():
        raise FileNotFoundError(f"提示词文件不存在：{prompt_path.absolute()}")

    raw_prompt = prompt_path.read_text(encoding="utf-8")

    if kwargs:
        rendered_prompt = raw_prompt.format(**kwargs)
        logger.debug(f"提示词渲染成功，替换变量：{list(kwargs.keys())}")
        return rendered_prompt
    return raw_prompt


if __name__ == "__main__":
    print(load_prompt("ec_query_rewrite_and_intent", history_text="无", query="亚马逊FBA运费怎么算？"))
