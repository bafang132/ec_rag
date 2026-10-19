"""
语音交互服务模块（可选能力）：STT 语音转文字 + TTS 文字转语音。

经 OpenAI 兼容接口调用；未配置时接口返回明确错误，不影响主链路。
"""
import os
from typing import Optional

from app.shared.config.lm_config import lm_config
from app.shared.runtime.logger import logger


def transcribe_from_upload(file) -> str:
    """
    语音转文字（STT）。

    :param file: FastAPI UploadFile 对象
    :return: 识别文本
    """
    if not lm_config.api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY，语音识别不可用")
    from openai import OpenAI

    client = OpenAI(api_key=lm_config.api_key, base_url=lm_config.base_url or None)
    audio = client.audio.transcriptions.create(
        model=os.getenv("STT_MODEL", "whisper-1"),
        file=(file.filename, file.file),
    )
    return audio.text or ""


def text_to_speech(text: str) -> bytes:
    """
    文字转语音（TTS），返回 WAV 音频字节。

    :param text: 输入文本
    :return: 音频字节
    """
    if not lm_config.api_key:
        raise RuntimeError("未配置 OPENAI_API_KEY，语音合成不可用")
    from openai import OpenAI

    client = OpenAI(api_key=lm_config.api_key, base_url=lm_config.base_url or None)
    response = client.audio.speech.create(
        model=os.getenv("TTS_MODEL", "tts-1"),
        voice=os.getenv("TTS_VOICE", "nova"),
        input=text,
    )
    return response.read()
