"""
项目日志工具类。

优先使用 loguru（安装后自动获得彩色控制台 + 按日切分文件日志）；
未安装 loguru 时自动降级为标准库 logging，接口保持一致
（info/debug/warning/error/exception/success），保证任何环境可运行。
"""
import logging
import os
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOG_DIR = PROJECT_ROOT / "logs"

# -------------------------- 配置读取 --------------------------
LOG_CONSOLE_ENABLE = os.getenv("LOG_CONSOLE_ENABLE", "True").lower() == "true"
LOG_CONSOLE_LEVEL = os.getenv("LOG_CONSOLE_LEVEL", "INFO").upper()
LOG_FILE_ENABLE = os.getenv("LOG_FILE_ENABLE", "True").lower() == "true"
LOG_FILE_LEVEL = os.getenv("LOG_FILE_LEVEL", "INFO").upper()
LOG_FILE_RETENTION = os.getenv("LOG_FILE_RETENTION", "7 days")

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def _build_stdlib_logger() -> logging.Logger:
    """loguru 不可用时的标准库降级实现（接口兼容子集）。"""
    std_logger = logging.getLogger("ec_rag")
    std_logger.setLevel(logging.INFO)
    std_logger.handlers.clear()
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
    )
    if LOG_CONSOLE_ENABLE:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        sh.setLevel(_LEVELS.get(LOG_CONSOLE_LEVEL, 20))
        std_logger.addHandler(sh)
    if LOG_FILE_ENABLE:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(
                LOG_DIR / f"app_{time.strftime('%Y%m%d')}.log", encoding="utf-8"
            )
            fh.setFormatter(fmt)
            fh.setLevel(_LEVELS.get(LOG_FILE_LEVEL, 20))
            std_logger.addHandler(fh)
        except OSError:
            pass
    std_logger.propagate = False

    class _Compat:
        """把 logging.Logger 包装成 loguru 风格接口。"""

        def __init__(self, base: logging.Logger):
            self._base = base

        def _log(self, level: int, msg, *args, **kwargs):
            if args:
                try:
                    msg = str(msg) % args
                except (TypeError, ValueError):
                    pass
            self._base.log(level, str(msg), exc_info=kwargs.get("exc_info", False))

        def debug(self, msg, *args, **kwargs):
            self._log(logging.DEBUG, msg, *args, **kwargs)

        def info(self, msg, *args, **kwargs):
            self._log(logging.INFO, msg, *args, **kwargs)

        def success(self, msg, *args, **kwargs):
            self._log(logging.INFO, msg, *args, **kwargs)

        def warning(self, msg, *args, **kwargs):
            self._log(logging.WARNING, msg, *args, **kwargs)

        def error(self, msg, *args, **kwargs):
            self._log(logging.ERROR, msg, *args, **kwargs)

        def exception(self, msg, *args, **kwargs):
            self._base.exception(str(msg) % args if args else str(msg))

        def critical(self, msg, *args, **kwargs):
            self._log(logging.CRITICAL, msg, *args, **kwargs)

    return _Compat(std_logger)


def _build_loguru_logger():
    """loguru 可用时返回配置好的 loguru logger。"""
    from loguru import logger as loguru_logger

    loguru_logger.remove()
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name: <20}</cyan>:<cyan>{function: <15}</cyan>:<cyan>{line: <4}</cyan> - "
        "<level>{message}</level>"
    )
    if LOG_CONSOLE_ENABLE:
        loguru_logger.add(
            sink=sys.stdout, level=LOG_CONSOLE_LEVEL, format=log_format,
            colorize=True, enqueue=True,
        )
    if LOG_FILE_ENABLE:
        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            loguru_logger.add(
                sink=LOG_DIR / "app_{time:YYYYMMDD}.log",
                level=LOG_FILE_LEVEL, format=log_format,
                rotation="00:00", retention=LOG_FILE_RETENTION,
                encoding="utf-8", enqueue=True, backtrace=True, diagnose=True,
            )
        except OSError:
            pass
    return loguru_logger


try:
    logger = _build_loguru_logger()
except ImportError:  # pragma: no cover - 取决于运行环境
    logger = _build_stdlib_logger()


# -------------------------- 节点/步骤日志装饰器 --------------------------

def _trace_id(state) -> str:
    if isinstance(state, Mapping):
        return str(state.get("session_id") or state.get("task_id") or "-")
    return "-"


def node_log(node_name: str):
    """LangGraph 节点日志装饰器：记录开始/完成/异常与耗时。"""

    def deco(func):
        @wraps(func)
        def wrapper(state, *args, **kwargs):
            trace_id = _trace_id(state)
            start_ts = time.time()
            logger.info(f"[{node_name}] 节点开始，追踪ID={trace_id}")
            try:
                result = func(state, *args, **kwargs)
                cost_ms = int((time.time() - start_ts) * 1000)
                logger.info(f"[{node_name}] 节点完成，追踪ID={trace_id}，耗时={cost_ms}ms")
                return result
            except Exception:
                logger.exception(f"[{node_name}] 节点异常，追踪ID={trace_id}")
                raise

        return wrapper

    return deco


def step_log(step_name: str):
    """服务内步骤日志装饰器：不吞异常，保持原有业务语义。"""

    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_ts = time.time()
            logger.info(f"[{step_name}] 步骤开始")
            try:
                result = func(*args, **kwargs)
                cost_ms = int((time.time() - start_ts) * 1000)
                logger.info(f"[{step_name}] 步骤完成，耗时={cost_ms}ms")
                return result
            except Exception:
                logger.exception(f"[{step_name}] 步骤异常")
                raise

        return wrapper

    return deco
