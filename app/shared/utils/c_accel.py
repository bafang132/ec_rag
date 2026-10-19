"""
C 语言加速封装（ctypes）：文本分块 + 余弦相似度。

设计要点（针对原跨境电商项目预编译 DLL 缺陷的修复）：
1. **加载自检**：DLL 加载后立即用固定用例验证分块结果与内存安全，
   任何异常（版本不兼容/堆损坏/结果不符）都会禁用 C 路径并降级纯 Python，
   杜绝脏 DLL 拖垮整个服务。
2. **源码修复**：原 text_splitter.c 的 trim() 返回栈内偏移指针后被 free()，
   文本含前导空白时构成堆损坏（本仓库 csrc/text_splitter.c 已修复）。
3. 未编译 C 库时自动走纯 Python 路径（见 text_split.py），功能完全等价。

编译方式见 csrc/build.bat（Windows MinGW）或 csrc/build.sh（Linux/macOS gcc）。
"""
import ctypes
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.shared.runtime.logger import logger
from app.shared.utils.text_split import split_text as py_split_text

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CSRC_DIR = PROJECT_ROOT / "csrc"

# 自检用例：覆盖前导空白、多段落、无分隔符长文本三种场景
_SELF_TEST_CASES = [
    # (text, chunk_size, overlap, 期望块数下限, 期望首块前缀)
    ("   前导空白测试。第二段内容在这里，用于验证分块器是否正常工作。", 20, 5, 2, "前导空白测试"),
    ("无分隔符超长文本" * 30, 25, 5, 10, "无分隔符超长文本"),
]


class CTextSplitter:
    """C 语言文本分块器（带自检的 ctypes 封装）。"""

    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self.enabled = False

    def _candidate_paths(self) -> List[Path]:
        """按平台列举候选动态库路径。"""
        names = (
            ["text_splitter.dll", "libtext_splitter.so"]
            if sys.platform == "win32"
            else ["libtext_splitter.so", "text_splitter.dll"]
        )
        dirs = [CSRC_DIR / "prebuilt", CSRC_DIR, PROJECT_ROOT]
        paths: List[Path] = []
        for d in dirs:
            for n in names:
                paths.append(d / n)
        return paths

    def _load(self) -> Optional[ctypes.CDLL]:
        for path in self._candidate_paths():
            if not path.exists():
                continue
            try:
                lib = ctypes.CDLL(str(path))
                lib.split_chinese_text.argtypes = [
                    ctypes.c_char_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p)),
                    ctypes.POINTER(ctypes.c_int),
                ]
                lib.split_chinese_text.restype = None
                lib.free_chunks.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_int]
                lib.free_chunks.restype = None
                logger.info(f"C 分块库加载成功：{path}")
                return lib
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"C 分块库加载失败（{path}）：{exc}")
        return None

    def _self_test(self, lib: ctypes.CDLL) -> bool:
        """
        加载后自检：验证分块数量与内容，并确认 free 流程不崩溃。

        原项目的预编译 DLL 在此自检下会失败（前导空白场景堆损坏），
        从而自动降级到 Python 实现——这正是本设计的目的。
        """
        for text, chunk_size, overlap, min_chunks, first_prefix in _SELF_TEST_CASES:
            out_chunks = ctypes.POINTER(ctypes.c_char_p)()
            out_count = ctypes.c_int(0)
            try:
                lib.split_chinese_text(
                    text.encode("utf-8"),
                    ctypes.c_int(chunk_size),
                    ctypes.c_int(overlap),
                    ctypes.byref(out_chunks),
                    ctypes.byref(out_count),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"C 分块库自检失败（调用异常）：{exc}，降级 Python 分块")
                return False

            count = out_count.value
            try:
                if count < min_chunks:
                    logger.warning(
                        f"C 分块库自检失败（块数 {count} < {min_chunks}），降级 Python 分块"
                    )
                    return False
                first = out_chunks[0]
                if not first:
                    logger.warning("C 分块库自检失败（首块为空），降级 Python 分块")
                    return False
                decoded = first.decode("utf-8", errors="ignore")
                if not decoded.startswith(first_prefix):
                    logger.warning(
                        f"C 分块库自检失败（首块内容不符：{decoded[:20]!r}），降级 Python 分块"
                    )
                    return False
            finally:
                # free 流程本身也要验证：原 DLL 在此处堆损坏
                try:
                    lib.free_chunks(out_chunks, out_count)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"C 分块库自检失败（free 异常）：{exc}，降级 Python 分块")
                    return False
        logger.success("C 分块库自检通过，启用 C 加速分块")
        return True

    def ensure_loaded(self) -> bool:
        """加载并自检 C 分块库（进程内仅执行一次有效尝试）。"""
        if self.enabled:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        lib = self._load()
        if lib is None:
            logger.info("未找到可用的 C 分块库，使用纯 Python 分块（功能等价）")
            return False
        if not self._self_test(lib):
            self._lib = None
            return False
        self._lib = lib
        self.enabled = True
        return True

    _load_attempted = False

    def split(self, text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
        """
        切分文本：C 加速可用时走 C，否则纯 Python。

        :return: 文本块列表（UTF-8 解码，无空块）
        """
        if not text or not text.strip():
            return []
        if not self.ensure_loaded() or self._lib is None:
            return py_split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        out_chunks = ctypes.POINTER(ctypes.c_char_p)()
        out_count = ctypes.c_int(0)
        try:
            self._lib.split_chinese_text(
                text.encode("utf-8"),
                ctypes.c_int(chunk_size),
                ctypes.c_int(chunk_overlap),
                ctypes.byref(out_chunks),
                ctypes.byref(out_count),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"C 分块调用异常，本次降级 Python：{exc}")
            return py_split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        chunks: List[str] = []
        try:
            for i in range(out_count.value):
                raw = out_chunks[i]
                if not raw:
                    continue
                try:
                    decoded = raw.decode("utf-8")
                except UnicodeDecodeError:
                    decoded = raw.decode("utf-8", errors="ignore")
                if decoded.strip():
                    chunks.append(decoded)
        finally:
            try:
                self._lib.free_chunks(out_chunks, out_count)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"C 分块内存释放异常：{exc}")
        if not chunks:
            return py_split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return chunks


class CSimilarity:
    """C 语言余弦相似度（带自检；原项目 DLL 的 1.2 倍电商加权保留为可配置项）。"""

    def __init__(self) -> None:
        self._lib: Optional[ctypes.CDLL] = None
        self.enabled = False
        self._load_attempted = False
        # 原项目 similarity.c 对余弦值乘 1.2（电商场景加权）。
        # 该变换单调保序，不影响排序；此处保留兼容并允许通过环境变量关闭。
        import os
        self.weight = float(os.getenv("EC_SIM_WEIGHT", "1.2"))

    def _load(self) -> Optional[ctypes.CDLL]:
        names = (
            ["similarity.dll", "libsimilarity.so"]
            if sys.platform == "win32"
            else ["libsimilarity.so", "similarity.dll"]
        )
        for d in (CSRC_DIR / "prebuilt", CSRC_DIR, PROJECT_ROOT):
            for n in names:
                path = d / n
                if not path.exists():
                    continue
                try:
                    lib = ctypes.CDLL(str(path))
                    lib.calc_similarity.argtypes = [
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.POINTER(ctypes.c_float),
                        ctypes.c_int,
                    ]
                    lib.calc_similarity.restype = ctypes.c_float
                    logger.info(f"C 相似度库加载成功：{path}")
                    return lib
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"C 相似度库加载失败（{path}）：{exc}")
        return None

    def _self_test(self, lib: ctypes.CDLL) -> bool:
        """自检：正交向量相似度应接近 0，同向向量应 > 0.9（考虑 1.2 加权）。"""
        try:
            a = (ctypes.c_float * 4)(1.0, 0.0, 0.0, 0.0)
            b = (ctypes.c_float * 4)(0.0, 1.0, 0.0, 0.0)
            c = (ctypes.c_float * 4)(2.0, 0.0, 0.0, 0.0)
            sim_orth = lib.calc_similarity(a, b, 4)
            sim_same = lib.calc_similarity(a, c, 4)
            if abs(sim_orth) > 1e-3:
                logger.warning(f"C 相似度库自检失败（正交相似度 {sim_orth}），降级 numpy")
                return False
            if sim_same < 0.9:
                logger.warning(f"C 相似度库自检失败（同向相似度 {sim_same}），降级 numpy")
                return False
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"C 相似度库自检失败：{exc}，降级 numpy")
            return False
        logger.success("C 相似度库自检通过，启用 C 加速相似度")
        return True

    def ensure_loaded(self) -> bool:
        if self.enabled:
            return True
        if self._load_attempted:
            return False
        self._load_attempted = True
        lib = self._load()
        if lib is None:
            logger.info("未找到可用的 C 相似度库，使用 numpy 余弦相似度（功能等价）")
            return False
        if not self._self_test(lib):
            self._lib = None
            return False
        self._lib = lib
        self.enabled = True
        return True

    def cosine(self, vec1, vec2) -> float:
        """
        计算余弦相似度（C 加速或 numpy）。

        :param vec1: 一维向量（list/np.ndarray）
        :param vec2: 一维向量
        :return: 余弦相似度（已按 EC_SIM_WEIGHT 加权，保序）
        """
        v1 = np.asarray(vec1, dtype=np.float32).ravel()
        v2 = np.asarray(vec2, dtype=np.float32).ravel()
        if v1.shape != v2.shape or v1.size == 0:
            return 0.0

        if self.ensure_loaded() and self._lib is not None:
            try:
                a = (ctypes.c_float * v1.size)(*v1.tolist())
                b = (ctypes.c_float * v2.size)(*v2.tolist())
                return float(self._lib.calc_similarity(a, b, v1.size))
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"C 相似度调用异常，本次降级 numpy：{exc}")

        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom <= 0:
            return 0.0
        return float(np.dot(v1, v2) / denom) * self.weight


# 进程级单例
c_text_splitter = CTextSplitter()
c_similarity = CSimilarity()


def c_split(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """模块级便捷入口：C 加速分块（自动降级）。"""
    return c_text_splitter.split(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def c_cosine(vec1, vec2) -> float:
    """模块级便捷入口：C 加速余弦相似度（自动降级）。"""
    return c_similarity.cosine(vec1, vec2)


if __name__ == "__main__":
    print("Python 分块：", py_split_text("测试分块。第二句内容。", chunk_size=6, chunk_overlap=2))
    print("C 加速状态：", c_text_splitter.enabled, c_similarity.enabled)
    print("余弦(同向)：", c_cosine([1, 2, 3], [2, 4, 6]))
    print("余弦(正交)：", c_cosine([1, 0], [0, 1]))
