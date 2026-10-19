"""
并行搜索包装模块：向量检索与联网搜索并发执行，减少总耗时。
"""
import threading
from typing import Optional


def run_in_background(target, args=None, kwargs=None) -> threading.Thread:
    """在后台线程中运行函数，不阻塞当前线程。"""
    args = args or ()
    kwargs = kwargs or {}
    t = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    t.start()
    return t


class _ParallelSearchResult:
    """线程安全的并行搜索结果容器。"""

    def __init__(self):
        self.vector_result: Optional[list[dict]] = None
        self.web_result: Optional[list[dict]] = None
        self.vector_done = threading.Event()
        self.web_done = threading.Event()

    def set_vector(self, result: list[dict]) -> None:
        self.vector_result = result
        self.vector_done.set()

    def set_web(self, result: list[dict]) -> None:
        self.web_result = result
        self.web_done.set()


def parallel_search(
    *,
    rewritten_query: str,
    ec_filters: Optional[dict] = None,
    scene: str = "",
    limit: int = 15,
    use_web_search: bool = False,
) -> tuple[list[dict], Optional[list[dict]]]:
    """
    并行执行向量检索与联网搜索（总耗时 ≈ max(两者) 而非和）。

    :return: (向量检索结果, 联网搜索结果或 None)
    """
    from app.rag.query.search_service import search_ec_chunks
    from app.rag.query.web_search_service import web_search
    from app.shared.runtime.logger import logger

    result = _ParallelSearchResult()

    def _vector_search():
        try:
            res = search_ec_chunks(
                rewritten_query=rewritten_query,
                ec_filters=ec_filters,
                scene=scene,
                limit=limit,
            )
            result.set_vector(res)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"向量检索失败：{exc}", exc_info=True)
            result.set_vector([])

    def _web_search_fn():
        try:
            res = web_search(rewritten_query)
            result.set_web(res)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"联网搜索失败：{exc}")
            result.set_web([])

    vector_thread = run_in_background(_vector_search)

    web_docs: Optional[list[dict]] = None
    if use_web_search:
        web_thread = run_in_background(_web_search_fn)
        vector_thread.join()
        web_thread.join(timeout=15.0)
        web_docs = result.web_result
    else:
        vector_thread.join()

    return result.vector_result or [], web_docs
