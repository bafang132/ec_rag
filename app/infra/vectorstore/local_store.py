"""
本地向量库（numpy 实现，零外部依赖，开箱即用）。

定位：Milvus 不可用时的默认向量库，功能对齐 Milvus 混合检索的核心链路：
- 稠密路：查询向量与库存向量的余弦相似度（faiss 可用时自动加速，否则 numpy 暴力检索）
- 稀疏路：对库存内容做 BM25 关键词打分（不依赖 Embedding 提供方是否输出稀疏向量）
- 融合：两路分数 min-max 归一化后按权重加权（默认 dense 0.9 / sparse 0.1，与 Milvus 加权策略一致）
- 标量过滤：支持 item_name / platform / market / category 等元数据等值过滤
  （默认关闭，见 retrieval_config.use_scalar_filter）

持久化格式（persist_dir）：
- vectors.npy : (N, dim) float32 稠密矩阵
- meta.json   : 切片元数据列表（与向量行一一对应，chunk_id 自增）

线程安全：所有读写操作持进程级 RLock。
"""
import json
import math
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from app.shared.model.embedding_utils import tokenize
from app.shared.runtime.logger import logger
from app.shared.utils.path_util import ensure_dir


class LocalVectorStore:
    """本地 numpy 向量库。"""

    backend = "local"

    def __init__(self, persist_dir: str | Path, dim: int = 1024) -> None:
        # 相对路径基于项目根解析，保证从任何工作目录启动行为一致
        persist_path = Path(persist_dir)
        if not persist_path.is_absolute():
            from app.shared.utils.path_util import PROJECT_ROOT

            persist_path = PROJECT_ROOT / persist_path
        self.persist_dir = ensure_dir(persist_path)
        self.dim = dim
        self._lock = threading.RLock()
        self._vectors: Optional[np.ndarray] = None   # (N, dim) float32
        self._meta: list[dict] = []                  # 与向量行对齐的元数据
        self._next_id = 1
        self._doc_freq: dict[str, int] = {}          # BM25 文档频率
        self._doc_len: list[int] = []                # BM25 文档长度
        self._avgdl: float = 1.0
        self._load()

    # ---------------------- 持久化 ----------------------

    @property
    def _vectors_path(self) -> Path:
        return self.persist_dir / "vectors.npy"

    @property
    def _meta_path(self) -> Path:
        return self.persist_dir / "meta.json"

    def _load(self) -> None:
        """从磁盘加载向量库（不存在则初始化为空库）。"""
        with self._lock:
            if self._meta_path.exists():
                try:
                    self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
                    self._next_id = max((int(m.get("chunk_id", 0)) for m in self._meta), default=0) + 1
                except (json.JSONDecodeError, OSError) as exc:
                    logger.warning(f"本地向量库元数据损坏，重置为空库：{exc}")
                    self._meta = []
            if self._vectors_path.exists() and self._meta:
                try:
                    self._vectors = np.load(self._vectors_path)
                    if self._vectors.shape[0] != len(self._meta):
                        logger.warning(
                            f"向量库维度不一致（vectors={self._vectors.shape[0]}, meta={len(self._meta)}），重建为空库"
                        )
                        self._vectors, self._meta = None, []
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"本地向量库向量损坏，重建为空库：{exc}")
                    self._vectors, self._meta = None, []
            self._rebuild_bm25()
            logger.info(
                f"本地向量库加载完成：{len(self._meta)} 个切片，dim={self.dim}，目录={self.persist_dir}"
            )

    def _save(self) -> None:
        """全量持久化（KB 规模下全量写可接受；调用方已持锁）。"""
        if self._vectors is not None and len(self._meta):
            np.save(self._vectors_path, self._vectors)
            self._meta_path.write_text(
                json.dumps(self._meta, ensure_ascii=False), encoding="utf-8"
            )
        elif not self._meta:
            # 空库：清理旧文件，保证删除全部条目后重启不残留
            for path in (self._vectors_path, self._meta_path):
                if path.exists():
                    path.unlink()

    def _rebuild_bm25(self) -> None:
        """重建 BM25 统计量（文档频率/长度）。"""
        self._doc_freq = {}
        self._doc_len = []
        total_len = 0
        for m in self._meta:
            tokens = tokenize(m.get("content", ""))
            self._doc_len.append(len(tokens))
            total_len += len(tokens)
            for tok in set(tokens):
                self._doc_freq[tok] = self._doc_freq.get(tok, 0) + 1
        self._avgdl = (total_len / len(self._meta)) if self._meta else 1.0

    # ---------------------- 写入 ----------------------

    def add_chunks(self, chunks: list[dict]) -> int:
        """
        批量写入切片（每个 chunk 需含 dense_vector 与 content）。

        :param chunks: 切片列表（含 dense_vector/item_name/content 等元数据）
        :return: 实际写入条数
        """
        if not chunks:
            return 0
        with self._lock:
            new_vectors = []
            new_meta = []
            for chunk in chunks:
                dense = chunk.get("dense_vector")
                if not dense:
                    logger.warning("切片缺少 dense_vector，跳过")
                    continue
                vector = np.asarray(dense, dtype=np.float32)
                if vector.shape[0] != self.dim:
                    logger.warning(
                        f"向量维度 {vector.shape[0]} 与库维度 {self.dim} 不一致，跳过该切片"
                    )
                    continue
                meta = {k: v for k, v in chunk.items() if k not in ("dense_vector", "sparse_vector")}
                meta["chunk_id"] = self._next_id
                self._next_id += 1
                new_vectors.append(vector)
                new_meta.append(meta)

            if not new_vectors:
                return 0

            matrix = np.vstack(new_vectors)
            if self._vectors is None or not len(self._meta):
                self._vectors = matrix
                self._meta = new_meta
            else:
                self._vectors = np.vstack([self._vectors, matrix])
                self._meta.extend(new_meta)
            self._rebuild_bm25()
            self._save()
            logger.info(f"本地向量库写入 {len(new_vectors)} 个切片，总数 {len(self._meta)}")
            return len(new_vectors)

    def delete_by_item(self, item_name: str) -> int:
        """按 item_name 删除全部切片。"""
        with self._lock:
            keep_idx = [i for i, m in enumerate(self._meta) if m.get("item_name") != item_name]
            deleted = len(self._meta) - len(keep_idx)
            if deleted:
                self._meta = [self._meta[i] for i in keep_idx]
                if self._meta and self._vectors is not None:
                    self._vectors = self._vectors[keep_idx]
                else:
                    self._vectors = None
                self._rebuild_bm25()
                self._save()
            logger.info(f"本地向量库删除主体[{item_name}]切片 {deleted} 个")
            return deleted

    def delete_by_chunk_id(self, chunk_id: int) -> int:
        """按 chunk_id 删除单个切片。"""
        with self._lock:
            keep_idx = [i for i, m in enumerate(self._meta) if int(m.get("chunk_id", -1)) != int(chunk_id)]
            deleted = len(self._meta) - len(keep_idx)
            if deleted:
                self._meta = [self._meta[i] for i in keep_idx]
                if self._meta and self._vectors is not None:
                    self._vectors = self._vectors[keep_idx]
                else:
                    self._vectors = None
                self._rebuild_bm25()
                self._save()
            return deleted

    # ---------------------- 检索 ----------------------

    @staticmethod
    def _minmax(scores: np.ndarray) -> np.ndarray:
        """min-max 归一化；全同值时返回全 1（避免除零）。"""
        if scores.size == 0:
            return scores
        lo, hi = float(scores.min()), float(scores.max())
        if hi - lo < 1e-9:
            return np.ones_like(scores)
        return (scores - lo) / (hi - lo)

    def _bm25_scores(self, query_tokens: list[str], candidate_idx: Optional[np.ndarray] = None) -> np.ndarray:
        """计算候选切片的 BM25 分数（k1=1.5, b=0.75）。"""
        n_docs = len(self._meta)
        scores = np.zeros(n_docs, dtype=np.float32)
        if n_docs == 0 or not query_tokens:
            return scores
        k1, b = 1.5, 0.75
        q_terms = set(query_tokens)
        for i in range(n_docs):
            if candidate_idx is not None and i not in set(candidate_idx.tolist()):
                continue
            tokens = tokenize(self._meta[i].get("content", ""))
            if not tokens:
                continue
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            dl = self._doc_len[i] if i < len(self._doc_len) else len(tokens)
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                df = self._doc_freq.get(term, 0)
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                score += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / max(self._avgdl, 1e-6)))
            scores[i] = score
        return scores

    def _dense_scores(self, dense_vector: list[float]) -> np.ndarray:
        """计算稠密余弦相似度（向量入库时已由提供方归一化，此处再归一一次保底）。"""
        n_docs = len(self._meta)
        if n_docs == 0 or self._vectors is None:
            return np.zeros(0, dtype=np.float32)
        q = np.asarray(dense_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        v_norms = np.linalg.norm(self._vectors, axis=1, keepdims=True)
        v_norms[v_norms == 0] = 1.0
        normalized = self._vectors / v_norms
        return normalized @ q

    def search(
        self,
        dense_vector: list[float],
        query_text: str = "",
        sparse_vector: Optional[dict] = None,
        expr: Optional[str] = None,
        limit: int = 15,
        output_fields: Optional[list[str]] = None,
        dense_weight: float = 0.9,
    ) -> list[dict]:
        """
        混合检索：稠密余弦 + BM25 关键词，加权融合后返回 Top-N。

        :param dense_vector: 查询稠密向量
        :param query_text: 查询原始文本（BM25 稀疏路分词用；为空则仅稠密路）
        :param sparse_vector: 查询稀疏向量（保留参数以兼容 Milvus 接口；本地库以 BM25 实现稀疏路）
        :param expr: 简单的元数据等值过滤表达式，如 "platform == '亚马逊' and category == '物流仓储'"
        :param limit: 返回条数
        :param output_fields: 返回的元数据字段（默认全部）
        :param dense_weight: 稠密路权重（稀疏路为 1 - dense_weight）
        :return: 归一化切片列表（含 score）
        """
        with self._lock:
            if not self._meta:
                return []

            # 1) 标量过滤（默认关闭，由上层控制是否传 expr）
            candidate_idx: Optional[np.ndarray] = None
            if expr:
                predicate = self._compile_expr(expr)
                if predicate is not None:
                    mask = np.array([predicate(m) for m in self._meta], dtype=bool)
                    candidate_idx = np.where(mask)[0]
                    if candidate_idx.size == 0:
                        return []

            # 2) 两路打分（min-max 归一化后加权融合，策略与 Milvus 加权 ranker 一致）
            dense_scores = self._dense_scores(dense_vector)
            sparse_weight = 1.0 - dense_weight
            query_tokens = tokenize(query_text) if query_text else []
            if sparse_weight > 0 and query_tokens:
                bm25 = self._bm25_scores(query_tokens, candidate_idx)
                bm25_norm = self._minmax(bm25)
                dense_norm = self._minmax(dense_scores)
                final_scores = dense_weight * dense_norm + sparse_weight * bm25_norm
            else:
                final_scores = self._minmax(dense_scores)

            # 3) 候选中排序取 Top-N
            if candidate_idx is not None:
                cand_scores = final_scores[candidate_idx]
                order = np.argsort(-cand_scores)[:limit]
                top_idx = candidate_idx[order]
            else:
                order = np.argsort(-final_scores)[:limit]
                top_idx = order

            fields = output_fields
            results = []
            for rank, idx in enumerate(top_idx, start=1):
                meta = self._meta[int(idx)]
                item = dict(meta)
                if fields:
                    item = {k: v for k, v in item.items() if k in fields or k == "chunk_id"}
                item["score"] = round(float(final_scores[int(idx)]), 6)
                item["rank"] = rank
                results.append(item)
            return results

    @staticmethod
    def _compile_expr(expr: str) -> Optional[Callable[[dict], bool]]:
        """
        把简单的 Milvus 风格等值表达式编译为谓词函数。

        支持：field == 'value' / field != 'value'，and 连接；无法解析时返回 None（不过滤）。
        """
        expr = (expr or "").strip()
        if not expr:
            return None
        conditions: list[Callable[[dict], bool]] = []
        for part in expr.split(" and "):
            part = part.strip()
            if "==" in part:
                field, value = part.split("==", 1)
                value = value.strip().strip("'\"")
                conditions.append(lambda m, f=field.strip(), v=value: str(m.get(f, "")) == v)
            elif "!=" in part:
                field, value = part.split("!=", 1)
                value = value.strip().strip("'\"")
                conditions.append(lambda m, f=field.strip(), v=value: str(m.get(f, "")) != v)
            else:
                logger.warning(f"本地向量库无法解析过滤表达式片段：{part!r}，已忽略")
                return None
        return lambda m: all(c(m) for c in conditions)

    # ---------------------- 管理查询 ----------------------

    def query_all(
        self,
        predicate: Optional[Callable[[dict], bool]] = None,
        output_fields: Optional[list[str]] = None,
        limit: int = 10000,
    ) -> list[dict]:
        """全量/条件查询切片（管理页聚合用）。"""
        with self._lock:
            results = []
            for meta in self._meta[:limit]:
                if predicate and not predicate(meta):
                    continue
                item = dict(meta)
                if output_fields:
                    item = {k: v for k, v in item.items() if k in output_fields or k == "chunk_id"}
                results.append(item)
            return results

    def list_items(self, limit: int = 10000) -> list[dict]:
        """按 item_name 聚合主体列表（管理页）。"""
        with self._lock:
            items: dict[str, dict] = {}
            for meta in self._meta[:limit]:
                name = meta.get("item_name") or ""
                if not name:
                    continue
                if name not in items:
                    items[name] = {
                        "item_name": name,
                        "file_title": meta.get("file_title", ""),
                        "title": meta.get("title", ""),
                        "platform": meta.get("platform", ""),
                        "market": meta.get("market", ""),
                        "category": meta.get("category", ""),
                        "doc_type": meta.get("doc_type", ""),
                        "chunk_count": 0,
                        "char_count": 0,
                    }
                items[name]["chunk_count"] += 1
                items[name]["char_count"] += len(meta.get("content", ""))
            return list(items.values())

    def count(self) -> int:
        """切片总数。"""
        with self._lock:
            return len(self._meta)

    def is_empty(self) -> bool:
        with self._lock:
            return not self._meta
