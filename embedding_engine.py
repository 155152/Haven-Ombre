# ============================================================
# Module: Embedding Engine (embedding_engine.py)
# 模块：向量化引擎
#
# Generates embeddings via Gemini API (OpenAI-compatible),
# stores them in SQLite, and provides cosine similarity search.
# 通过 Gemini API（OpenAI 兼容）生成 embedding，
# 存储在 SQLite 中，提供余弦相似度搜索。
#
# Depended on by: server.py, bucket_manager.py
# 被谁依赖：server.py, bucket_manager.py
# ============================================================

import os
import json
import math
import sqlite3
import logging
import asyncio
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger("ombre_brain.embedding")


class EmbeddingEngine:
    """
    Embedding generation + SQLite vector storage + cosine search.
    向量生成 + SQLite 向量存储 + 余弦搜索。
    """

    def __init__(self, config: dict):
        dehy_cfg = config.get("dehydration", {})
        embed_cfg = config.get("embedding", {})

        self.api_key = embed_cfg.get("api_key") or dehy_cfg.get("api_key", "")
        self.base_url = (
            embed_cfg.get("base_url")
            or dehy_cfg.get("base_url")
            or "https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        self.model = embed_cfg.get("model", "gemini-embedding-001")
        self.enabled = bool(self.api_key) and embed_cfg.get("enabled", True)
        self.max_chars = self._int_between(embed_cfg.get("max_chars", 6000), 6000, 500, 32000)
        self.query_instruction = str(
            embed_cfg.get("query_instruction")
            or "Given a memory search query, retrieve relevant long-term memory passages."
        ).strip()
        self.document_instruction = str(embed_cfg.get("document_instruction") or "").strip()

        # --- SQLite path: buckets_dir/embeddings.db ---
        db_path = os.path.join(config["buckets_dir"], "embeddings.db")
        self.db_path = db_path

        # --- Initialize client ---
        if self.enabled:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=30.0,
            )
        else:
            self.client = None

        # --- Initialize SQLite ---
        self._init_db()
        # Semantic search used to JSON-decode every stored vector and calculate
        # cosine similarity in Python on every query. Cache one normalized NumPy
        # matrix per process and rebuild only when the SQLite store changes.
        self._search_cache_signature: tuple[int, int, int, int] | None = None
        self._search_cache_bucket_ids: list[str] = []
        self._search_cache_matrix: np.ndarray | None = None

    def _init_db(self):
        """Create embeddings table if not exists."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                bucket_id TEXT PRIMARY KEY,
                embedding TEXT NOT NULL,
                model TEXT,
                dimension INTEGER,
                updated_at TEXT NOT NULL
            )
        """)
        self._ensure_column(conn, "embeddings", "model", "TEXT")
        self._ensure_column(conn, "embeddings", "dimension", "INTEGER")
        conn.commit()
        conn.close()

    async def generate_and_store(self, bucket_id: str, content: str) -> bool:
        """
        Generate embedding for content and store in SQLite.
        为内容生成 embedding 并存入 SQLite。
        Returns True on success, False on failure.
        """
        if not self.enabled or not content or not content.strip():
            return False

        try:
            embedding = await self._generate_embedding(content, kind="document")
            if not embedding:
                return False
            self._store_embedding(bucket_id, embedding)
            return True
        except Exception as e:
            logger.warning(f"Embedding generation failed for {bucket_id}: {e}")
            return False

    async def embed_text(self, text: str, *, kind: str = "document") -> list[float]:
        """Generate one embedding without storing it; used by auxiliary retrieval indexes."""
        if not self.enabled or not str(text or "").strip():
            return []
        return await self._generate_embedding(text, kind=kind)

    async def _generate_embedding(self, text: str, *, kind: str = "document") -> list[float]:
        """Call API to generate embedding vector."""
        # Truncate to avoid token limits
        prepared = self._prepare_embedding_input(text, kind=kind)
        truncated = prepared[: self.max_chars]
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=truncated,
            )
            if response.data and len(response.data) > 0:
                return response.data[0].embedding
            return []
        except Exception as e:
            logger.warning(f"Embedding API call failed: {e}")
            return []

    def _store_embedding(self, bucket_id: str, embedding: list[float]):
        """Store embedding in SQLite."""
        from utils import now_iso
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            """
            INSERT OR REPLACE INTO embeddings (bucket_id, embedding, model, dimension, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (bucket_id, json.dumps(embedding), self.model, len(embedding), now_iso()),
        )
        conn.commit()
        conn.close()
        self._invalidate_search_cache()

    def delete_embedding(self, bucket_id: str):
        """Remove embedding when bucket is deleted."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM embeddings WHERE bucket_id = ?", (bucket_id,))
        conn.commit()
        conn.close()
        self._invalidate_search_cache()

    async def get_embedding(self, bucket_id: str) -> list[float] | None:
        """Retrieve stored embedding for a bucket. Returns None if not found."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT embedding, model, dimension FROM embeddings WHERE bucket_id = ?", (bucket_id,)
        ).fetchone()
        conn.close()
        if row:
            try:
                embedding = json.loads(row[0])
                if not self._row_matches_current_model(row[1], row[2], embedding):
                    return None
                return embedding
            except json.JSONDecodeError:
                return None
        return None

    async def get_embeddings(self, bucket_ids: list[str]) -> dict[str, list[float]]:
        """Retrieve stored embeddings for several buckets with one SQLite read."""
        unique_ids = list(
            dict.fromkeys(
                str(item or "").strip()
                for item in bucket_ids
                if str(item or "").strip()
            )
        )
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                f"SELECT bucket_id, embedding, model, dimension FROM embeddings WHERE bucket_id IN ({placeholders})",
                unique_ids,
            ).fetchall()
        finally:
            conn.close()

        output: dict[str, list[float]] = {}
        for bucket_id, payload, model, dimension in rows:
            try:
                embedding = json.loads(payload)
            except (json.JSONDecodeError, TypeError):
                continue
            if self._row_matches_current_model(model, dimension, embedding):
                output[str(bucket_id)] = embedding
        return output

    async def search_similar(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        Search for buckets similar to query text using a cached normalized matrix.
        Returns list of (bucket_id, similarity_score) sorted by score desc.
        搜索与查询文本相似的桶。向量矩阵仅在 SQLite 变化时重建。
        """
        if not self.enabled:
            return []

        try:
            query_embedding = await self._generate_embedding(query, kind="query")
            if not query_embedding:
                return []
        except Exception as e:
            logger.warning(f"Query embedding failed: {e}")
            return []

        bucket_ids, matrix = self._search_index()
        if matrix is None or not bucket_ids or matrix.size == 0:
            return []

        query_vector = np.asarray(query_embedding, dtype=np.float32)
        if query_vector.ndim != 1 or matrix.shape[1] != query_vector.shape[0]:
            return []
        norm = float(np.linalg.norm(query_vector))
        if norm <= 0:
            return []
        query_vector = query_vector / norm
        scores = matrix @ query_vector
        count = max(0, min(int(top_k), len(bucket_ids)))
        if count <= 0:
            return []
        if count >= len(bucket_ids):
            indices = np.argsort(scores)[::-1]
        else:
            indices = np.argpartition(scores, len(scores) - count)[-count:]
            indices = indices[np.argsort(scores[indices])[::-1]]
        return [(bucket_ids[int(index)], float(scores[int(index)])) for index in indices[:count]]

    def warm_search_cache(self) -> int:
        """Build the normalized semantic matrix before the first user recall."""
        bucket_ids, _matrix = self._search_index(force=True)
        return len(bucket_ids)

    def _search_index(self, *, force: bool = False) -> tuple[list[str], np.ndarray | None]:
        signature = self._embedding_store_signature()
        if (
            not force
            and self._search_cache_matrix is not None
            and self._search_cache_signature == signature
        ):
            return self._search_cache_bucket_ids, self._search_cache_matrix

        conn = sqlite3.connect(self.db_path)
        try:
            rows = conn.execute(
                "SELECT bucket_id, embedding, model, dimension FROM embeddings"
            ).fetchall()
        finally:
            conn.close()

        bucket_ids: list[str] = []
        vectors: list[np.ndarray] = []
        expected_dimension: int | None = None
        for bucket_id, emb_json, model, dimension in rows:
            try:
                stored_embedding = json.loads(emb_json)
                if not self._row_matches_current_model(model, dimension, stored_embedding):
                    continue
                vector = np.asarray(stored_embedding, dtype=np.float32)
                if vector.ndim != 1 or vector.size == 0:
                    continue
                if expected_dimension is None:
                    expected_dimension = int(vector.size)
                if int(vector.size) != expected_dimension:
                    continue
                norm = float(np.linalg.norm(vector))
                if norm <= 0:
                    continue
                bucket_ids.append(str(bucket_id))
                vectors.append(vector / norm)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        matrix = np.vstack(vectors).astype(np.float32, copy=False) if vectors else None
        self._search_cache_bucket_ids = bucket_ids
        self._search_cache_matrix = matrix
        self._search_cache_signature = self._embedding_store_signature()
        return bucket_ids, matrix

    def _invalidate_search_cache(self) -> None:
        self._search_cache_signature = None
        self._search_cache_bucket_ids = []
        self._search_cache_matrix = None

    def _embedding_store_signature(self) -> tuple[int, int, int, int]:
        def stat_pair(path: str) -> tuple[int, int]:
            try:
                stat = os.stat(path)
                return int(stat.st_mtime_ns), int(stat.st_size)
            except OSError:
                return 0, 0

        db_mtime, db_size = stat_pair(self.db_path)
        wal_mtime, wal_size = stat_pair(f"{self.db_path}-wal")
        return db_mtime, db_size, wal_mtime, wal_size

    def _prepare_embedding_input(self, text: str, *, kind: str) -> str:
        raw = str(text or "")
        if kind == "query" and self.query_instruction:
            return f"Instruct: {self.query_instruction}\nQuery: {raw}"
        if kind == "document" and self.document_instruction:
            return f"Instruct: {self.document_instruction}\nDocument: {raw}"
        return raw

    def _row_matches_current_model(self, model: str | None, dimension: int | None, embedding: list[float]) -> bool:
        if not embedding:
            return False
        if model != self.model:
            return False
        try:
            stored_dimension = int(dimension)
        except (TypeError, ValueError):
            return False
        return stored_dimension == len(embedding)

    @staticmethod
    def _ensure_column(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
        if any(row[1] == column for row in rows):
            return
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    @staticmethod
    def _int_between(value, default: int, min_value: int, max_value: int) -> int:
        try:
            number = int(value)
        except (TypeError, ValueError):
            number = default
        return max(min_value, min(max_value, number))

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if len(a) != len(b) or not a:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
