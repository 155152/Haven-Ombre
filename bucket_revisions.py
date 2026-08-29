from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any


class BucketRevisionStore:
    """Append-only history for bucket body replacements."""

    def __init__(self, config: dict):
        config = config or {}
        revision_cfg = (
            config.get("bucket_revisions", {})
            if isinstance(config.get("bucket_revisions", {}), dict)
            else {}
        )
        state_dir = config.get("state_dir") or os.path.join(
            os.path.dirname(os.path.abspath(config.get("buckets_dir", "buckets"))),
            "state",
        )
        self.db_path = str(
            revision_cfg.get("db_path")
            or os.path.join(state_dir, "bucket_revisions.sqlite")
        )
        parent_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(parent_dir, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._connect()
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bucket_revisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bucket_id TEXT NOT NULL,
                    previous_content TEXT NOT NULL,
                    previous_sha256 TEXT NOT NULL,
                    replacement_sha256 TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    changed_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bucket_revisions_bucket_changed
                ON bucket_revisions(bucket_id, changed_at DESC, id DESC)
                """
            )
            conn.commit()
        finally:
            conn.close()

    def preserve_before_replace(
        self,
        bucket_id: str,
        previous_content: str,
        replacement_content: str,
        *,
        reason: str = "bucket_update",
    ) -> bool:
        """
        Persist the current body before a replacement.

        Returns False for a no-op replacement. SQLite errors intentionally
        propagate so callers fail closed instead of overwriting unversioned text.
        """
        previous_text = str(previous_content or "")
        replacement_text = str(replacement_content or "")
        if previous_text == replacement_text:
            return False

        conn = self._connect()
        try:
            conn.execute(
                """
                INSERT INTO bucket_revisions (
                    bucket_id,
                    previous_content,
                    previous_sha256,
                    replacement_sha256,
                    reason,
                    changed_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(bucket_id),
                    previous_text,
                    self._sha256(previous_text),
                    self._sha256(replacement_text),
                    str(reason or "bucket_update"),
                    self._now_iso(),
                ),
            )
            conn.commit()
        finally:
            conn.close()
        return True

    def list_for_bucket(self, bucket_id: str) -> list[dict[str, Any]]:
        conn = self._connect()
        try:
            rows = conn.execute(
                """
                SELECT id, bucket_id, previous_content, previous_sha256,
                       replacement_sha256, reason, changed_at
                FROM bucket_revisions
                WHERE bucket_id = ?
                ORDER BY id ASC
                """,
                (str(bucket_id),),
            ).fetchall()
        finally:
            conn.close()
        return [dict(row) for row in rows]

    @staticmethod
    def _sha256(content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
