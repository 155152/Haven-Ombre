#!/usr/bin/env python3
"""Plan and apply a traceable Cyberboss -> Ombre memory backfill.

Dry-run is the default. The planner treats production conversation JSONL as the
canonical raw-dialogue source, uses Stone messages only for coverage auditing
and conservative non-Codex fallback, and preserves existing structured memory
as source-record buckets without rewriting it.

Private Stone experiences are deliberately *not* downgraded into ordinary
source-record buckets because Stone kept them behind a private recall scope.
They remain visible in the manifest as privacy-preserved pending items until an
equivalent Ombre private-memory import path is explicitly chosen.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import frontmatter
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bucket_manager import BucketManager  # noqa: E402
from raw_events import RawEventStore, raw_event_text_looks_injected, strip_raw_client_context  # noqa: E402

RAW_DUPLICATE_TOLERANCE_SECONDS = 15.0
STONE_FALLBACK_SOURCES = frozenset({"cyberboss_runtime_history"})
STRUCTURED_MEMORY_FILES = (
    "memory/personality-anchor-rebuild.md",
    "memory/preferences.md",
    "memory/patterns.md",
    "memory/active-projects.md",
    "memory/auto-captured-notes.md",
    "memory/legacy-operational-rules.md",
)
RUNTIME_AUDIT_FILES = (
    "runtime/main-continuity-packets.json",
    "runtime/session-breath.json",
    "runtime/handoff.md",
)
STONE_RUNTIME_PREFIXES = (
    "<environment_context>",
    "SYSTEM ACTION MODE:",
    "LIFE CONTINUITY",
    "CAPABILITY CONFIGURATION REFRESH",
    "KEYWORD GUIDANCE",
    "AUTONOMOUS CHECK-IN WAKE",
    "Internal Runtime continuity",
    "This item is continuity data only.",
    "This item is injected continuity only.",
)


@dataclass(frozen=True)
class PlanItem:
    source_type: str
    source_path: str
    source_id: str
    event_time: str
    content_hash: str
    target_layer: str
    target_representation: str
    already_imported: bool
    action: str
    reason: str
    content_chars: int
    role: str = ""
    bucket_id: str = ""


@dataclass
class RawCandidate:
    item: PlanItem
    role: str
    text: str
    created_at: str
    conversation_id: str
    source_event_id: str
    metadata: dict[str, Any]


@dataclass
class SourceRecordCandidate:
    item: PlanItem
    content: str
    created_at: str
    metadata: dict[str, Any]
    tags: list[str]


class ExistingRawIndex:
    def __init__(self, rows: Iterable[sqlite3.Row] = ()):
        self.by_role_text: dict[tuple[str, str], list[float | None]] = defaultdict(list)
        for row in rows:
            role = str(row["role"] or "").strip().lower()
            text = strip_raw_client_context(str(row["text"] or "")).strip()
            if role and text:
                self.by_role_text[(role, text)].append(parse_epoch(row["created_at"]))

    def contains_near(self, role: str, text: str, created_at: str, *, tolerance: float = RAW_DUPLICATE_TOLERANCE_SECONDS) -> bool:
        key = (str(role or "").strip().lower(), strip_raw_client_context(str(text or "")).strip())
        candidates = self.by_role_text.get(key, [])
        if not candidates:
            return False
        target = parse_epoch(created_at)
        if target is None:
            return True
        for value in candidates:
            if value is not None and abs(value - target) <= tolerance:
                return True
        return False

    def add(self, role: str, text: str, created_at: str) -> None:
        role_key = str(role or "").strip().lower()
        text_key = strip_raw_client_context(str(text or "")).strip()
        if role_key and text_key:
            self.by_role_text[(role_key, text_key)].append(parse_epoch(created_at))


def sha256_text(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def stable_bucket_id(source_type: str, source_id: str) -> str:
    digest = hashlib.sha256(f"{source_type}\0{source_id}".encode("utf-8")).hexdigest()[:24]
    return f"source_cyberboss_{digest}"


def parse_epoch(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def iso_from_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("Ombre config must be a mapping")
    return value


def open_readonly_sqlite(path: Path) -> sqlite3.Connection:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def load_existing_raw_index(raw_db: Path) -> ExistingRawIndex:
    if not raw_db.exists():
        return ExistingRawIndex()
    conn = open_readonly_sqlite(raw_db)
    try:
        rows = conn.execute("SELECT role, text, created_at FROM raw_events").fetchall()
    finally:
        conn.close()
    return ExistingRawIndex(rows)


_BUCKET_STEM_CACHE: dict[str, set[str]] = {}


def bucket_exists(buckets_dir: Path, bucket_id: str) -> bool:
    if not buckets_dir.exists():
        return False
    cache_key = str(buckets_dir.resolve())
    stems = _BUCKET_STEM_CACHE.get(cache_key)
    if stems is None:
        stems = {path.stem for path in buckets_dir.rglob("*.md") if path.is_file()}
        _BUCKET_STEM_CACHE[cache_key] = stems
    return bucket_id in stems


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except Exception:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def conversation_rows(state_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((state_dir / "conversations").glob("*.jsonl")):
        relative = path.relative_to(state_dir).as_posix()
        for value in read_jsonl(path):
            role = str(value.get("type") or "").strip().lower()
            text = str(value.get("text") or "").strip()
            if role not in {"user", "assistant"} or not text:
                continue
            rows.append({**value, "_source_path": relative})
    return rows


def stone_runtime_noise(text: str) -> bool:
    stripped = str(text or "").lstrip()
    lowered = stripped.lower()
    return any(lowered.startswith(prefix.lower()) for prefix in STONE_RUNTIME_PREFIXES)


def plan_conversations(
    state_dir: Path,
    target_raw: ExistingRawIndex,
    planned_raw: ExistingRawIndex,
) -> tuple[list[PlanItem], list[RawCandidate], list[dict[str, Any]]]:
    items: list[PlanItem] = []
    candidates: list[RawCandidate] = []
    rows = conversation_rows(state_dir)
    for row in rows:
        role = str(row.get("type") or "").strip().lower()
        text = strip_raw_client_context(str(row.get("text") or "")).strip()
        if not text:
            continue
        source_id = str(row.get("id") or row.get("turnId") or sha256_text(f"{role}\0{text}\0{row.get('timestamp', '')}"))
        created_at = str(row.get("timestamp") or "").strip()
        conversation_id = str(row.get("threadId") or "").strip()
        already = target_raw.contains_near(role, text, created_at)
        duplicate_in_plan = planned_raw.contains_near(role, text, created_at)
        internal_wake = role == "user" and stone_runtime_noise(text)
        if already:
            action = "skip"
            reason = "already_in_ombre_raw"
        elif duplicate_in_plan:
            action = "skip"
            reason = "duplicate_in_canonical_conversation_source"
        elif internal_wake:
            action = "skip"
            reason = "canonical_conversation_internal_runtime_context"
        else:
            action = "import"
            reason = "canonical_visible_conversation"
        item = PlanItem(
            source_type="cyberboss_conversation",
            source_path=str(row.get("_source_path") or ""),
            source_id=source_id,
            event_time=created_at,
            content_hash=sha256_text(text),
            target_layer="raw",
            target_representation="raw_event",
            already_imported=already,
            action=action,
            reason=reason,
            content_chars=len(text),
            role=role,
        )
        items.append(item)
        if action == "import":
            meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
            candidate = RawCandidate(
                item=item,
                role=role,
                text=text,
                created_at=created_at,
                conversation_id=conversation_id,
                source_event_id=f"conversation:{source_id}",
                metadata={
                    "backfill_source": "cyberboss_conversation",
                    "provider": str(meta.get("provider") or ""),
                    "runtime_id": str(meta.get("runtimeId") or ""),
                    "turn_id": str(row.get("turnId") or ""),
                },
            )
            candidates.append(candidate)
            planned_raw.add(role, text, created_at)
    return items, candidates, rows


def plan_stone_messages(
    state_dir: Path,
    conversation_history: list[dict[str, Any]],
    target_raw: ExistingRawIndex,
    planned_raw: ExistingRawIndex,
) -> tuple[list[PlanItem], list[RawCandidate]]:
    db_path = state_dir / "stone-memory" / "home" / ".stone_memory" / "stone-memory.db"
    if not db_path.exists():
        return [], []
    visible_index = ExistingRawIndex(
        {
            "role": row.get("type"),
            "text": row.get("text"),
            "created_at": row.get("timestamp"),
        }
        for row in conversation_history
    )
    conn = open_readonly_sqlite(db_path)
    try:
        rows = conn.execute(
            "SELECT message_seq, thread_id, message_id, timestamp, source_date, role, text, source FROM messages ORDER BY message_seq"
        ).fetchall()
    finally:
        conn.close()

    items: list[PlanItem] = []
    candidates: list[RawCandidate] = []
    source_path = db_path.relative_to(state_dir).as_posix()
    for row in rows:
        role = str(row["role"] or "").strip().lower()
        text = strip_raw_client_context(str(row["text"] or "")).strip()
        created_at = str(row["timestamp"] or "").strip()
        stone_source = str(row["source"] or "").strip()
        source_id = f"stone-message:{row['message_seq']}"
        already = target_raw.contains_near(role, text, created_at)
        covered_by_visible = visible_index.contains_near(role, text, created_at)
        covered_by_plan = planned_raw.contains_near(role, text, created_at)
        reason = ""
        action = "skip"
        if already:
            reason = "already_in_ombre_raw"
        elif covered_by_visible or covered_by_plan:
            reason = "covered_by_visible_conversation"
        elif raw_event_text_looks_injected(text, {"role": role}):
            reason = "ombre_injected_context_guard"
        elif stone_runtime_noise(text):
            reason = "stone_internal_runtime_context"
        elif stone_source == "codex":
            reason = "stone_codex_only_unverified_runtime_history"
        elif stone_source not in STONE_FALLBACK_SOURCES:
            reason = "stone_source_not_approved_for_raw_fallback"
        else:
            action = "import"
            reason = "stone_verified_runtime_history_fallback"
        item = PlanItem(
            source_type="stone_message",
            source_path=source_path,
            source_id=source_id,
            event_time=created_at,
            content_hash=sha256_text(text),
            target_layer="raw",
            target_representation="raw_event",
            already_imported=already,
            action=action,
            reason=reason,
            content_chars=len(text),
            role=role,
        )
        items.append(item)
        if action == "import":
            candidates.append(
                RawCandidate(
                    item=item,
                    role=role,
                    text=text,
                    created_at=created_at,
                    conversation_id=str(row["thread_id"] or ""),
                    source_event_id=source_id,
                    metadata={
                        "backfill_source": "stone_message_fallback",
                        "stone_source": stone_source,
                        "stone_message_id": str(row["message_id"] or ""),
                        "source_date": str(row["source_date"] or ""),
                    },
                )
            )
            planned_raw.add(role, text, created_at)
    return items, candidates


def _structured_item(
    *,
    source_type: str,
    source_path: str,
    source_id: str,
    event_time: str,
    content: str,
    buckets_dir: Path,
    reason: str,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> tuple[PlanItem, SourceRecordCandidate]:
    bucket_id = stable_bucket_id(source_type, source_id)
    already = bucket_exists(buckets_dir, bucket_id)
    item = PlanItem(
        source_type=source_type,
        source_path=source_path,
        source_id=source_id,
        event_time=event_time,
        content_hash=sha256_text(content),
        target_layer="source_record",
        target_representation="bucket_source_record",
        already_imported=already,
        action="skip" if already else "import",
        reason="source_record_exists" if already else reason,
        content_chars=len(content),
        bucket_id=bucket_id,
    )
    return item, SourceRecordCandidate(
        item=item,
        content=content,
        created_at=event_time,
        metadata=metadata or {},
        tags=tags or [],
    )


def plan_stone_structured(state_dir: Path, buckets_dir: Path) -> tuple[list[PlanItem], list[SourceRecordCandidate]]:
    db_path = state_dir / "stone-memory" / "home" / ".stone_memory" / "stone-memory.db"
    if not db_path.exists():
        return [], []
    conn = open_readonly_sqlite(db_path)
    try:
        feelings = conn.execute(
            "SELECT id, thread_id, source_date, event_time, order_key, content, summary_mode, importance, source, created_at, updated_at FROM feelings ORDER BY id"
        ).fetchall()
        features = conn.execute(
            "SELECT id, thread_id, source_date, category, content, importance, source, created_at, updated_at FROM features ORDER BY id"
        ).fetchall()
        notebooks = conn.execute(
            "SELECT id, thread_id, topic_id, title, relative_path, visibility, tags_json, body_text, revision, created_at, updated_at FROM notebook_entries ORDER BY id"
        ).fetchall()
    finally:
        conn.close()

    source_path = db_path.relative_to(state_dir).as_posix()
    items: list[PlanItem] = []
    records: list[SourceRecordCandidate] = []
    for row in feelings:
        content = str(row["content"] or "").strip()
        if not content:
            continue
        item, record = _structured_item(
            source_type="stone_feeling",
            source_path=source_path,
            source_id=f"stone-feeling:{row['id']}",
            event_time=str(row["event_time"] or row["source_date"] or row["created_at"] or ""),
            content=content,
            buckets_dir=buckets_dir,
            reason="preserve_stone_derived_memory_as_source_evidence",
            metadata={
                "stone_thread_id": str(row["thread_id"] or ""),
                "source_date": str(row["source_date"] or ""),
                "summary_mode": str(row["summary_mode"] or ""),
                "stone_importance": row["importance"],
                "stone_source": str(row["source"] or ""),
                "stone_updated_at": str(row["updated_at"] or ""),
            },
            tags=["source_record", "cyberboss_backfill", "stone_feeling"],
        )
        items.append(item)
        if item.action == "import":
            records.append(record)

    for row in features:
        content = str(row["content"] or "").strip()
        if not content:
            continue
        category = str(row["category"] or "").strip()
        item, record = _structured_item(
            source_type="stone_feature",
            source_path=source_path,
            source_id=f"stone-feature:{row['id']}",
            event_time=str(row["source_date"] or row["created_at"] or ""),
            content=content,
            buckets_dir=buckets_dir,
            reason="preserve_stone_derived_memory_as_source_evidence",
            metadata={
                "stone_thread_id": str(row["thread_id"] or ""),
                "source_date": str(row["source_date"] or ""),
                "stone_category": category,
                "stone_importance": row["importance"],
                "stone_source": str(row["source"] or ""),
                "stone_updated_at": str(row["updated_at"] or ""),
            },
            tags=["source_record", "cyberboss_backfill", "stone_feature", category] if category else ["source_record", "cyberboss_backfill", "stone_feature"],
        )
        items.append(item)
        if item.action == "import":
            records.append(record)

    for row in notebooks:
        content = str(row["body_text"] or "").strip()
        if not content:
            continue
        item, record = _structured_item(
            source_type="stone_notebook",
            source_path=source_path,
            source_id=f"stone-notebook:{row['id']}",
            event_time=str(row["updated_at"] or row["created_at"] or ""),
            content=content,
            buckets_dir=buckets_dir,
            reason="preserve_stone_notebook_as_source_evidence",
            metadata={
                "stone_thread_id": str(row["thread_id"] or ""),
                "stone_topic_id": str(row["topic_id"] or ""),
                "stone_title": str(row["title"] or ""),
                "stone_relative_path": str(row["relative_path"] or ""),
                "stone_visibility": str(row["visibility"] or ""),
                "stone_revision": row["revision"],
            },
            tags=["source_record", "cyberboss_backfill", "stone_notebook"],
        )
        items.append(item)
        if item.action == "import":
            records.append(record)
    return items, records


def plan_memory_files(state_dir: Path, buckets_dir: Path) -> tuple[list[PlanItem], list[SourceRecordCandidate]]:
    items: list[PlanItem] = []
    records: list[SourceRecordCandidate] = []
    for relative in STRUCTURED_MEMORY_FILES:
        path = state_dir / relative
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        item, record = _structured_item(
            source_type="cyberboss_memory_file",
            source_path=relative,
            source_id=relative,
            event_time=iso_from_mtime(path),
            content=content,
            buckets_dir=buckets_dir,
            reason="preserve_current_memory_file_as_source_evidence",
            metadata={"source_file": relative},
            tags=["source_record", "cyberboss_backfill", "memory_file"],
        )
        items.append(item)
        if item.action == "import":
            records.append(record)

    relationship = state_dir / "memory" / "relationship_state.yaml"
    if relationship.exists():
        content = relationship.read_text(encoding="utf-8").strip()
        items.append(
            PlanItem(
                source_type="cyberboss_memory_file",
                source_path="memory/relationship_state.yaml",
                source_id="memory/relationship_state.yaml",
                event_time=iso_from_mtime(relationship),
                content_hash=sha256_text(content),
                target_layer="audit_only",
                target_representation="retired_compatibility_stub",
                already_imported=False,
                action="skip",
                reason="relationship_state_explicitly_retired_not_recallable",
                content_chars=len(content),
            )
        )

    index_path = state_dir / "memory" / "index.jsonl"
    for row in read_jsonl(index_path):
        source_id = str(row.get("id") or row.get("key") or "").strip()
        content = str(row.get("text") or "").strip()
        if not source_id or not content:
            continue
        status = str(row.get("status") or "").strip().lower()
        category = str(row.get("category") or "").strip()
        if status != "active" or category in {"legacy_retired", "recent_bridge", "relationship_state"}:
            items.append(
                PlanItem(
                    source_type="cyberboss_memory_index",
                    source_path="memory/index.jsonl",
                    source_id=source_id,
                    event_time=str(row.get("createdAt") or row.get("deletedAt") or ""),
                    content_hash=sha256_text(content),
                    target_layer="audit_only",
                    target_representation="legacy_memory_index",
                    already_imported=False,
                    action="skip",
                    reason="inactive_or_ephemeral_legacy_memory_index_item",
                    content_chars=len(content),
                )
            )
            continue
        item, record = _structured_item(
            source_type="cyberboss_memory_index",
            source_path="memory/index.jsonl",
            source_id=source_id,
            event_time=str(row.get("createdAt") or ""),
            content=content,
            buckets_dir=buckets_dir,
            reason="preserve_active_legacy_memory_as_source_evidence",
            metadata={
                "legacy_category": category,
                "legacy_key": str(row.get("key") or ""),
                "legacy_priority": row.get("priority"),
                "legacy_storage": str(row.get("storage") or ""),
            },
            tags=["source_record", "cyberboss_backfill", "legacy_memory", category] if category else ["source_record", "cyberboss_backfill", "legacy_memory"],
        )
        items.append(item)
        if item.action == "import":
            records.append(record)

    promise_path = state_dir / "memory" / "promise-candidates.jsonl"
    for row in read_jsonl(promise_path):
        source_id = str(row.get("id") or "").strip()
        content = str(row.get("summary") or (row.get("source") or {}).get("text") or "").strip()
        if not source_id:
            continue
        confirmed = row.get("confirmed") is True
        status = str(row.get("status") or "").strip().lower()
        action = "skip"
        reason = "unconfirmed_promise_candidate"
        target = "candidate_audit"
        if confirmed and status in {"confirmed", "open", "active", "done", "completed", "closed"} and content:
            item, record = _structured_item(
                source_type="cyberboss_promise",
                source_path="memory/promise-candidates.jsonl",
                source_id=source_id,
                event_time=str((row.get("source") or {}).get("receivedAt") or row.get("createdAt") or row.get("updatedAt") or ""),
                content=content,
                buckets_dir=buckets_dir,
                reason="preserve_confirmed_promise_as_source_evidence",
                metadata={"promise_status": status, "promise_confirmed": True},
                tags=["source_record", "cyberboss_backfill", "promise", status],
            )
            items.append(item)
            if item.action == "import":
                records.append(record)
            continue
        items.append(
            PlanItem(
                source_type="cyberboss_promise_candidate",
                source_path="memory/promise-candidates.jsonl",
                source_id=source_id,
                event_time=str((row.get("source") or {}).get("receivedAt") or row.get("createdAt") or row.get("updatedAt") or ""),
                content_hash=sha256_text(content or json.dumps(row, ensure_ascii=False, sort_keys=True)),
                target_layer="audit_only",
                target_representation=target,
                already_imported=False,
                action=action,
                reason=reason,
                content_chars=len(content),
            )
        )
    return items, records


def plan_private_experiences(state_dir: Path) -> list[PlanItem]:
    index_path = state_dir / "stone-memory" / "private-experiences" / "index.json"
    if not index_path.exists():
        return []
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    mapping = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), dict) else {}
    items: list[PlanItem] = []
    for source_id, meta in mapping.items():
        if not isinstance(meta, dict):
            continue
        body_file = str(meta.get("bodyFile") or "").strip()
        body_path = state_dir / "stone-memory" / "private-experiences" / "bodies" / body_file
        content = body_path.read_text(encoding="utf-8").strip() if body_path.exists() else ""
        items.append(
            PlanItem(
                source_type=f"stone_private_{str(meta.get('source') or 'experience')}",
                source_path=body_path.relative_to(state_dir).as_posix() if body_path.exists() else index_path.relative_to(state_dir).as_posix(),
                source_id=str(source_id),
                event_time=str(meta.get("timestamp") or meta.get("updatedAt") or ""),
                content_hash=sha256_text(content),
                target_layer="private",
                target_representation="private_source_evidence_pending",
                already_imported=False,
                action="skip",
                reason="privacy_boundary_preserved_no_equivalent_private_recall_import_selected",
                content_chars=len(content),
            )
        )
    return items


def plan_runtime_continuity(state_dir: Path) -> list[PlanItem]:
    items: list[PlanItem] = []
    for relative in RUNTIME_AUDIT_FILES:
        path = state_dir / relative
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8").strip()
        items.append(
            PlanItem(
                source_type="runtime_continuity",
                source_path=relative,
                source_id=relative,
                event_time=iso_from_mtime(path),
                content_hash=sha256_text(content),
                target_layer="runtime_only",
                target_representation="continuity_audit",
                already_imported=False,
                action="skip",
                reason="ephemeral_runtime_continuity_not_long_term_memory",
                content_chars=len(content),
            )
        )
    return items


def build_plan(state_dir: Path, config: dict[str, Any]) -> tuple[list[PlanItem], list[RawCandidate], list[SourceRecordCandidate]]:
    state_dir = state_dir.resolve()
    ombre_state = Path(str(config.get("state_dir") or ROOT / "state")).resolve()
    buckets_dir = Path(str(config.get("buckets_dir") or ROOT / "buckets")).resolve()
    raw_db = ombre_state / "raw_events.sqlite"
    target_raw = load_existing_raw_index(raw_db)
    planned_raw = load_existing_raw_index(raw_db)

    items: list[PlanItem] = []
    raw_candidates: list[RawCandidate] = []
    source_records: list[SourceRecordCandidate] = []

    conversation_items, conversation_candidates, history = plan_conversations(state_dir, target_raw, planned_raw)
    items.extend(conversation_items)
    raw_candidates.extend(conversation_candidates)

    stone_items, stone_candidates = plan_stone_messages(state_dir, history, target_raw, planned_raw)
    items.extend(stone_items)
    raw_candidates.extend(stone_candidates)

    structured_items, structured_records = plan_stone_structured(state_dir, buckets_dir)
    items.extend(structured_items)
    source_records.extend(structured_records)

    file_items, file_records = plan_memory_files(state_dir, buckets_dir)
    items.extend(file_items)
    source_records.extend(file_records)

    items.extend(plan_private_experiences(state_dir))
    items.extend(plan_runtime_continuity(state_dir))
    return items, raw_candidates, source_records


def summarize(items: list[PlanItem]) -> dict[str, Any]:
    return {
        "total_sources": len(items),
        "actions": dict(sorted(Counter(item.action for item in items).items())),
        "source_types": dict(sorted(Counter(item.source_type for item in items).items())),
        "targets": dict(sorted(Counter(item.target_representation for item in items).items())),
        "reasons": dict(sorted(Counter(item.reason for item in items).items())),
        "already_imported": sum(1 for item in items if item.already_imported),
        "to_import": sum(1 for item in items if item.action == "import"),
        "skipped": sum(1 for item in items if item.action == "skip"),
    }


def _frontmatter_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_frontmatter_safe(item) for item in value]
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _bulk_ingest_raw(raw_store: RawEventStore, events: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    now = raw_store._now_iso()
    normalized: list[dict[str, Any]] = []
    rejected = Counter()
    for raw in events:
        event, reason = raw_store._normalize_event(raw, default_source=source, ingested_at=now)
        if reason:
            rejected[str(reason)] += 1
        elif event:
            normalized.append(event)

    inserted = 0
    duplicate = 0
    conn = raw_store._connect()
    try:
        for event in normalized:
            cursor = conn.execute(
                """
                INSERT OR IGNORE INTO raw_events
                (source, source_event_id, event_hash, role, text, created_at, ingested_at,
                 conversation_id, session_id, client, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event["source"],
                    event["source_event_id"],
                    event["event_hash"],
                    event["role"],
                    event["text"],
                    event["created_at"],
                    event["ingested_at"],
                    event["conversation_id"],
                    event["session_id"],
                    event["client"],
                    event["metadata_json"],
                ),
            )
            if cursor.rowcount:
                inserted += 1
                if raw_store.fts_enabled:
                    try:
                        conn.execute(
                            """
                            INSERT INTO raw_events_fts(rowid, text, source, conversation_id, session_id)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (
                                int(cursor.lastrowid or 0),
                                event["text"],
                                event["source"],
                                event["conversation_id"],
                                event["session_id"],
                            ),
                        )
                    except sqlite3.OperationalError:
                        pass
            else:
                duplicate += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {"inserted": inserted, "duplicate": duplicate, "rejected": dict(rejected)}


async def apply_plan(
    *,
    config: dict[str, Any],
    raw_candidates: list[RawCandidate],
    source_records: list[SourceRecordCandidate],
) -> dict[str, Any]:
    raw_store = RawEventStore(config)
    bucket_mgr = BucketManager(config)
    raw_inserted = 0
    raw_duplicates = 0
    raw_skipped = Counter()
    batch: list[dict[str, Any]] = []

    def flush_raw() -> None:
        nonlocal raw_inserted, raw_duplicates, batch
        if not batch:
            return
        result = _bulk_ingest_raw(raw_store, batch, source="cyberboss_backfill")
        raw_inserted += int(result.get("inserted") or 0)
        raw_duplicates += int(result.get("duplicate") or 0)
        raw_skipped.update(result.get("rejected") or {})
        batch = []

    for candidate in raw_candidates:
        batch.append(
            {
                "role": candidate.role,
                "text": candidate.text,
                "source_event_id": candidate.source_event_id,
                "created_at": candidate.created_at,
                "conversation_id": candidate.conversation_id,
                "session_id": "main_relation",
                "client": "cyberboss_backfill",
                "metadata": candidate.metadata,
            }
        )
        if len(batch) >= 500:
            flush_raw()
    flush_raw()

    source_created = 0
    source_exists = 0
    buckets_dir = Path(str(config["buckets_dir"]))
    existing_source_ids = {path.stem for path in buckets_dir.rglob("source_cyberboss_*.md") if path.is_file()} if buckets_dir.exists() else set()
    for record in source_records:
        item = record.item
        if item.bucket_id in existing_source_ids:
            source_exists += 1
            continue
        extra_metadata = {
            "title": f"Cyberboss source {item.source_type} {item.source_id}",
            "source_type": item.source_type,
            "source_path": item.source_path,
            "source_id": item.source_id,
            "source_time": item.event_time,
            "source_sha256": item.content_hash,
            "target_representation": item.target_representation,
            "backfill_version": 1,
            **{key: _frontmatter_safe(value) for key, value in record.metadata.items()},
        }
        await bucket_mgr.create(
            record.content,
            bucket_id=item.bucket_id,
            importance=5,
            tags=[tag for tag in record.tags if tag],
            domain=["cyberboss", "source_evidence"],
            source="cyberboss_backfill",
            bucket_type="source",
            created=record.created_at or None,
            extra_metadata=extra_metadata,
        )
        existing_source_ids.add(item.bucket_id)
        source_created += 1

    return {
        "raw_inserted": raw_inserted,
        "raw_duplicates": raw_duplicates,
        "raw_skipped": dict(raw_skipped),
        "source_records_created": source_created,
        "source_records_existing": source_exists,
    }


def print_samples(items: list[PlanItem], limit: int) -> None:
    if limit <= 0:
        return
    grouped: dict[str, list[PlanItem]] = defaultdict(list)
    for item in items:
        grouped[item.source_type].append(item)
    sample_payload = {
        source_type: [asdict(item) for item in values[:limit]]
        for source_type, values in sorted(grouped.items())
    }
    print(json.dumps({"samples": sample_payload}, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Cyberboss -> Ombre memory backfill planner")
    parser.add_argument(
        "--cyberboss-state",
        default=os.environ.get("CYBERBOSS_STATE_DIR", ""),
        help="Cyberboss production state directory (the .cyberboss directory)",
    )
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="Ombre config.yaml")
    parser.add_argument("--apply", action="store_true", help="Apply import. Omit for a read-only dry-run.")
    parser.add_argument("--jsonl", action="store_true", help="Stream the full migration manifest as JSONL to stdout.")
    parser.add_argument("--sample", type=int, default=2, help="Sample rows per source type in summary mode.")
    args = parser.parse_args()

    if not args.cyberboss_state:
        parser.error("--cyberboss-state or CYBERBOSS_STATE_DIR is required")
    state_dir = Path(args.cyberboss_state).resolve()
    if not state_dir.exists():
        parser.error(f"Cyberboss state directory does not exist: {state_dir}")
    config = load_config(Path(args.config).resolve())

    items, raw_candidates, source_records = build_plan(state_dir, config)
    header = {
        "mode": "apply" if args.apply else "dry-run",
        "cyberboss_state": str(state_dir),
        "ombre_state": str(Path(str(config.get("state_dir") or "")).resolve()),
        "summary": summarize(items),
        "planned_raw_imports": len(raw_candidates),
        "planned_source_record_imports": len(source_records),
    }
    print(json.dumps(header, ensure_ascii=False, indent=None if args.jsonl else 2))
    if args.jsonl:
        for item in items:
            print(json.dumps(asdict(item), ensure_ascii=False, sort_keys=True))
    else:
        print_samples(items, max(0, args.sample))

    if not args.apply:
        return 0
    result = asyncio.run(apply_plan(config=config, raw_candidates=raw_candidates, source_records=source_records))
    print(json.dumps({"apply_result": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
