import asyncio
import json
import sqlite3
from pathlib import Path

from scripts.backfill_cyberboss_memory import apply_plan, build_plan


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _create_stone_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE messages (
            message_seq INTEGER PRIMARY KEY,
            thread_id TEXT,
            message_id TEXT,
            timestamp TEXT,
            source_date TEXT,
            role TEXT,
            text TEXT,
            source TEXT
        );
        CREATE TABLE feelings (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            source_date TEXT,
            event_time TEXT,
            order_key TEXT,
            content TEXT,
            summary_mode TEXT,
            importance INTEGER,
            source TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE features (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            source_date TEXT,
            category TEXT,
            content TEXT,
            importance INTEGER,
            source TEXT,
            created_at TEXT,
            updated_at TEXT
        );
        CREATE TABLE notebook_entries (
            id TEXT PRIMARY KEY,
            thread_id TEXT,
            topic_id TEXT,
            title TEXT,
            relative_path TEXT,
            visibility TEXT,
            tags_json TEXT,
            body_text TEXT,
            revision INTEGER,
            created_at TEXT,
            updated_at TEXT
        );
        """
    )
    conn.executemany(
        "INSERT INTO messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (1, "main", "visible-copy", "2026-08-01T10:00:00Z", "2026-08-01", "user", "你好", "cyberboss_runtime_history"),
            (2, "main", "runtime-gap", "2026-08-01T10:01:00Z", "2026-08-01", "assistant", "这是可见历史补漏", "cyberboss_runtime_history"),
            (3, "main", "dream-json", "2026-08-01T10:02:00Z", "2026-08-01", "assistant", "[experience source=dream reality=unreal-dream] 梦境", "json"),
            (4, "main", "internal", "2026-08-01T10:03:00Z", "2026-08-01", "user", "AUTONOMOUS CHECK-IN WAKE — not a message from 32.", "codex"),
        ],
    )
    conn.execute(
        "INSERT INTO feelings VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        ("feel-1", "main", "2026-08-01", "2026-08-01T10:04:00Z", "1", "记得那天很开心", "full", 7, "miner", "2026-08-01T10:05:00Z", "2026-08-01T10:05:00Z"),
    )
    conn.commit()
    conn.close()


def _config(tmp_path: Path) -> dict:
    return {
        "buckets_dir": str(tmp_path / "ombre" / "buckets"),
        "state_dir": str(tmp_path / "ombre" / "state"),
        "matching": {},
        "scoring_weights": {},
        "wikilink": {},
    }


def test_backfill_plan_filters_runtime_noise_and_is_idempotent(tmp_path):
    state = tmp_path / "cyberboss" / ".cyberboss"
    _write_jsonl(
        state / "conversations" / "2026-08-01.jsonl",
        [
            {
                "id": "conv-user",
                "type": "user",
                "timestamp": "2026-08-01T10:00:00Z",
                "threadId": "main",
                "text": "你好",
                "meta": {"provider": "weixin", "runtimeId": "codex"},
            },
            {
                "id": "conv-wake",
                "type": "user",
                "timestamp": "2026-08-01T10:00:30Z",
                "threadId": "main",
                "text": "AUTONOMOUS CHECK-IN WAKE — not a message from 32. Continue as Xia Yizhou.",
                "meta": {"provider": "weixin", "runtimeId": "codex"},
            },
            {
                "id": "conv-assistant",
                "type": "assistant",
                "timestamp": "2026-08-01T10:00:40Z",
                "threadId": "main",
                "text": "你好呀",
                "meta": {"runtimeId": "codex"},
            },
        ],
    )
    (state / "memory").mkdir(parents=True, exist_ok=True)
    (state / "memory" / "preferences.md").write_text("喜欢安静地一起读书。\n", encoding="utf-8")
    _create_stone_db(state / "stone-memory" / "home" / ".stone_memory" / "stone-memory.db")

    config = _config(tmp_path)
    items, raw_candidates, source_records = build_plan(state, config)
    reasons = {item.source_id: item.reason for item in items}

    assert reasons["conv-wake"] == "canonical_conversation_internal_runtime_context"
    assert reasons["stone-message:1"] == "covered_by_visible_conversation"
    assert reasons["stone-message:2"] == "stone_verified_runtime_history_fallback"
    assert reasons["stone-message:3"] == "stone_source_not_approved_for_raw_fallback"
    assert reasons["stone-message:4"] == "stone_internal_runtime_context"
    assert {candidate.text for candidate in raw_candidates} == {"你好", "你好呀", "这是可见历史补漏"}
    assert any(record.item.source_type == "stone_feeling" for record in source_records)
    assert any(record.item.source_path == "memory/preferences.md" for record in source_records)

    first = asyncio.run(apply_plan(config=config, raw_candidates=raw_candidates, source_records=source_records))
    assert first["raw_inserted"] == 3
    assert first["raw_duplicates"] == 0
    assert first["raw_skipped"] == {}
    assert first["source_records_created"] == 2

    raw_db = Path(config["state_dir"]) / "raw_events.sqlite"
    conn = sqlite3.connect(raw_db)
    try:
        rows = conn.execute("SELECT role, text FROM raw_events ORDER BY id").fetchall()
    finally:
        conn.close()
    assert rows == [("user", "你好"), ("assistant", "你好呀"), ("assistant", "这是可见历史补漏")]

    second_items, second_raw, second_records = build_plan(state, config)
    assert second_raw == []
    assert second_records == []
    assert sum(item.already_imported for item in second_items if item.target_representation == "raw_event") >= 3

    second = asyncio.run(apply_plan(config=config, raw_candidates=second_raw, source_records=second_records))
    assert second == {
        "raw_inserted": 0,
        "raw_duplicates": 0,
        "raw_skipped": {},
        "source_records_created": 0,
        "source_records_existing": 0,
    }
