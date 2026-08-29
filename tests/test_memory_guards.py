import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from bucket_manager import BucketManager
from decay_engine import DecayEngine
from memory_layers import LAYER_ANCHOR, LAYER_ARCHIVE, LAYER_CORE, infer_bucket_layer
from reflection_engine import ReflectionEngine
from raw_events import RawEventStore


def _bucket_meta(**metadata):
    return {"metadata": metadata}


def test_permanent_digested_stays_core():
    assert infer_bucket_layer(_bucket_meta(type="permanent", digested=True)) == LAYER_CORE


def test_pinned_digested_stays_core():
    assert infer_bucket_layer(_bucket_meta(pinned=True, digested=True)) == LAYER_CORE


def test_anchor_digested_stays_anchor():
    assert infer_bucket_layer(_bucket_meta(anchor=True, digested=True)) == LAYER_ANCHOR


def test_pinned_archived_is_archive():
    assert infer_bucket_layer(_bucket_meta(pinned=True, archived=True)) == LAYER_ARCHIVE


def test_anchor_resolved_is_archive():
    assert infer_bucket_layer(_bucket_meta(anchor=True, resolved=True)) == LAYER_ARCHIVE


def test_source_record_is_exempt_from_decay():
    class FakeBucketManager:
        def __init__(self):
            self.archived = []

        async def list_all(self, include_archive=False):
            assert include_archive is False
            return [
                {
                    "id": "source-evidence",
                    "metadata": {
                        "type": "source",
                        "tags": ["source_record", "cyberboss_backfill"],
                        "importance": 1,
                        "created": "2020-01-01T00:00:00+00:00",
                    },
                }
            ]

        async def archive(self, bucket_id):
            self.archived.append(bucket_id)
            return True

        async def update(self, bucket_id, **kwargs):
            raise AssertionError("source evidence must not be auto-resolved")

    async def scenario():
        manager = FakeBucketManager()
        engine = DecayEngine({"decay": {"threshold": 999}}, manager)
        result = await engine.run_decay_cycle()
        assert result["checked"] == 0
        assert result["archived"] == 0
        assert manager.archived == []
        assert engine.calculate_score({"type": "source", "tags": ["source_record"]}) == 999.0

    asyncio.run(scenario())


def _manager(tmp_path):
    return BucketManager(
        {
            "buckets_dir": str(tmp_path / "buckets"),
            "state_dir": str(tmp_path / "state"),
        }
    )


def test_source_record_activate_preserves_source_type(tmp_path):
    async def scenario():
        manager = _manager(tmp_path)
        bucket_id = await manager.create(
            "source evidence",
            bucket_id="source-activate-test",
            tags=["source_record", "cyberboss_backfill", "stone_feature", "relation"],
            domain=["cyberboss", "source_evidence"],
            bucket_type="source",
            source="cyberboss_backfill",
        )
        assert await manager.archive(bucket_id)
        assert await manager.activate(bucket_id)
        current = await manager.get(bucket_id)
        assert current["metadata"]["type"] == "source"
        assert "active" not in current["metadata"]
        assert "deprecated" not in current["metadata"]
        assert "resolved" not in current["metadata"]

    asyncio.run(scenario())


def test_source_record_skips_enrichment():
    async def scenario():
        engine = ReflectionEngine(
            {
                "identity": {"ai_name": "夏以昼", "user_name": "32"},
                "reflection": {"enabled": True},
            }
        )

        class Manager:
            async def get(self, bucket_id):
                return {
                    "id": bucket_id,
                    "content": "verbatim evidence",
                    "metadata": {
                        "type": "source",
                        "tags": ["source_record", "cyberboss_backfill"],
                    },
                }

            async def update(self, *args, **kwargs):
                raise AssertionError("source record must not be enriched")

        class EdgeStore:
            def add_edges(self, *args, **kwargs):
                raise AssertionError("source record must not create memory edges")

        result = await engine.enrich_bucket(
            "source-evidence",
            Manager(),
            EdgeStore(),
            force=True,
        )
        assert result["status"] == "skipped_source_record"

    asyncio.run(scenario())


def test_source_record_skips_edge_backfill():
    async def scenario():
        engine = ReflectionEngine(
            {
                "identity": {"ai_name": "夏以昼", "user_name": "32"},
                "reflection": {"enabled": True},
            }
        )

        class Manager:
            async def get(self, bucket_id):
                return {
                    "id": bucket_id,
                    "content": "verbatim evidence",
                    "metadata": {
                        "type": "source",
                        "tags": ["source_record", "cyberboss_backfill"],
                    },
                }

        class EdgeStore:
            def add_edges(self, *args, **kwargs):
                raise AssertionError("source record must not create memory edges")

        result = await engine.backfill_edges_for_bucket(
            "source-evidence",
            Manager(),
            EdgeStore(),
            dry_run=False,
        )
        assert result["status"] == "skipped"
        assert result["reason"] == "not_edge_backfillable"
        assert result["edges"] == 0

    asyncio.run(scenario())


def test_daily_chat_prompt_requires_confidence():
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )
    prompt = engine._daily_chat_memory_prompt(max_candidates=3)
    assert '"confidence": 0.72' in prompt
    assert "每条候选必须给 confidence" in prompt


def test_review_normalization_can_retain_low_confidence_candidate():
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )
    candidates = engine._normalize_daily_chat_memory_candidates(
        "2026-08-21",
        [
            {
                "kind": "stable_preference",
                "title": "明确偏好",
                "content": "32明确表示以后更喜欢这种安排，并希望后续继续保持。",
                "confidence": 0.2,
                "source_event_ids": [101],
                "source_turn_ids": [1],
            }
        ],
        [{"id": 1, "raw_event_ids": [101]}],
        max_candidates=3,
        min_confidence=0.0,
    )
    assert len(candidates) == 1
    assert candidates[0]["confidence"] == 0.2
    assert candidates[0]["source_event_ids"] == [101]


def test_daily_chat_provenance_filters_hallucinated_ids_and_derives_events():
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )
    candidates = engine._normalize_daily_chat_memory_candidates(
        "2026-08-21",
        [
            {
                "kind": "commitment",
                "title": "明确约定",
                "content": "32和夏以昼明确约定以后继续保持这种相处方式。",
                "confidence": 0.8,
                "source_turn_ids": [2, 999],
                "source_event_ids": [9999],
            }
        ],
        [
            {"id": 1, "raw_event_ids": [101]},
            {"id": 2, "raw_event_ids": [102, 103]},
        ],
        max_candidates=3,
        min_confidence=0.0,
        require_source_provenance=True,
    )
    assert len(candidates) == 1
    assert candidates[0]["source_turn_ids"] == [2]
    assert candidates[0]["source_event_ids"] == [102, 103]


def test_review_candidate_without_source_turn_ids_is_rejected():
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )
    candidates = engine._normalize_daily_chat_memory_candidates(
        "2026-08-21",
        [
            {
                "kind": "relationship_anchor",
                "title": "缺少来源",
                "content": "这是一条看起来合理但没有精确来源 turn 的候选。",
                "confidence": 0.9,
            }
        ],
        [{"id": 1, "raw_event_ids": [101]}],
        max_candidates=3,
        min_confidence=0.0,
        require_source_provenance=True,
    )
    assert candidates == []


def test_review_provenance_aligner_keeps_only_real_turn_ids(monkeypatch):
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )

    monkeypatch.setattr(
        engine,
        "_daily_chat_memory_model_client",
        lambda candidate=False: (object(), "test-model", False),
    )

    async def fake_completion(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "alignments": [
                                    {"candidate_index": 0, "source_turn_refs": [2, 999]},
                                    {"candidate_index": 1, "source_turn_refs": [999]},
                                ]
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(engine, "_daily_chat_memory_create_completion", fake_completion)

    async def scenario():
        candidates = [
            {"kind": "stable_preference", "title": "A", "content": "内容A"},
            {"kind": "commitment", "title": "B", "content": "内容B"},
        ]
        turns = [
            {"id": 1, "user_text": "u1", "assistant_text": "a1", "raw_event_ids": [101]},
            {"id": 2, "user_text": "u2", "assistant_text": "a2", "raw_event_ids": [102]},
        ]
        aligned = await engine._align_daily_chat_memory_candidate_sources(
            "2026-08-21",
            candidates,
            turns,
        )
        assert aligned[0]["source_turn_ids"] == [2]
        assert aligned[0]["source_event_ids"] == [102]
        assert "source_turn_ids" not in aligned[1]
        assert "source_event_ids" not in aligned[1]

    asyncio.run(scenario())


def test_review_provenance_aligner_maps_refs_to_raw_event_ids(monkeypatch):
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )
    monkeypatch.setattr(
        engine,
        "_daily_chat_memory_model_client",
        lambda candidate=False: (object(), "test-model", False),
    )

    async def fake_completion(*args, **kwargs):
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps(
                            {
                                "alignments": [
                                    {"candidate_index": 0, "source_turn_refs": [1, 2]}
                                ]
                            }
                        )
                    )
                )
            ]
        )

    monkeypatch.setattr(engine, "_daily_chat_memory_create_completion", fake_completion)

    async def scenario():
        aligned = await engine._align_daily_chat_memory_candidate_sources(
            "2026-08-21",
            [{"kind": "relationship_anchor", "title": "A", "content": "内容A"}],
            [
                {"id": None, "user_text": "u1", "assistant_text": "a1", "raw_event_ids": [101]},
                {"id": None, "user_text": "u2", "assistant_text": "a2", "raw_event_ids": [102, 103]},
            ],
        )
        assert "source_turn_ids" not in aligned[0]
        assert aligned[0]["source_event_ids"] == [101, 102, 103]

    asyncio.run(scenario())


def test_review_accepts_precise_event_only_provenance():
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )
    candidates = engine._normalize_daily_chat_memory_candidates(
        "2026-08-21",
        [
            {
                "kind": "relationship_anchor",
                "title": "有事件来源",
                "content": "这条候选只引用真实 raw event provenance。",
                "confidence": 0.9,
                "source_event_ids": [102, 9999],
            }
        ],
        [
            {"id": None, "raw_event_ids": [101]},
            {"id": None, "raw_event_ids": [102, 103]},
        ],
        max_candidates=3,
        min_confidence=0.0,
        require_source_provenance=True,
    )
    assert len(candidates) == 1
    assert candidates[0]["source_turn_ids"] == []
    assert candidates[0]["source_event_ids"] == [102]


def test_daily_chat_fallback_uses_dedicated_timeout_client():
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {
                "enabled": True,
                "daily_chat_memory_timeout_seconds": 180,
            },
            "dehydration": {
                "api_key": "test-key",
                "base_url": "https://example.invalid/v1",
                "model": "test-model",
            },
        }
    )
    client, model, dedicated = engine._daily_chat_memory_model_client(candidate=True)
    assert client is engine.daily_chat_memory_dehydration_client
    assert client is not engine.dehydration_client
    assert model == "test-model"
    assert dedicated is False


def test_raw_event_history_offset_reads_distinct_chunk(tmp_path):
    store = RawEventStore(
        {
            "buckets_dir": str(tmp_path / "buckets"),
            "state_dir": str(tmp_path / "state"),
        }
    )
    base = datetime(2026, 8, 21, 0, 0, tzinfo=timezone.utc)
    events = [
        {
            "source_event_id": f"event-{index}",
            "role": "user" if index % 2 == 0 else "assistant",
            "text": f"message-{index}",
            "created_at": base.replace(hour=index).isoformat(),
        }
        for index in range(6)
    ]
    result = store.ingest(events, source="test")
    assert result["inserted"] == 6

    chunk = store.list_events_between(
        start_at=base,
        end_at=base.replace(hour=23),
        limit=2,
        offset=2,
    )
    assert [item["source_event_id"] for item in chunk] == ["event-3", "event-2"]


def test_content_replace_preserves_previous_body(tmp_path):
    async def scenario():
        manager = _manager(tmp_path)
        bucket_id = await manager.create(
            "old body",
            bucket_id="revision-test",
            domain=["tests"],
        )

        assert await manager.update(
            bucket_id,
            content="new body",
            revision_reason="test_replace",
        )

        revisions = manager.revision_store.list_for_bucket(bucket_id)
        assert len(revisions) == 1
        revision = revisions[0]
        assert revision["previous_content"] == "old body"
        assert revision["previous_sha256"] == hashlib.sha256(b"old body").hexdigest()
        assert revision["replacement_sha256"] == hashlib.sha256(b"new body").hexdigest()
        assert revision["reason"] == "test_replace"

        current = await manager.get(bucket_id)
        assert current["content"] == "new body"

    asyncio.run(scenario())


def test_same_content_replace_does_not_create_revision(tmp_path):
    async def scenario():
        manager = _manager(tmp_path)
        bucket_id = await manager.create(
            "same body",
            bucket_id="same-content-test",
            domain=["tests"],
        )

        assert await manager.update(bucket_id, content="same body")
        assert manager.revision_store.list_for_bucket(bucket_id) == []

    asyncio.run(scenario())


def test_revision_failure_blocks_markdown_overwrite(tmp_path, monkeypatch):
    async def scenario():
        manager = _manager(tmp_path)
        bucket_id = await manager.create(
            "protected old body",
            bucket_id="fail-closed-test",
            domain=["tests"],
        )

        def fail_revision(*args, **kwargs):
            raise sqlite3.OperationalError("revision store unavailable")

        monkeypatch.setattr(manager.revision_store, "preserve_before_replace", fail_revision)

        with pytest.raises(sqlite3.OperationalError, match="revision store unavailable"):
            await manager.update(bucket_id, content="must not be written")

        current = await manager.get(bucket_id)
        assert current["content"] == "protected old body"

    asyncio.run(scenario())
