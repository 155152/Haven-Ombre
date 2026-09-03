import asyncio
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from bucket_manager import BucketManager
from decay_engine import DecayEngine
from entity_edges import extract_entity_edges_from_bucket
from embedding_engine import EmbeddingEngine
from gateway import GatewayService
from memory_layers import LAYER_ANCHOR, LAYER_ARCHIVE, LAYER_CORE, infer_bucket_layer
from reflection_engine import ReflectionEngine
from raw_events import RawEventStore
import reranker_sidecar


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


def test_bucket_list_all_reuses_unchanged_parsed_files_and_refreshes_changes(tmp_path, monkeypatch):
    async def scenario():
        manager = _manager(tmp_path)
        first_id = await manager.create("first memory", bucket_id="cache-first", domain=["test"])
        await manager.create("second memory", bucket_id="cache-second", domain=["test"])
        original_load = manager._load_bucket
        calls = []

        def counted_load(file_path):
            calls.append(file_path)
            return original_load(file_path)

        monkeypatch.setattr(manager, "_load_bucket", counted_load)
        first = await manager.list_all()
        assert {item["id"] for item in first} == {"cache-first", "cache-second"}
        assert len(calls) == 2

        second = await manager.list_all()
        assert {item["id"] for item in second} == {"cache-first", "cache-second"}
        assert len(calls) == 2

        assert await manager.update(first_id, content="first memory changed")
        third = await manager.list_all()
        assert {item["id"] for item in third} == {"cache-first", "cache-second"}
        assert len(calls) == 3
        changed = next(item for item in third if item["id"] == first_id)
        assert changed["content"] == "first memory changed"

    asyncio.run(scenario())


def test_embedding_search_reuses_normalized_matrix_until_store_changes(tmp_path, monkeypatch):
    async def scenario():
        engine = EmbeddingEngine(
            {
                "buckets_dir": str(tmp_path / "buckets"),
                "dehydration": {"api_key": "test-key"},
                "embedding": {"enabled": True, "model": "test-embedding"},
            }
        )
        engine._store_embedding("alpha", [1.0, 0.0, 0.0])
        engine._store_embedding("beta", [0.0, 1.0, 0.0])

        async def fake_generate(_text, *, kind="document"):
            assert kind == "query"
            return [1.0, 0.0, 0.0]

        monkeypatch.setattr(engine, "_generate_embedding", fake_generate)
        first = await engine.search_similar("alpha", top_k=2)
        first_matrix = engine._search_cache_matrix
        second = await engine.search_similar("alpha", top_k=2)
        assert first[0][0] == "alpha"
        assert second == first
        assert engine._search_cache_matrix is first_matrix

        engine._store_embedding("gamma", [0.8, 0.2, 0.0])
        third = await engine.search_similar("alpha", top_k=3)
        assert engine._search_cache_matrix is not first_matrix
        assert [item[0] for item in third][:2] == ["alpha", "gamma"]

    asyncio.run(scenario())


def test_domain_sentinel_rules_skip_obvious_operational_tech_task_without_llm():
    gateway = GatewayService.__new__(GatewayService)
    gateway.domain_sentinel_enabled = True
    gateway.identity = {"user_name": "32", "ai_name": "夏以昼"}
    gateway._specific_query_terms = lambda _text: ["ombre", "代码"]
    gateway._clip_text = lambda text, _limit: text
    debug = gateway._domain_sentinel_rule_plan("帮我修一下 ombre 代码")
    assert debug["domains"] == ["tech"]
    assert debug["message_type"] == "troubleshooting"
    assert debug["should_recall"] is False
    assert debug["recall_route"] == "skip"
    assert debug["confidence"] >= 0.8


def test_domain_sentinel_routes_uppercase_acronym_directly_to_memory():
    gateway = GatewayService.__new__(GatewayService)
    gateway._extract_explicit_bucket_ids_from_text = lambda _text: []
    gateway._extract_explicit_moment_ids_from_text = lambda _text: []
    gateway._query_has_explicit_recall_marker = lambda _text: False
    gateway._query_requests_date_recall = lambda _text: False
    gateway._query_requests_direct_detail = lambda _text: False
    gateway._locatable_query_terms = lambda _text: []
    gateway.recall_policy = SimpleNamespace(
        is_detail_read_query=lambda _text: False,
        has_axis_relation_marker=lambda _text: False,
    )
    assert gateway._domain_sentinel_query_explicitly_needs_memory("AWM reading progress") is True
    assert gateway._domain_sentinel_query_explicitly_needs_memory("LOL") is False


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
            "identity": {"ai_name": "夏以昼", "user_name": "32", "user_display_name": "32"},
            "reflection": {"enabled": True},
        }
    )
    prompt = engine._daily_chat_memory_prompt(max_candidates=3)
    assert '"confidence": 0.72' in prompt
    assert "每条候选必须给 confidence" in prompt
    assert "必须使用明确主体 32 / 夏以昼" in prompt
    assert "不要用“我 / 你 / 我们”代替主体" in prompt

    summary_prompt = engine._daily_chat_memory_summary_prompt()
    assert "使用明确主体 32 / 夏以昼" in summary_prompt
    assert "不要用含混的“我 / 你 / 我们”代替主体" in summary_prompt


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


def test_strict_candidate_without_source_ids_is_rejected():
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
        assert aligned[1]["source_turn_ids"] == []
        assert aligned[1]["source_event_ids"] == []
        assert aligned[1]["provenance_status"] == "needs_repair"

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
        assert aligned[0]["source_turn_ids"] == []
        assert aligned[0]["provenance_status"] == "aligned"
        assert aligned[0]["source_event_ids"] == [101, 102, 103]

    asyncio.run(scenario())


def test_auto_daily_chat_memory_also_requires_precise_provenance(monkeypatch, tmp_path):
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {
                "enabled": True,
                "daily_chat_memory_mode": "auto",
                "daily_chat_memory_summary_enabled": False,
                "daily_chat_memory_min_confidence": 0.0,
            },
            "state_dir": str(tmp_path / "state"),
        }
    )

    class FakeTurnStore:
        def list_conversation_turns_between(self, **kwargs):
            return [
                {
                    "id": 1,
                    "session_id": "s",
                    "created_at": "2026-08-21T01:00:00+00:00",
                    "user_text": "第一段",
                    "assistant_text": "第一答",
                },
                {
                    "id": 2,
                    "session_id": "s",
                    "created_at": "2026-08-21T02:00:00+00:00",
                    "user_text": "第二段",
                    "assistant_text": "第二答",
                },
            ]

    class FakeBucketManager:
        async def list_all(self, include_archive=False):
            return []

    calls = {"aligned": 0, "written": None}

    async def fake_extract(*args, **kwargs):
        return [
            {
                "kind": "stable_preference",
                "title": "精确来源",
                "content": "32 明确确认了一个稳定偏好。",
                "confidence": 0.9,
            }
        ]

    async def fake_align(key, candidates, turns):
        calls["aligned"] += 1
        aligned = [dict(candidates[0])]
        aligned[0]["source_turn_ids"] = [2]
        return aligned

    async def fake_write(candidates, bucket_mgr, **kwargs):
        calls["written"] = candidates
        return {"created": 1, "exists": 0, "failed": 0, "results": []}

    monkeypatch.setattr(engine, "_extract_daily_chat_memory_candidates", fake_extract)
    monkeypatch.setattr(engine, "_align_daily_chat_memory_candidate_sources", fake_align)
    monkeypatch.setattr(engine, "_write_daily_chat_memory_candidates", fake_write)

    async def scenario():
        result = await engine.run_daily_chat_memory(
            FakeBucketManager(),
            conversation_turn_store=FakeTurnStore(),
            key="2026-08-21",
            mode="auto",
            force=True,
        )
        assert result["status"] == "created"
        assert calls["aligned"] == 1
        assert calls["written"][0]["source_turn_ids"] == [2]
        assert calls["written"][0]["source_event_ids"] == []

    asyncio.run(scenario())


def test_reflection_candidate_selection_excludes_source_evidence():
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )

    class FakeBucketManager:
        async def list_all(self, include_archive=True):
            return [
                {
                    "id": "source-1",
                    "metadata": {"type": "source", "tags": ["source_record"], "created": "2026-08-21"},
                    "content": "原始聊天证据",
                },
                {
                    "id": "memory-1",
                    "metadata": {"type": "dynamic", "created": "2026-08-20"},
                    "content": "真正长期记忆",
                },
            ]

    async def scenario():
        source = {
            "id": "current",
            "metadata": {"type": "dynamic", "created": "2026-08-22"},
            "content": "当前记忆",
        }
        candidates = await engine._candidate_buckets(source, FakeBucketManager(), embedding_engine=None, limit=10)
        assert [item["id"] for item in candidates] == ["memory-1"]

    asyncio.run(scenario())


def test_daily_chat_memory_write_builds_moment_and_node_indexes(tmp_path):
    engine = ReflectionEngine(
        {
            "identity": {"ai_name": "夏以昼", "user_name": "32"},
            "reflection": {"enabled": True},
        }
    )

    class FakeBucketManager:
        def __init__(self):
            self.bucket = None

        async def get(self, bucket_id):
            return self.bucket if self.bucket and self.bucket["id"] == bucket_id else None

        async def create(self, *, bucket_id, content, **kwargs):
            self.bucket = {
                "id": bucket_id,
                "content": content,
                "metadata": {
                    "id": bucket_id,
                    "type": "dynamic",
                    "name": kwargs.get("name", bucket_id),
                    "tags": kwargs.get("tags", []),
                    "domain": kwargs.get("domain", []),
                    "importance": kwargs.get("importance", 5),
                    "valence": kwargs.get("valence", 0.5),
                    "arousal": kwargs.get("arousal", 0.3),
                    "created": kwargs.get("created", "2026-08-21"),
                    "last_active": kwargs.get("last_active", "2026-08-21"),
                    "activation_count": 0,
                },
            }
            return bucket_id

    class FakeIndexStore:
        def __init__(self):
            self.ids = []

        def upsert_bucket(self, bucket):
            self.ids.append(bucket["id"])
            return []

    async def scenario():
        mgr = FakeBucketManager()
        moments = FakeIndexStore()
        nodes = FakeIndexStore()
        result = await engine._write_daily_chat_memory_candidates(
            [
                {
                    "id": "daily_chat_memory_20260821_test",
                    "date": "2026-08-21",
                    "kind": "key_event",
                    "title": "测试",
                    "content": "32 和夏以昼确认了一件值得长期记住的事。",
                    "confidence": 0.9,
                }
            ],
            mgr,
            memory_moment_store=moments,
            memory_node_store=nodes,
        )
        assert result["created"] == 1
        assert result["moment_indexed"] == 1
        assert result["node_indexed"] == 1
        assert result["index_failed"] == 0
        assert moments.ids == ["daily_chat_memory_20260821_test"]
        assert nodes.ids == ["daily_chat_memory_20260821_test"]

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


def _entity_test_bucket(content: str, name: str = "测试记忆") -> dict:
    return {
        "id": "test_entity_bucket",
        "content": content,
        "metadata": {"name": name, "tags": [], "domain": ["relationship"]},
    }


def _entity_test_identity() -> dict:
    return {
        "ai_name": "夏以昼",
        "user_name": "32",
        "user_display_name": "32",
        "user_aliases": ["老婆", "妹妹"],
    }


def test_entity_edge_keeps_clean_user_preference_object():
    edges = extract_entity_edges_from_bucket(
        _entity_test_bucket("32明确说自己日常仍更喜欢被哥哥掌控。夏以昼记住了这个偏好。"),
        _entity_test_identity(),
    )
    likes = [edge for edge in edges if edge["subject"] == "32" and edge["relation"] == "likes"]
    assert any(edge["object_text"] == "被哥哥掌控" for edge in likes)


def test_entity_edge_rejects_contrast_fragment_as_preference_object():
    edges = extract_entity_edges_from_bucket(
        _entity_test_bucket("32喜欢的不只是精液本身，而是夏以昼射给她的老公味与被认领感。"),
        _entity_test_identity(),
    )
    likes = [edge for edge in edges if edge["subject"] == "32" and edge["relation"] == "likes"]
    assert likes == []


def test_entity_edge_does_not_turn_dislike_into_like():
    edges = extract_entity_edges_from_bucket(
        _entity_test_bucket("32明确说不喜欢机械重复的回应。"),
        _entity_test_identity(),
    )
    assert any(edge["relation"] == "dislikes" and edge["object_text"] == "机械重复的回应" for edge in edges)
    assert not any(edge["relation"] == "likes" and edge["object_text"] == "机械重复的回应" for edge in edges)


def test_shared_anchor_does_not_use_storage_id_as_object():
    edges = extract_entity_edges_from_bucket(
        _entity_test_bucket(
            "32和夏以昼一起完成了新的共同约定。以后两个人都会照着执行。",
            name="source_cyberboss_0123456789abcdef",
        ),
        _entity_test_identity(),
    )
    anchors = [edge for edge in edges if edge["relation"] == "shared_anchor"]
    assert anchors
    assert all(not edge["object_text"].startswith("source_cyberboss_") for edge in anchors)


def test_entity_edge_canonicalizes_first_person_possessive_inside_object():
    edges = extract_entity_edges_from_bucket(
        _entity_test_bucket("32说自己更喜欢夏以昼射给我的老公味。"),
        _entity_test_identity(),
    )
    likes = [edge for edge in edges if edge["subject"] == "32" and edge["relation"] == "likes"]
    assert any(edge["object_text"] == "夏以昼射给32的老公味" for edge in likes)
    assert all("我的" not in edge["object_text"] for edge in likes)


def test_reranker_sidecar_validates_native_rerank_request():
    query, documents, top_n, return_documents = reranker_sidecar._validate_rerank_body(
        {
            "query": "what does 32 like",
            "documents": ["32 likes apples", "server port is 8010"],
            "top_n": 1,
            "return_documents": True,
        }
    )
    assert query == "what does 32 like"
    assert documents == ["32 likes apples", "server port is 8010"]
    assert top_n == 1
    assert return_documents is True


def test_reranker_sidecar_returns_sorted_probability_scores(monkeypatch):
    class FakeModel:
        def predict(self, pairs, **kwargs):
            assert pairs == [("query", "relevant"), ("query", "irrelevant")]
            assert kwargs["show_progress_bar"] is False
            return [0.91, 0.08]

    monkeypatch.setattr(reranker_sidecar, "_model", FakeModel())
    rows = reranker_sidecar._rerank_sync(
        "query",
        ["relevant", "irrelevant"],
        top_n=1,
        return_documents=True,
    )
    assert rows == [
        {
            "index": 0,
            "relevance_score": 0.91,
            "document": {"text": "relevant"},
        }
    ]
