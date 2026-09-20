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
from memory_moments import MemoryMomentStore
from reflection_engine import ReflectionEngine
from raw_events import RawEventStore
from recall_policy import RecallPolicy, build_query_anchor_plan, direct_candidate_satisfies_anchor_plan
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


def test_domain_sentinel_ordinary_chat_uses_implicit_evidence_mode():
    gateway = GatewayService.__new__(GatewayService)
    gateway._domain_sentinel_query_explicitly_needs_memory = lambda _text: False
    debug = {
        "message_type": "ordinary_chat",
        "should_recall": False,
        "recall_route": "implicit",
        "confidence": 0.9,
    }
    assert gateway._domain_sentinel_recall_mode(debug, "住阿姨家也还是很累") == "implicit"
    assert gateway._domain_sentinel_should_skip_recall(debug, "住阿姨家也还是很累") is False


def test_domain_sentinel_troubleshooting_remains_hard_skip():
    gateway = GatewayService.__new__(GatewayService)
    gateway._domain_sentinel_query_explicitly_needs_memory = lambda _text: False
    debug = {
        "message_type": "troubleshooting",
        "should_recall": False,
        "recall_route": "skip",
        "confidence": 0.9,
    }
    assert gateway._domain_sentinel_recall_mode(debug, "帮我修一下代码") == "hard_skip"
    assert gateway._domain_sentinel_should_skip_recall(debug, "帮我修一下代码") is True


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


def test_emotional_anchor_matches_semantic_state_variant_with_same_event_anchor():
    query = "住阿姨家的时候家务活之类的啥也不用想 但住她家的时候也累累的"
    plan = build_query_anchor_plan(query)
    assert plan.has_direct_constraints is True
    node = {
        "content": "32在阿姨家长期无法独处和持续儿童噪声会明显消耗她。",
        "metadata": {"name": "32在阿姨家最想要的是回到安静的自己家"},
    }
    assert direct_candidate_satisfies_anchor_plan(node, plan) is True


def test_recall_query_plan_excludes_negated_explanation_from_required_axis():
    policy = RecallPolicy()
    query = "住阿姨家的时候家务活之类的啥也不用想 但住她家的时候也累累的"
    plan = policy.plan_query(query)
    assert "家务活" in plan.locatable_terms
    assert "家务活" in plan.excluded_axis_terms
    assert "家务活" not in plan.activated_axis_terms
    assert plan.axis_polarity_debug["contrast_detected"] is True


def test_axis_polarity_does_not_treat_is_it_question_as_negation():
    policy = RecallPolicy()
    retained, excluded, debug = policy._axis_polarity_terms(
        "工作是不是让我累的原因",
        ("工作",),
    )
    assert retained == ("工作",)
    assert excluded == ()
    assert debug["excluded_terms"] == []


def test_axis_polarity_excludes_directly_rejected_cause_but_keeps_positive_axis():
    policy = RecallPolicy()
    retained, excluded, _debug = policy._axis_polarity_terms(
        "工作倒不是问题，但是通勤让我很累",
        ("工作", "通勤"),
    )
    assert retained == ("通勤",)
    assert excluded == ("工作",)


def test_axis_polarity_keeps_term_when_it_is_negated_then_reasserted():
    policy = RecallPolicy()
    retained, excluded, _debug = policy._axis_polarity_terms(
        "工作不是主因，但工作安排还是让我累",
        ("工作",),
    )
    assert retained == ("工作",)
    assert excluded == ()


def test_retrieval_alias_anchor_direct_signal_requires_rare_alias_and_strict_anchor_match():
    gateway = GatewayService.__new__(GatewayService)
    gateway.recall_policy = RecallPolicy()
    gateway.config = {}
    gateway.word_map_store = None
    query = "住阿姨家的时候家务活之类的啥也不用想 但住她家的时候也累累的"
    item = {
        "bucket": {
            "id": "memory-aunt-home",
            "content": "32在阿姨家长期无法独处和持续儿童噪声会明显消耗她。",
            "metadata": {"name": "32在阿姨家最想要的是回到安静的自己家"},
        },
        "retrieval_alias_match": True,
        "retrieval_alias_terms": ["阿姨"],
        "retrieval_alias_term_bucket_count": 14,
    }
    assert gateway._retrieval_alias_anchor_direct_signal(query, item) is True
    item["retrieval_alias_term_bucket_count"] = 118
    assert gateway._retrieval_alias_anchor_direct_signal(query, item) is False
    item["retrieval_alias_term_bucket_count"] = 14
    assert gateway._retrieval_alias_anchor_direct_signal("今天阿姨来了吗", item) is False


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


def _associative_test_config(tmp_path):
    return {
        "buckets_dir": str(tmp_path / "buckets"),
        "state_dir": str(tmp_path / "state"),
        "reflection": {
            "enabled": True,
            "enrich_on_write": True,
            "model": "test-model",
            "api_key": "test-key",
            "base_url": "http://127.0.0.1:9/v1",
            "associative_triggers_enabled": True,
            "associative_trigger_min_confidence": 0.62,
        },
    }


def _associative_test_bucket(content="32提过以后如果连续加班几天，就会想把周末空出来彻底休息。"):
    return {
        "id": "memory-associative-test",
        "content": content,
        "metadata": {
            "name": "连续加班后的休息偏好",
            "domain": ["life"],
            "tags": ["stable_preference"],
            "importance": 6,
        },
    }


def test_retrieval_alias_search_segments_natural_chinese_query(tmp_path):
    store = MemoryMomentStore(_associative_test_config(tmp_path))
    bucket = {
        "id": "memory-aunt-home",
        "content": "## Fact\n\n32在阿姨家长期无法独处和持续儿童噪声会明显消耗她。",
        "metadata": {
            "name": "32在阿姨家最想要的是回到安静的自己家",
            "domain": ["relationship"],
            "tags": ["environment_preference"],
            "importance": 5,
        },
    }
    store.upsert_bucket(bucket)
    for index in range(30):
        store.upsert_bucket({
            "id": f"generic-no-need-{index}",
            "content": f"## Fact\n\n这是一条只包含不用处理某件事的泛化记忆 {index}。",
            "metadata": {
                "name": f"不用处理某件事情的泛化记忆 {index}",
                "domain": ["life"],
                "tags": ["generic"],
                "importance": 2,
            },
        })
    hits = store.search_retrieval_aliases(
        "住阿姨家的时候家务活之类的啥也不用想 但住她家的时候也累累的",
        limit=20,
    )
    target = next((row for row in hits if row["bucket_id"] == bucket["id"]), None)
    assert target is not None
    assert "阿姨" in target["matched_terms"]
    assert target["matched_term_bucket_count"] == 1


def test_associative_trigger_store_search_and_source_hash_invalidation(tmp_path):
    store = MemoryMomentStore(_associative_test_config(tmp_path))
    bucket = _associative_test_bucket()
    store.upsert_bucket(bucket)
    source_hash = store.associative_source_hash(bucket)
    assert store.replace_associative_triggers(
        bucket["id"],
        source_hash,
        [
            {
                "trigger_type": "bridge",
                "trigger_text": "临时决定周末还要不要继续排工作",
                "confidence": 0.82,
                "embedding": [1.0, 0.0, 0.0],
                "embedding_model": "test",
            },
            {
                "trigger_type": "horizon",
                "channel": "avoidance_habit",
                "trigger_text": "连续透支之后开始主动保护完整休息日",
                "confidence": 0.75,
                "embedding": [0.0, 1.0, 0.0],
                "embedding_model": "test",
            },
        ],
    ) == 2
    hits = store.search_associative_triggers(
        [0.99, 0.05, 0.0],
        eligible_bucket_ids={bucket["id"]},
        embedding_model="test",
        top_k=4,
        min_cosine=0.8,
    )
    assert len(hits) == 1
    assert hits[0]["trigger_type"] == "bridge"
    assert store.search_associative_triggers(
        [0.99, 0.05, 0.0],
        eligible_bucket_ids={bucket["id"]},
        embedding_model="different-model",
        min_cosine=0.8,
    ) == []
    store.upsert_bucket(bucket)
    assert len(store.list_associative_triggers(bucket["id"])) == 2
    changed = _associative_test_bucket("32后来明确表示连续加班后周末也愿意安排轻量工作，不再要求整天空出来。")
    store.upsert_bucket(changed)
    assert store.list_associative_triggers(bucket["id"]) == []


def test_associative_trigger_search_ignores_stale_index_version(tmp_path):
    store = MemoryMomentStore(_associative_test_config(tmp_path))
    bucket = _associative_test_bucket()
    store.upsert_bucket(bucket)
    store.replace_associative_triggers(
        bucket["id"],
        "legacy-source-hash",
        [{
            "trigger_type": "bridge",
            "trigger_text": "旧版联想索引不应继续参与召回",
            "confidence": 0.9,
            "embedding": [1.0, 0.0, 0.0],
            "embedding_model": "test",
        }],
    )
    assert len(store.list_associative_triggers(bucket["id"])) == 1
    assert store.search_associative_triggers(
        [1.0, 0.0, 0.0],
        eligible_bucket_ids={bucket["id"]},
        embedding_model="test",
        min_cosine=0.8,
    ) == []


def test_reflection_builds_associative_triggers_once_per_source_hash(tmp_path):
    store = MemoryMomentStore(_associative_test_config(tmp_path))
    bucket = _associative_test_bucket()
    store.upsert_bucket(bucket)
    engine = ReflectionEngine(_associative_test_config(tmp_path))
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        payload = {
            "bridge_triggers": [
                {"text": "临时决定周末还要不要继续排工作", "confidence": 0.84},
            ],
            "horizon_triggers": [
                {
                    "channel": "avoidance_habit",
                    "text": "连续透支之后开始主动保护完整休息日",
                    "confidence": 0.76,
                }
            ],
        }
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload, ensure_ascii=False)))]
        )

    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    class FakeEmbedding:
        enabled = True
        model = "fake-embedding"

        async def embed_text(self, text, *, kind="document"):
            return [1.0, float(len(text) % 7 + 1), 0.5]

    first = asyncio.run(engine._refresh_associative_triggers(
        bucket,
        memory_moment_store=store,
        embedding_engine=FakeEmbedding(),
    ))
    assert first == {"status": "ok", "written": 2}
    assert len(calls) == 1
    second = asyncio.run(engine._refresh_associative_triggers(
        bucket,
        memory_moment_store=store,
        embedding_engine=FakeEmbedding(),
    ))
    assert second == {"status": "current", "written": 2}
    assert len(calls) == 1


def test_reflection_caches_empty_associative_trigger_result(tmp_path):
    store = MemoryMomentStore(_associative_test_config(tmp_path))
    bucket = _associative_test_bucket()
    store.upsert_bucket(bucket)
    engine = ReflectionEngine(_associative_test_config(tmp_path))
    calls = []

    async def fake_create(**kwargs):
        calls.append(kwargs)
        payload = {"bridge_triggers": [], "horizon_triggers": []}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
        )

    engine.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    class FakeEmbedding:
        enabled = True
        model = "fake-embedding"

        async def embed_text(self, text, *, kind="document"):
            raise AssertionError("empty trigger result should not request embeddings")

    first = asyncio.run(engine._refresh_associative_triggers(
        bucket,
        memory_moment_store=store,
        embedding_engine=FakeEmbedding(),
    ))
    assert first == {"status": "empty", "written": 0}
    state = store.get_associative_trigger_state(bucket["id"])
    assert state["status"] == "empty"
    assert state["embedding_model"] == "fake-embedding"
    assert len(calls) == 1

    second = asyncio.run(engine._refresh_associative_triggers(
        bucket,
        memory_moment_store=store,
        embedding_engine=FakeEmbedding(),
    ))
    assert second == {"status": "current_empty", "written": 0}
    assert len(calls) == 1


def _configure_associative_gateway(gateway, store, embedding_engine):
    gateway.associative_trigger_enabled = True
    gateway.associative_trigger_top_k = 8
    gateway.associative_trigger_strong_gate = 0.72
    gateway.associative_trigger_soft_gate = 0.48
    gateway.associative_trigger_soft_effective_gate = 0.50
    gateway.associative_trigger_margin = 0.08
    gateway.associative_trigger_soft_confidence = 0.72
    gateway.embedding_query_timeout_seconds = 1.0
    gateway.embedding_engine = embedding_engine
    gateway.memory_moment_store = store


def test_recall_policy_natural_memory_request_allows_archive_targets_without_formulaic_date_wording():
    policy = RecallPolicy()
    plan = policy.plan_query("宝贝 我去亲戚公司办公室那阵子你记得吗")
    assert plan.explicit_old_memory is True
    assert plan.allow_archive_targets is True

    future_instruction = policy.plan_query("你记得明天提醒我处理肉")
    assert future_instruction.explicit_old_memory is False
    assert future_instruction.allow_archive_targets is False


def test_recall_policy_entity_anchor_terms_prefer_people_and_named_entities_over_generic_scene_nouns():
    policy = RecallPolicy()
    plan = policy.plan_query("宝贝 好难受啊 我上个月不是去阿姨那了吗 就办公室去了半个月你记得吗")
    assert "阿姨" in plan.locatable_terms
    assert "办公室" in plan.locatable_terms
    assert plan.entity_anchor_terms == ("阿姨",)


def test_gateway_auto_route_plan_owns_recall_intent_and_search_depth():
    gateway = GatewayService.__new__(GatewayService)
    gateway.recall_policy = RecallPolicy()
    gateway._query_requests_direct_detail = lambda _query: False
    gateway._query_requests_date_recall = lambda _query: False

    continuation = gateway._hook_recall_route_plan("等下晚点要回去做饭 然后来送饭")
    assert continuation["route"] == "implicit"
    assert continuation["search_depth"] == "shallow"

    affection = gateway._hook_recall_route_plan("我可以亲亲你吗")
    assert affection["route"] == "skip"
    assert affection["search_depth"] == "none"

    remembered = gateway._hook_recall_route_plan(
        "宝贝 好难受啊 我上个月不是去阿姨那了吗 就办公室去了半个月你记得吗"
    )
    assert remembered["route"] == "explicit"
    assert remembered["search_depth"] == "deep"

    trend = gateway._hook_recall_route_plan("按我之前的体重趋势，我什么时候能到55kg？")
    assert trend["route"] == "explicit"
    assert trend["search_depth"] == "deep"


def test_focused_direct_candidates_use_common_admission_pipeline_without_heavy_search():
    gateway = GatewayService.__new__(GatewayService)
    gateway.recall_policy = SimpleNamespace(
        plan_query=lambda _query, context_mode="": SimpleNamespace(
            allow_archive_targets=True,
            locatable_terms=("阿姨", "办公室"),
            entity_anchor_terms=("阿姨",),
        )
    )
    canonical = {
        "id": "historical_aunt_work",
        "metadata": {"name": "阿姨公司工作压力", "type": "archived"},
        "content": "住阿姨家期间去阿姨公司的办公室工作，寄住和工作压力叠在一起。",
    }
    source = {
        "id": "source_work_notes",
        "metadata": {"name": "legacy-work-notes", "type": "source", "source_record": True},
        "content": "阿姨公司的旧记录。",
    }
    irrelevant = {
        "id": "historical_office_roleplay",
        "metadata": {"name": "办公室角色扮演", "type": "archived"},
        "content": "只提到了办公室，没有阿姨。",
    }
    gateway._list_gateway_buckets = lambda **_kwargs: asyncio.sleep(
        0, result=[irrelevant, source, canonical]
    )
    gateway._is_source_record_bucket = lambda bucket: bool(
        (bucket.get("metadata") or {}).get("source_record")
    )
    gateway._normalized_recall_query = lambda query: query
    captured = {}

    async def fake_select(query, session_id, buckets, **kwargs):
        captured["query"] = query
        captured["session_id"] = session_id
        captured["bucket_ids"] = [bucket["id"] for bucket in buckets]
        captured["kwargs"] = kwargs
        return [canonical], [], {"final_bucket_ids": [canonical["id"]]}

    gateway._select_dynamic_buckets = fake_select
    gateway._hook_recall_card_from_bucket = lambda bucket, **_kwargs: {
        "id": f"ombre:{bucket['id']}",
        "bucket_id": bucket["id"],
        "title": "阿姨公司工作压力",
        "text": bucket["content"],
        "reading_note": {"reliability": "direct_match"},
    }
    gateway._format_selected_bucket_debug = lambda bucket, **_kwargs: {"bucket_id": bucket["id"]}
    gateway._format_suppressed_bucket_debug = lambda item, **_kwargs: item

    cards, recalled_ids, debug = asyncio.run(
        gateway._hook_recall_focused_cards(
            "你还记得上个月阿姨公司办公室那件事吗",
            "acceptance-session",
            max_cards=1,
            max_chars=600,
        )
    )

    assert captured["bucket_ids"] == ["historical_office_roleplay", "historical_aunt_work"]
    assert captured["kwargs"]["allow_semantic"] is False
    assert captured["kwargs"]["allow_query_planner"] is False
    assert captured["kwargs"]["allow_associative"] is False
    assert captured["kwargs"]["allow_archived_candidates"] is True
    assert captured["kwargs"]["allow_semantic_session_dedupe"] is True
    assert captured["kwargs"]["allow_rerank"] is False
    assert captured["kwargs"]["allow_expansion"] is False
    assert recalled_ids == ["historical_aunt_work"]
    assert len(cards) == 1
    assert debug["hook_recall_debug"]["source_record_fallback"] is False


def test_explicit_entity_anchor_blocks_unrelated_deep_candidate():
    gateway = GatewayService.__new__(GatewayService)
    gateway.recall_policy = RecallPolicy()
    gateway.config = {}
    gateway.word_map_store = None
    gateway._is_source_record_bucket = lambda _bucket: False

    query = "你还记得我以前提过的XQZ987蓝色彗星杯吗"
    unrelated = {
        "id": "unrelated-bedtime",
        "metadata": {"name": "睡前日常"},
        "content": "以前提过一些睡前日常，但没有那件物品。",
        "_recall_signal": {
            "semantic_score": 0.99,
            "rerank_score": 0.99,
            "matched_query_terms": ["以前", "提过"],
        },
    }
    anchored = {
        "id": "anchored",
        "metadata": {"name": "XQZ987 蓝色彗星杯"},
        "content": "XQZ987 蓝色彗星杯是当时提过的物品。",
        "_recall_signal": {
            "matched_query_terms": ["XQZ987"],
        },
    }

    assert gateway._explicit_entity_anchor_keys(query) == ["xqz987"]
    assert gateway._bucket_matches_explicit_entity_anchor(
        query, unrelated, signal=unrelated["_recall_signal"]
    ) is False
    assert gateway._hook_bucket_has_strong_topic_evidence(
        query, unrelated, allow_signal=False
    ) is False
    assert gateway._bucket_matches_explicit_entity_anchor(
        query, anchored, signal=anchored["_recall_signal"]
    ) is True
    assert gateway._hook_bucket_has_strong_topic_evidence(
        query, anchored, allow_signal=False
    ) is True


def test_focused_full_unified_pipeline_can_return_empty_without_heavy_fallback():
    gateway = GatewayService.__new__(GatewayService)
    gateway.recall_policy = SimpleNamespace(
        plan_query=lambda _query, context_mode="": SimpleNamespace(allow_archive_targets=True)
    )

    async def focused_empty(*_args, **_kwargs):
        return [], [], {"hook_recall_debug": {"candidate_count": 1, "direct_scan_hit": False}}

    async def unified_empty(*_args, **_kwargs):
        return [], [], {"hook_recall_debug": {"candidate_count": 3}}

    async def fail_prepare(*_args, **_kwargs):
        raise AssertionError("deep unified recall must be allowed to return zero cards")

    gateway._hook_recall_focused_cards = focused_empty
    gateway._hook_recall_fast_cards = unified_empty
    gateway.prepare_payload = fail_prepare
    gateway._hook_recall_full_dynamic_context = lambda *_args, **_kwargs: ""
    gateway._clip_text = lambda value, _limit: value
    gateway._render_hook_recall_full_additional_context = lambda value: value

    response = asyncio.run(gateway._handle_hook_recall_full(
        query="你还记得上次说过的那件事吗？",
        session_id="main_relation",
        messages=[{"role": "user", "content": "你还记得上次说过的那件事吗？"}],
        model="test-model",
        max_cards=2,
        max_chars=1200,
        max_context_chars=3600,
        include_diffused=False,
        include_context_debug=False,
        include_debug=True,
        focused_full=True,
    ))
    body = response.body.decode("utf-8")
    assert response.status_code == 200
    assert '"cards":[]' in body
    assert '"mode":"full_unified"' in body
    assert '"focused_full_hit":false' in body
    assert '"deep_fallback_used":true' in body


def test_recent_repeat_bypass_requires_explicit_user_recall_intent():
    gateway = GatewayService.__new__(GatewayService)
    gateway.recall_policy = RecallPolicy()
    gateway._query_requests_direct_detail = lambda _query: False
    gateway._query_requests_date_recall = lambda _query: False
    gateway._extract_explicit_bucket_ids_from_text = lambda _query: set()
    gateway._bucket_repeat_bypass_has_direct_evidence = lambda _query, _item: True
    item = {"bucket": {"id": "daily-pref"}}

    assert gateway._session_hard_exclude_bucket_bypass(
        "老公给我摸摸嘛～～", item
    ) is False
    assert gateway._session_semantic_dedupe_bypass(
        "老公给我摸摸嘛～～", item
    ) is False

    assert gateway._session_hard_exclude_bucket_bypass(
        "你还记得我之前说的那个偏好吗？", item
    ) is True
    assert gateway._session_semantic_dedupe_bypass(
        "你还记得我之前说的那个偏好吗？", item
    ) is True


def test_recent_repeat_set_includes_previously_strong_candidates():
    gateway = GatewayService.__new__(GatewayService)
    gateway.skip_recent_rounds = 5
    gateway.semantic_session_dedupe_enabled = True
    gateway.state_store = SimpleNamespace(
        list_injection_debug=lambda **_kwargs: [{
            "payload": {
                "recalled_moment_debug": [{
                    "bucket_id": "daily-pref",
                    "admission_reason": "strong_semantic",
                    "semantic_score": 0.91,
                    "injected": True,
                }],
            },
        }]
    )

    assert gateway._session_hard_exclude_bucket_ids("main_relation") == {"daily-pref"}
    assert gateway._session_semantic_dedupe_source_bucket_ids("main_relation") == ["daily-pref"]


def test_source_record_hook_card_is_always_query_fragment_not_whole_container():
    gateway = GatewayService.__new__(GatewayService)
    gateway.self_anchor_entry_bucket_id = ""
    gateway.identity = {
        "ai_name": "夏以昼",
        "user_name": "32",
        "user_display_name": "32",
        "user_aliases": ["老婆", "妹妹"],
        "relationship_terms": [],
    }
    gateway.recall_policy = SimpleNamespace(specific_query_terms=lambda _query: ["55kg"])
    source_bucket = {
        "id": "source_weight_notes",
        "metadata": {
            "name": "memory/auto-captured-notes.md",
            "type": "source",
            "tags": ["source_record", "cyberboss_backfill"],
        },
        "content": (
            "FILE_HEAD_FARM_THREAD_NOISE "
            + "农场提醒和线程切换。" * 40
            + "希望明天起来我能下55kg，最近也在看体重趋势。"
            + "checkin 调试和处理肉提醒。" * 40
            + " FILE_TAIL_CHECKIN_MEAT_NOISE"
        ),
        "score": 88,
    }

    card = gateway._hook_recall_card_from_bucket(
        source_bucket,
        query="我什么时候能下55kg啊",
        max_chars=1200,
    )

    assert card is not None
    assert card["source_kind"] == "source_record"
    assert card["render_shape"] == "source_fragment"
    assert "55kg" in card["text"]
    assert len(card["text"]) < len(source_bucket["content"]) / 2
    assert "FILE_HEAD_FARM_THREAD_NOISE" not in card["text"]
    assert "FILE_TAIL_CHECKIN_MEAT_NOISE" not in card["text"]
    assert card["reading_note"]["flags"] == ["source_record_fragment_only"]


def test_focused_full_rank_prefers_canonical_memory_over_source_container_when_both_match():
    gateway = GatewayService.__new__(GatewayService)
    gateway.recall_policy = SimpleNamespace(specific_query_terms=lambda _query: ["55kg"])
    gateway._matched_query_term_is_specific = lambda term: term == "55kg"
    canonical = {
        "id": "historical_weight_goal",
        "metadata": {"name": "减重目标55 52 50", "type": "dynamic"},
        "content": "目标先到55kg，再到52和50，并结合真实体重趋势判断速度。",
    }
    source = {
        "id": "source_weight_notes",
        "metadata": {"name": "auto-captured-notes", "type": "source", "tags": ["source_record"]},
        "content": "希望明天起来我能下55kg。",
    }

    assert gateway._hook_focused_bucket_rank(
        "我什么时候能下55kg啊", canonical
    ) > gateway._hook_focused_bucket_rank(
        "我什么时候能下55kg啊", source
    )


def test_fast_bucket_ranking_prefers_canonical_over_source_when_evidence_tier_matches():
    gateway = GatewayService.__new__(GatewayService)
    gateway.recall_fusion_mode = "dynamic"
    gateway._bucket_recall_rank = lambda _q, _bucket, _score=0.0: (0, 0.0)
    canonical = {
        "bucket": {
            "id": "historical_work",
            "metadata": {"type": "dynamic", "name": "亲戚公司工作压力"},
        },
        "exact_anchor_match": True,
        "score": 0.61,
    }
    source = {
        "bucket": {
            "id": "source_work",
            "metadata": {"type": "source", "tags": ["source_record"], "name": "legacy source"},
        },
        "exact_anchor_match": True,
        "score": 0.99,
    }

    assert gateway._bucket_final_candidate_rank("亲戚公司", canonical) < gateway._bucket_final_candidate_rank("亲戚公司", source)


def test_breath_source_evidence_is_merged_after_canonical_direct_moments():
    from server import _merge_source_record_synthetic_moments

    canonical = {"bucket_id": "historical_clean", "moment_id": "m-clean", "text": "canonical"}
    source = {"bucket_id": "source_legacy", "moment_id": "m-source", "text": "fragment"}
    merged = _merge_source_record_synthetic_moments([canonical], [source])
    assert [item["bucket_id"] for item in merged] == ["historical_clean", "source_legacy"]


def test_breath_secondary_direct_label_is_not_reported_as_associative_diffusion():
    from server import _format_secondary_direct_moment

    block = _format_secondary_direct_moment({
        "bucket_id": "historical_work",
        "moment_id": "m-work",
        "metadata": {"bucket_name": "亲戚公司工作压力"},
        "section": "fact",
        "text": "不想做直播，也不想去亲戚公司工作。",
    })
    assert "次级直接命中" in block
    assert "不是联想扩散" in block


def test_gateway_startup_warms_associative_query_embedding_when_triggers_exist():
    calls = []

    class FakeStore:
        def has_associative_triggers(self):
            return True

    class FakeEmbedding:
        enabled = True

        async def embed_text(self, text, *, kind="query"):
            calls.append((text, kind))
            return [1.0, 0.0, 0.0]

    gateway = GatewayService.__new__(GatewayService)
    gateway.associative_trigger_enabled = True
    gateway.associative_trigger_startup_warm_timeout_seconds = 1.0
    gateway.embedding_engine = FakeEmbedding()
    gateway.memory_moment_store = FakeStore()

    assert asyncio.run(gateway._warm_associative_query_embedding()) is True
    assert calls == [("记忆联想检索预热", "query")]


def test_gateway_associative_trigger_channel_accepts_strong_hit_without_returning_trigger_text(tmp_path):
    store = MemoryMomentStore(_associative_test_config(tmp_path))
    bucket = _associative_test_bucket()
    store.upsert_bucket(bucket)
    store.replace_associative_triggers(
        bucket["id"],
        store.associative_source_hash(bucket),
        [{
            "trigger_type": "bridge",
            "trigger_text": "周末临时又想继续塞工作",
            "confidence": 0.8,
            "embedding": [1.0, 0.0, 0.0],
            "embedding_model": "test",
        }],
    )

    class FakeEmbedding:
        enabled = True
        model = "test"

        async def embed_text(self, text, *, kind="query"):
            assert kind == "query"
            return [1.0, 0.01, 0.0]

    gateway = GatewayService.__new__(GatewayService)
    _configure_associative_gateway(gateway, store, FakeEmbedding())
    hits = asyncio.run(gateway._get_associative_trigger_candidates("周末要不继续干活", {bucket["id"]}))
    assert set(hits) == {bucket["id"]}
    assert hits[bucket["id"]]["cosine"] >= 0.99
    assert hits[bucket["id"]]["acceptance"] == "strong"
    assert "trigger_text" not in hits[bucket["id"]]
    gateway._is_source_record_bucket = lambda _bucket: False
    assert gateway._hook_bucket_has_strong_topic_evidence(
        "完全不同的词面",
        {
            **bucket,
            "_recall_signal": {
                "associative_trigger_match": True,
                "associative_trigger_cosine": 0.91,
                "associative_trigger_confidence": 0.8,
                "associative_trigger_margin": 0.01,
            },
        },
        allow_signal=True,
    ) is True


def test_gateway_associative_trigger_soft_hit_requires_competitor_margin():
    class FakeEmbedding:
        enabled = True
        model = "test"

        async def embed_text(self, text, *, kind="query"):
            raise AssertionError("precomputed query embedding must be reused")

    class FakeStore:
        def __init__(self, competitor_score):
            self.competitor_score = competitor_score

        def has_associative_triggers(self):
            return True

        def search_associative_triggers(self, *_args, **_kwargs):
            return [
                {
                    "bucket_id": "target",
                    "score": 0.66,
                    "confidence": 0.82,
                    "trigger_type": "bridge",
                    "channel": "",
                    "trigger_text": "must never escape into gateway candidate payload",
                },
                {
                    "bucket_id": "competitor",
                    "score": self.competitor_score,
                    "confidence": 0.80,
                    "trigger_type": "bridge",
                    "channel": "",
                    "trigger_text": "competitor",
                },
            ]

    gateway = GatewayService.__new__(GatewayService)
    _configure_associative_gateway(gateway, FakeStore(0.40), FakeEmbedding())
    hits = asyncio.run(gateway._get_associative_trigger_candidates(
        "亲戚家借住",
        {"target", "competitor"},
        query_embedding=[1.0, 0.0, 0.0],
    ))
    assert set(hits) == {"target"}
    assert hits["target"]["acceptance"] == "soft_margin"
    assert hits["target"]["margin"] == pytest.approx(0.26)
    assert "trigger_text" not in hits["target"]

    gateway.memory_moment_store = FakeStore(0.62)
    ambiguous = asyncio.run(gateway._get_associative_trigger_candidates(
        "亲戚家借住",
        {"target", "competitor"},
        query_embedding=[1.0, 0.0, 0.0],
    ))
    assert ambiguous == {}


def test_gateway_associative_trigger_soft_signal_uses_confidence_adjusted_effective_gate():
    gateway = GatewayService.__new__(GatewayService)
    gateway.associative_trigger_strong_gate = 0.72
    gateway.associative_trigger_soft_gate = 0.48
    gateway.associative_trigger_soft_effective_gate = 0.50
    gateway.associative_trigger_margin = 0.08
    gateway.associative_trigger_soft_confidence = 0.72

    assert gateway._associative_trigger_direct_signal({
        "cosine": 0.5273,
        "confidence": 0.82,
        "margin": 0.20,
    }) is True
    assert gateway._associative_trigger_direct_signal({
        "cosine": 0.5169,
        "confidence": 0.66,
        "margin": 0.20,
    }) is False
    assert gateway._associative_trigger_direct_signal({
        "cosine": 0.5148,
        "confidence": 0.82,
        "margin": 0.01,
    }) is False
    assert gateway._associative_trigger_direct_signal({
        "cosine": 0.4564,
        "confidence": 0.82,
        "margin": 0.30,
    }) is False


def test_gateway_associative_trigger_counts_as_reliable_picker_signal():
    gateway = GatewayService.__new__(GatewayService)
    gateway.associative_trigger_strong_gate = 0.72
    gateway.associative_trigger_soft_gate = 0.48
    gateway.associative_trigger_soft_effective_gate = 0.50
    gateway.associative_trigger_margin = 0.08
    gateway.associative_trigger_soft_confidence = 0.72
    gateway._planner_lexical_direct_signal = lambda _item: False
    gateway._word_map_direct_signal = lambda _item: False
    item = {
        "associative_trigger_match": True,
        "associative_trigger_cosine": 0.66,
        "associative_trigger_confidence": 0.82,
        "associative_trigger_margin": 0.20,
    }
    assert gateway._dynamic_bucket_item_has_reliable_recall_signal("借住几天", item) is True


def test_gateway_associative_hard_evidence_bypasses_category_overview_direct_match_requirement():
    gateway = GatewayService.__new__(GatewayService)
    gateway.associative_trigger_strong_gate = 0.72
    gateway.associative_trigger_soft_gate = 0.48
    gateway.associative_trigger_soft_effective_gate = 0.50
    gateway.associative_trigger_margin = 0.08
    gateway.associative_trigger_soft_confidence = 0.72
    gateway._is_self_anchor_recall_excluded_bucket = lambda _bucket: False
    gateway._bucket_evidence_labels = lambda _query, _item: ["associative_trigger"]
    gateway._planner_lexical_direct_signal = lambda _item: False
    gateway._entity_edge_direct_signal = lambda _item: False
    gateway._is_identity_name_candidate_bucket = lambda _query, _bucket: False
    gateway._query_anchor_plan = lambda _query: {}
    gateway._anchor_plan_direct_rejection = lambda _bucket, _plan: None
    gateway._axis_lite_bucket_rejection = lambda _query, _item, _plan: None
    gateway._recall_query_plan = lambda _query: SimpleNamespace(explicit_old_memory=False)
    gateway._bucket_relevance_node = lambda bucket: bucket
    gateway._bucket_has_query_topic_evidence = lambda _query, _bucket: False
    gateway._word_map_direct_signal = lambda _item: False
    gateway._bucket_is_tech_domain = lambda _bucket: False
    gateway.recall_policy = SimpleNamespace(
        assess=lambda *_args, **_kwargs: SimpleNamespace(
            admit_direct=True,
            reason="non_explicit_query",
            debug={},
        )
    )
    item = {
        "bucket": {"id": "target", "metadata": {"name": "private space"}},
        "dynamic_anchor_plan": {
            "category_overview": True,
            "category_terms": ["住宿"],
            "required_terms": [],
        },
        "category_overview_item": False,
        "associative_trigger_match": True,
        "associative_trigger_cosine": 0.66,
        "associative_trigger_confidence": 0.82,
        "associative_trigger_margin": 0.20,
    }
    assert gateway._admit_bucket_for_recall("亲戚家借住", item) is True
    assert item["admission_reason"] == "non_explicit_query"


def test_gateway_associative_trigger_direct_signal_bypasses_axis_lite_guard():
    gateway = GatewayService.__new__(GatewayService)
    gateway.associative_trigger_strong_gate = 0.72
    gateway.associative_trigger_soft_gate = 0.48
    gateway.associative_trigger_soft_effective_gate = 0.50
    gateway.associative_trigger_margin = 0.08
    gateway.associative_trigger_soft_confidence = 0.72
    gateway._query_requests_direct_detail = lambda _query: False
    gateway._planner_lexical_direct_signal = lambda _item: False
    gateway._word_map_direct_signal = lambda _item: False
    gateway._entity_edge_direct_signal = lambda _item: False
    gateway.recall_policy = SimpleNamespace(
        is_detail_read_query=lambda _query: False,
        has_strong_score=lambda **_kwargs: False,
    )

    assert gateway._axis_lite_bypass_for_item(
        "如果我要去亲戚家借住几天，住宿上最该先考虑什么？",
        {
            "associative_trigger_match": True,
            "associative_trigger_cosine": 0.66,
            "associative_trigger_confidence": 0.82,
            "associative_trigger_margin": 0.26,
        },
    ) is True

    assert gateway._axis_lite_bypass_for_item(
        "为什么你又像客服模式了？",
        {
            "associative_trigger_match": True,
            "associative_trigger_cosine": 0.66,
            "associative_trigger_confidence": 0.82,
            "associative_trigger_margin": 0.04,
        },
    ) is False
