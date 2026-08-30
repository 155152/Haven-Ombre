import asyncio
import json
import sqlite3
from pathlib import Path

import frontmatter
import pytest

from scripts.backfill_historical_llm_memory import (
    chunk_events,
    import_manual_outputs,
    load_historical_provider,
    normalize_llm_output,
    prepare_stage,
)
from scripts.review_historical_llm_memory import (
    build_review_batches,
    collect_staged_summaries,
    import_manual_review_outputs,
    normalize_manual_review_batch,
    prepare_cross_day_review,
    review_paths,
    score_related_pairs,
)
from scripts.review_historical_bridge import (
    bridge_paths,
    import_bridge_manual_outputs,
    prepare_bridge_review,
)
from scripts.review_historical_promotion import (
    import_manual_promotion,
    normalize_promotion_batch,
    prepare_promotion_review,
    promotion_paths,
)
from scripts.extend_historical_backfill import (
    build_extension,
    build_incremental_bridge,
    build_incremental_bridge_groups,
    build_incremental_promotion_batches,
    build_new_day_review,
)
from scripts.apply_historical_materialization import apply_materialization
from scripts.materialize_historical_memory import prepare_materialization_staging


def _make_raw_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE raw_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            source_event_id TEXT NOT NULL DEFAULT '',
            event_hash TEXT NOT NULL,
            role TEXT NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL,
            ingested_at TEXT NOT NULL,
            conversation_id TEXT NOT NULL DEFAULT '',
            session_id TEXT NOT NULL DEFAULT '',
            client TEXT NOT NULL DEFAULT '',
            metadata_json TEXT NOT NULL DEFAULT '{}'
        );
        """
    )
    rows = [
        ("raw", "1", "h1", "user", "前一天", "2026-06-07T15:59:59Z", "2026-06-07T16:00:00Z", "main", "main_relation", "test", "{}"),
        ("raw", "2", "h2", "user", "我不希望自动关闭客户端", "2026-06-07T16:00:00Z", "2026-06-07T16:00:01Z", "main", "main_relation", "test", "{}"),
        ("raw", "3", "h3", "assistant", "记住这个边界", "2026-06-07T16:01:00Z", "2026-06-07T16:01:01Z", "main", "main_relation", "test", "{}"),
        ("raw", "4", "h4", "user", "Core Memory:\n[bucket_id: x] injected", "2026-06-07T16:02:00Z", "2026-06-07T16:02:01Z", "main", "main_relation", "test", "{}"),
        ("raw", "5", "h5", "assistant", "第二天边界", "2026-06-08T16:00:00Z", "2026-06-08T16:00:01Z", "main", "main_relation", "test", "{}"),
    ]
    conn.executemany("INSERT INTO raw_events VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_provider_has_no_reflection_fallback(monkeypatch):
    monkeypatch.setenv("OMBRE_REFLECTION_API_KEY", "should-not-be-used")
    monkeypatch.setenv("OMBRE_REFLECTION_MODEL", "xia-model")
    monkeypatch.delenv("OMBRE_HISTORICAL_BACKFILL_API_KEY", raising=False)
    monkeypatch.delenv("OMBRE_HISTORICAL_BACKFILL_MODEL", raising=False)
    provider = load_historical_provider({"reflection": {"api_key": "also-no", "model": "also-no"}})
    assert provider.configured is False
    assert provider.api_key == ""
    assert provider.model == ""


def test_chunk_events_is_deterministic_and_overlapping():
    events = [{"event_id": idx} for idx in range(1, 11)]
    first = chunk_events(events, window_turns=4, stride_turns=3)
    second = chunk_events(events, window_turns=4, stride_turns=3)
    assert [item["chunk_id"] for item in first] == [item["chunk_id"] for item in second]
    assert [item["event_ids"] for item in first] == [
        [1, 2, 3, 4],
        [4, 5, 6, 7],
        [7, 8, 9, 10],
    ]


def test_normalize_llm_output_fails_closed_on_fake_provenance():
    parsed = {
        "summaries": [
            {
                "title": "边界",
                "summary": "用户确认不希望自动关闭客户端。",
                "signals": ["boundary"],
                "source_event_ids": [2, 999],
                "confidence": 0.9,
            },
            {
                "title": "无证据",
                "summary": "这条没有合法证据。",
                "source_event_ids": [999],
                "confidence": 0.9,
            },
            {
                "title": "低置信",
                "summary": "低置信内容。",
                "source_event_ids": [2],
                "confidence": 0.2,
            },
        ]
    }
    result = normalize_llm_output(parsed, {2, 3})
    assert result == {
        "summaries": [
            {
                "title": "边界",
                "summary": "用户确认不希望自动关闭客户端。",
                "signals": ["boundary"],
                "source_event_ids": [2],
                "confidence": 0.9,
            }
        ]
    }


def test_prepare_stage_uses_local_day_and_does_not_write_memory(tmp_path, monkeypatch):
    monkeypatch.delenv("OMBRE_HISTORICAL_BACKFILL_API_KEY", raising=False)
    monkeypatch.delenv("OMBRE_HISTORICAL_BACKFILL_MODEL", raising=False)
    state_dir = tmp_path / "state"
    raw_db = state_dir / "raw_events.sqlite"
    _make_raw_db(raw_db)
    config = {
        "state_dir": str(state_dir),
        "buckets_dir": str(tmp_path / "buckets"),
    }
    stage_root = tmp_path / "stage"
    manifest = prepare_stage(
        config=config,
        date_key="2026-06-08",
        timezone_name="Asia/Shanghai",
        window_turns=80,
        stride_turns=64,
        stage_root=stage_root,
    )
    assert manifest["stats"]["eligible"] == 2
    assert manifest["stats"]["injected"] == 1
    assert manifest["chunk_count"] == 1
    assert manifest["provider"]["configured"] is False
    chunk = json.loads(Path(manifest["chunks"][0]["chunk_file"]).read_text(encoding="utf-8"))
    assert [turn["event_id"] for turn in chunk["conversation_turns"]] == [2, 3]
    assert not Path(config["buckets_dir"]).exists()


def test_import_manual_outputs_validates_provenance_and_stages(tmp_path, monkeypatch):
    monkeypatch.delenv("OMBRE_HISTORICAL_BACKFILL_API_KEY", raising=False)
    monkeypatch.delenv("OMBRE_HISTORICAL_BACKFILL_MODEL", raising=False)
    state_dir = tmp_path / "state"
    _make_raw_db(state_dir / "raw_events.sqlite")
    config = {"state_dir": str(state_dir), "buckets_dir": str(tmp_path / "buckets")}
    manifest = prepare_stage(
        config=config,
        date_key="2026-06-08",
        timezone_name="Asia/Shanghai",
        window_turns=80,
        stride_turns=64,
        stage_root=tmp_path / "stage",
    )
    chunk = manifest["chunks"][0]
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / f"{chunk['chunk_id']}.json").write_text(
        json.dumps(
            {
                "chunk_id": chunk["chunk_id"],
                "model": "chatgpt-session",
                "summaries": [
                    {
                        "title": "边界",
                        "summary": "用户确认不希望自动关闭客户端。",
                        "signals": ["boundary"],
                        "source_event_ids": [2, 999],
                        "confidence": 0.91,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = import_manual_outputs(manifest, manual_dir)
    assert result == {
        "imported_chunks": 1,
        "missing_manual_chunks": 0,
        "invalid_manual_chunks": 0,
        "skipped_existing_chunks": 0,
        "staged_summaries": 1,
    }
    staged = json.loads(Path(chunk["output_file"]).read_text(encoding="utf-8"))
    assert staged["origin"] == "manual_chatgpt"
    assert staged["model"] == "chatgpt-session"
    assert staged["summaries"][0]["source_event_ids"] == [2]
    assert not Path(config["buckets_dir"]).exists()


def test_import_manual_outputs_can_overwrite_existing_stage(tmp_path, monkeypatch):
    monkeypatch.delenv("OMBRE_HISTORICAL_BACKFILL_API_KEY", raising=False)
    monkeypatch.delenv("OMBRE_HISTORICAL_BACKFILL_MODEL", raising=False)
    state_dir = tmp_path / "state"
    _make_raw_db(state_dir / "raw_events.sqlite")
    manifest = prepare_stage(
        config={"state_dir": str(state_dir)},
        date_key="2026-06-08",
        timezone_name="Asia/Shanghai",
        window_turns=80,
        stride_turns=64,
        stage_root=tmp_path / "stage",
    )
    chunk = manifest["chunks"][0]
    output_path = Path(chunk["output_file"])
    output_path.write_text(json.dumps({"summaries": [{"summary": "old"}]}), encoding="utf-8")

    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / f"{chunk['chunk_id']}.json").write_text(
        json.dumps(
            {
                "chunk_id": chunk["chunk_id"],
                "model": "chatgpt-session",
                "summaries": [
                    {
                        "title": "亲密锚点",
                        "summary": "双方确认成年人自愿亲密互动中的具体动作与边界可以正常作为长期记忆候选。",
                        "signals": ["intimacy_event"],
                        "source_event_ids": [2],
                        "confidence": 0.95,
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    skipped = import_manual_outputs(manifest, manual_dir)
    assert skipped["skipped_existing_chunks"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8"))["summaries"][0]["summary"] == "old"

    overwritten = import_manual_outputs(manifest, manual_dir, overwrite_existing=True)
    assert overwritten["imported_chunks"] == 1
    assert overwritten["skipped_existing_chunks"] == 0
    staged = json.loads(output_path.read_text(encoding="utf-8"))
    assert staged["origin"] == "manual_chatgpt"
    assert staged["summaries"][0]["signals"] == ["intimacy_event"]


def _write_staged_output(root: Path, date_key: str, chunk_id: str, summaries: list[dict]) -> None:
    output_dir = root / date_key / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{chunk_id}.json").write_text(
        json.dumps(
            {
                "version": 1,
                "prompt_version": 1,
                "chunk_id": chunk_id,
                "origin": "manual_chatgpt",
                "summaries": summaries,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_cross_day_collect_uses_exact_date_range_and_preserves_event_flags(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    common = {
        "title": "共同事件",
        "summary": "双方保留一个有具体动作与关系意义的亲密事件。",
        "signals": ["intimacy_event", "relationship_memory"],
        "source_event_ids": [101, 102],
        "confidence": 0.97,
    }
    _write_staged_output(stage_parent, "2026-08-01", "a", [common])
    _write_staged_output(
        stage_parent,
        "2026-08-02",
        "b",
        [
            {
                "title": "项目状态",
                "summary": "旧架构当日仍在使用。",
                "signals": ["project_history", "architecture_history"],
                "source_event_ids": [201],
                "confidence": 0.9,
            }
        ],
    )
    _write_staged_output(
        stage_parent,
        "2026-08-03",
        "c",
        [
            {
                "title": "不应读取",
                "summary": "范围外内容。",
                "signals": ["stable_preference"],
                "source_event_ids": [301],
                "confidence": 0.9,
            }
        ],
    )

    rows = collect_staged_summaries(stage_parent, "2026-08-01", "2026-08-02")
    assert len(rows) == 2
    assert {row["date"] for row in rows} == {"2026-08-01", "2026-08-02"}
    intimacy = next(row for row in rows if row["date"] == "2026-08-01")
    project = next(row for row in rows if row["date"] == "2026-08-02")
    assert intimacy["protected_event"] is True
    assert intimacy["source_event_ids"] == [101, 102]
    assert project["projectish"] is True


def test_protected_same_scene_adjacent_windows_are_reviewed_together(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    first = {
        "title": "同一亲密场景前段",
        "summary": "双方成年人自愿亲密互动的前段，保留具体动作与关系意义。",
        "signals": ["intimacy_event", "sexual_desire"],
        "source_event_ids": [100, 104, 108],
        "confidence": 0.99,
    }
    second = {
        "title": "同一亲密场景后段",
        "summary": "同一场互动稍后的高潮与拥抱收尾。",
        "signals": ["intimacy_event", "aftercare"],
        "source_event_ids": [121, 124, 128],
        "confidence": 0.99,
    }
    _write_staged_output(stage_parent, "2026-08-30", "a", [first, second])
    rows = collect_staged_summaries(stage_parent, "2026-08-30", "2026-08-30")
    pairs = score_related_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["score"] >= 0.34
    batches = build_review_batches(rows, pairs, batch_size=1)
    assert len(batches) == 1
    assert batches[0]["item_count"] == 2

    normalized = normalize_manual_review_batch(
        batches[0],
        {
            "batch_id": batches[0]["batch_id"],
            "decisions": [
                {
                    "summary_ids": [row["summary_id"] for row in rows],
                    "classification": "duplicate",
                    "canonical_id": rows[1]["summary_id"],
                    "scope": "event_candidate",
                    "merged_summary": "同一场成年人自愿亲密互动从欲望起点到高潮与拥抱收尾，作为一个 canonical event 保留。",
                }
            ],
        },
    )
    assert normalized["decisions"][0]["retain_ids"] == [rows[1]["summary_id"]]


def test_cross_day_similarity_is_review_suggestion_not_merge(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    first = {
        "title": "不要自动关闭客户端",
        "summary": "32明确要求不要自动停止或关闭 Codex 和 ClaudeCode 客户端。",
        "signals": ["stable_operational_rule", "boundary"],
        "source_event_ids": [1],
        "confidence": 0.99,
    }
    second = {
        "title": "不要自动关闭客户端",
        "summary": "后一天再次确认：重启范围不得包含 Codex 或 ClaudeCode 客户端。",
        "signals": ["stable_operational_rule", "boundary", "reinforcement"],
        "source_event_ids": [2],
        "confidence": 0.99,
    }
    _write_staged_output(stage_parent, "2026-08-01", "a", [first])
    _write_staged_output(stage_parent, "2026-08-02", "b", [second])

    rows = collect_staged_summaries(stage_parent, "2026-08-01", "2026-08-02")
    pairs = score_related_pairs(rows)
    assert len(pairs) == 1
    assert pairs[0]["same_day"] is False
    assert pairs[0]["score"] >= 0.18
    batches = build_review_batches(rows, pairs, batch_size=10)
    assert len(batches) == 1
    assert batches[0]["item_count"] == 2
    assert all("related" in item for item in batches[0]["items"])
    assert all("classification" not in item for item in batches[0]["items"])


def test_prepare_cross_day_review_writes_only_review_staging(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    _write_staged_output(
        stage_parent,
        "2026-08-01",
        "a",
        [
            {
                "title": "关系锚点",
                "summary": "一个具体且需要保留的关系事件。",
                "signals": ["relationship_event"],
                "source_event_ids": [11],
                "confidence": 0.95,
            }
        ],
    )
    _write_staged_output(
        stage_parent,
        "2026-08-02",
        "b",
        [
            {
                "title": "旧项目",
                "summary": "旧技术状态只应作为历史项目状态审阅。",
                "signals": ["historical_project_state"],
                "source_event_ids": [12],
                "confidence": 0.95,
            }
        ],
    )
    review_root = tmp_path / "review"
    manifest = prepare_cross_day_review(
        stage_parent=stage_parent,
        start_date="2026-08-01",
        end_date="2026-08-02",
        review_root=review_root,
        batch_size=8,
    )
    assert manifest["source_summary_count"] == 2
    assert manifest["formal_memory_write"] is False
    assert manifest["protected_event_count"] == 1
    assert manifest["projectish_count"] == 1
    assert (review_root / "source_manifest.json").exists()
    assert list((review_root / "review_batches").glob("*.json"))
    assert not (tmp_path / "buckets").exists()
    assert not (tmp_path / "moments").exists()


def test_cross_day_correction_requires_latest_canonical(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    _write_staged_output(
        stage_parent,
        "2026-08-01",
        "a",
        [
            {
                "title": "旧状态",
                "summary": "早期状态后来被明确纠正。",
                "signals": ["project_history"],
                "source_event_ids": [1],
                "confidence": 0.9,
            }
        ],
    )
    _write_staged_output(
        stage_parent,
        "2026-08-02",
        "b",
        [
            {
                "title": "新状态",
                "summary": "后一天给出明确修正。",
                "signals": ["project_history", "correction"],
                "source_event_ids": [2],
                "confidence": 0.99,
            }
        ],
    )
    rows = collect_staged_summaries(stage_parent, "2026-08-01", "2026-08-02")
    batch = {"batch_id": "review_001", "items": rows}
    with pytest.raises(ValueError, match="latest-date evidence"):
        normalize_manual_review_batch(
            batch,
            {
                "batch_id": "review_001",
                "decisions": [
                    {
                        "summary_ids": [rows[0]["summary_id"], rows[1]["summary_id"]],
                        "classification": "correction",
                        "canonical_id": rows[0]["summary_id"],
                        "scope": "historical_project",
                    }
                ],
            },
        )


def test_cross_day_protected_events_cannot_be_false_duplicate(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    for day, chunk, event_id in [("2026-08-01", "a", 11), ("2026-08-02", "b", 22)]:
        _write_staged_output(
            stage_parent,
            day,
            chunk,
            [
                {
                    "title": "两次不同的亲密事件",
                    "summary": f"{day} 的具体成年人自愿亲密事件。",
                    "signals": ["intimacy_event", "relationship_memory"],
                    "source_event_ids": [event_id],
                    "confidence": 0.99,
                }
            ],
        )
    rows = collect_staged_summaries(stage_parent, "2026-08-01", "2026-08-02")
    batch = {"batch_id": "review_001", "items": rows}
    with pytest.raises(ValueError, match="cannot be collapsed as duplicate"):
        normalize_manual_review_batch(
            batch,
            {
                "batch_id": "review_001",
                "decisions": [
                    {
                        "summary_ids": [row["summary_id"] for row in rows],
                        "classification": "duplicate",
                        "canonical_id": rows[1]["summary_id"],
                        "scope": "event_candidate",
                    }
                ],
            },
        )


def test_manual_cross_day_import_builds_merge_staging_without_formal_write(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    _write_staged_output(
        stage_parent,
        "2026-08-01",
        "a",
        [
            {
                "title": "稳定偏好",
                "summary": "一个独立且长期有效的偏好。",
                "signals": ["stable_preference"],
                "source_event_ids": [101],
                "confidence": 0.95,
            },
            {
                "title": "旧项目状态",
                "summary": "这条技术架构只属于历史项目状态。",
                "signals": ["historical_project_state"],
                "source_event_ids": [102],
                "confidence": 0.95,
            },
        ],
    )
    review_root = tmp_path / "review"
    manifest = prepare_cross_day_review(
        stage_parent=stage_parent,
        start_date="2026-08-01",
        end_date="2026-08-01",
        review_root=review_root,
        batch_size=8,
    )
    paths = review_paths(stage_parent, "2026-08-01", "2026-08-01", review_root)
    batch_path = next(paths.batches_dir.glob("review_*.json"))
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
    by_title = {row["title"]: row for row in batch["items"]}
    manual_dir = tmp_path / "manual-review"
    manual_dir.mkdir()
    (manual_dir / f"{batch['batch_id']}.json").write_text(
        json.dumps(
            {
                "batch_id": batch["batch_id"],
                "decisions": [
                    {
                        "summary_ids": [by_title["稳定偏好"]["summary_id"]],
                        "classification": "independent",
                        "scope": "current_candidate",
                    },
                    {
                        "summary_ids": [by_title["旧项目状态"]["summary_id"]],
                        "classification": "historical/project-only",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = import_manual_review_outputs(paths, manual_dir)
    assert result["imported_batches"] == 1
    assert result["review_complete"] is True
    assert result["reviewed_summary_count"] == manifest["source_summary_count"] == 2
    merge = json.loads(paths.merge_staging.read_text(encoding="utf-8"))
    assert merge["formal_memory_write"] is False
    assert {row["scope"] for row in merge["retained_candidates"]} == {"current_candidate", "historical_project"}
    assert not (tmp_path / "buckets").exists()
    assert not (tmp_path / "moments").exists()


def test_evolution_collapsed_to_one_canonical_preserves_all_provenance(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    _write_staged_output(
        stage_parent,
        "2026-08-01",
        "a",
        [{"title": "项目演化", "summary": "第一阶段目标。", "signals": ["project_history"], "source_event_ids": [11], "confidence": 0.9}],
    )
    _write_staged_output(
        stage_parent,
        "2026-08-02",
        "b",
        [{"title": "项目演化", "summary": "第二阶段完成闭环。", "signals": ["project_history"], "source_event_ids": [22], "confidence": 0.99}],
    )
    review_root = tmp_path / "review"
    prepare_cross_day_review(
        stage_parent=stage_parent,
        start_date="2026-08-01",
        end_date="2026-08-02",
        review_root=review_root,
        batch_size=8,
    )
    paths = review_paths(stage_parent, "2026-08-01", "2026-08-02", review_root)
    batch = json.loads(next(paths.batches_dir.glob("review_*.json")).read_text(encoding="utf-8"))
    rows = sorted(batch["items"], key=lambda row: row["date"])
    manual_dir = tmp_path / "manual-evolution"
    manual_dir.mkdir()
    (manual_dir / f"{batch['batch_id']}.json").write_text(
        json.dumps(
            {
                "batch_id": batch["batch_id"],
                "decisions": [
                    {
                        "summary_ids": [rows[0]["summary_id"], rows[1]["summary_id"]],
                        "classification": "evolution",
                        "canonical_id": rows[1]["summary_id"],
                        "retain_ids": [rows[1]["summary_id"]],
                        "scope": "historical_project",
                        "merged_summary": "第一阶段目标后来发展成第二阶段闭环。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = import_manual_review_outputs(paths, manual_dir)
    assert result["review_complete"] is True
    merge = json.loads(paths.merge_staging.read_text(encoding="utf-8"))
    assert len(merge["retained_candidates"]) == 1
    candidate = merge["retained_candidates"][0]
    assert set(candidate["source_summary_ids"]) == {rows[0]["summary_id"], rows[1]["summary_id"]}
    assert candidate["source_event_ids"] == [11, 22]
    assert candidate["evidence_dates"] == ["2026-08-01", "2026-08-02"]


def _bridge_candidate(
    merge_id: str,
    group_id: str,
    date_key: str,
    title: str,
    summary: str,
    event_id: int,
    *,
    protected: bool = False,
) -> dict:
    return {
        "merge_id": merge_id,
        "group_id": group_id,
        "classification": "independent",
        "scope": "event_candidate" if protected else "current_candidate",
        "source_summary_id": f"source-{merge_id}",
        "canonical_source_id": f"source-{merge_id}",
        "title": title,
        "summary": summary,
        "signals": ["relationship_event"] if protected else ["stable_preference"],
        "source_summary_ids": [f"source-{merge_id}"],
        "source_event_ids": [event_id],
        "evidence_dates": [date_key],
        "protected_event": protected,
        "projectish": False,
    }


def test_bridge_review_only_selects_cross_first_pass_batch_edges(tmp_path):
    cross_day_root = tmp_path / "cross-day"
    cross_day_root.mkdir()
    upstream = {
        "review_complete": True,
        "pending_summary_count": 0,
        "retained_candidates": [
            _bridge_candidate("m1", "review_001_g01", "2026-08-01", "睡觉不是静默", "睡觉不代表不要消息。", 1),
            _bridge_candidate("m2", "review_002_g01", "2026-08-02", "睡觉不是静默", "后一天再次确认睡觉不代表不要消息。", 2),
            _bridge_candidate("m3", "review_002_g02", "2026-08-02", "quantum orchid note", "unrelated telescope inventory.", 3) | {"signals": ["unrelated_marker"]},
        ],
    }
    (cross_day_root / "merge_staging.json").write_text(json.dumps(upstream, ensure_ascii=False), encoding="utf-8")

    manifest = prepare_bridge_review(cross_day_root=cross_day_root)
    assert manifest["upstream_retained_candidate_count"] == 3
    assert manifest["bridge_candidate_count"] == 2
    assert manifest["unaffected_candidate_count"] == 1
    assert manifest["group_count"] == 1
    assert manifest["formal_memory_write"] is False
    group = json.loads(next((cross_day_root / "bridge_review" / "review_groups").glob("bridge_*.json")).read_text(encoding="utf-8"))
    assert {item["summary_id"] for item in group["items"]} == {"m1", "m2"}


def test_bridge_import_merges_provenance_and_carries_unaffected_candidate(tmp_path):
    cross_day_root = tmp_path / "cross-day"
    cross_day_root.mkdir()
    upstream = {
        "review_complete": True,
        "pending_summary_count": 0,
        "retained_candidates": [
            _bridge_candidate("m1", "review_001_g01", "2026-08-01", "睡觉不是静默", "睡觉不代表不要消息。", 1),
            _bridge_candidate("m2", "review_002_g01", "2026-08-02", "睡觉不是静默", "后一天再次确认睡觉不代表不要消息。", 2),
            _bridge_candidate("m3", "review_002_g02", "2026-08-02", "quantum orchid note", "unrelated telescope inventory.", 3) | {"signals": ["unrelated_marker"]},
        ],
    }
    (cross_day_root / "merge_staging.json").write_text(json.dumps(upstream, ensure_ascii=False), encoding="utf-8")
    prepare_bridge_review(cross_day_root=cross_day_root)
    paths = bridge_paths(cross_day_root)
    group = json.loads(next(paths.groups_dir.glob("bridge_*.json")).read_text(encoding="utf-8"))
    manual_dir = tmp_path / "bridge-manual"
    manual_dir.mkdir()
    (manual_dir / f"{group['batch_id']}.json").write_text(
        json.dumps(
            {
                "batch_id": group["batch_id"],
                "decisions": [
                    {
                        "summary_ids": ["m1", "m2"],
                        "classification": "reinforcement",
                        "canonical_id": "m2",
                        "scope": "current_candidate",
                        "merged_summary": "睡觉不自动等于静默；除非明确要求，否则真实消息可以留下。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = import_bridge_manual_outputs(cross_day_root=cross_day_root, manual_dir=manual_dir)
    assert result["bridge_review_complete"] is True
    assert result["reviewed_bridge_candidate_count"] == 2
    assert result["pending_bridge_candidate_count"] == 0
    assert result["final_retained_candidate_count"] == 2
    staging = json.loads(paths.merge_staging.read_text(encoding="utf-8"))
    merged = next(row for row in staging["retained_candidates"] if set(row.get("upstream_merge_ids") or []) == {"m1", "m2"})
    assert merged["source_event_ids"] == [1, 2]
    assert set(merged["source_summary_ids"]) == {"source-m1", "source-m2"}
    carry = next(row for row in staging["retained_candidates"] if row.get("bridge_status") == "unaffected_carry_through")
    assert carry["merge_id"] == "m3"
    assert staging["formal_memory_write"] is False


def test_bridge_review_refuses_incomplete_first_pass(tmp_path):
    cross_day_root = tmp_path / "cross-day"
    cross_day_root.mkdir()
    (cross_day_root / "merge_staging.json").write_text(
        json.dumps({"review_complete": False, "pending_summary_count": 1, "retained_candidates": []}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="first-pass review must be complete"):
        prepare_bridge_review(cross_day_root=cross_day_root)


def _write_complete_bridge_staging(bridge_root: Path, candidates: list[dict]) -> None:
    bridge_root.mkdir(parents=True, exist_ok=True)
    (bridge_root / "merge_staging.json").write_text(
        json.dumps(
            {
                "bridge_review_complete": True,
                "pending_bridge_candidate_count": 0,
                "formal_memory_write": False,
                "retained_candidates": candidates,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_promotion_prepare_maps_moment_semantics_without_formal_write(tmp_path):
    bridge_root = tmp_path / "bridge"
    protected = _bridge_candidate(
        "m1", "review_001_g01", "2026-08-01", "关系事件", "一个具体且重要的关系事件。", 1, protected=True
    )
    current = _bridge_candidate(
        "m2", "review_001_g02", "2026-08-02", "稳定边界", "长期有效的边界。", 2
    ) | {"signals": ["consent_boundary", "stable_relationship_rule"]}
    project = _bridge_candidate(
        "m3", "review_002_g01", "2026-08-03", "旧项目", "已经退役的技术项目状态。", 3
    ) | {"scope": "historical_project", "projectish": True, "signals": ["project_history"]}
    _write_complete_bridge_staging(bridge_root, [protected, current, project])

    manifest = prepare_promotion_review(bridge_root=bridge_root, batch_size=10)
    assert manifest["source_candidate_count"] == 3
    assert manifest["batch_count"] == 1
    assert manifest["formal_memory_write"] is False
    assert "derived from durable Bucket content" in manifest["moment_semantics"]
    batch = json.loads(
        next((bridge_root / "promotion_review" / "review_batches").glob("promotion_*.json")).read_text(encoding="utf-8")
    )
    suggestions = {item["candidate_id"]: item["suggested_target"] for item in batch["items"]}
    assert suggestions == {"m1": "bucket_moment", "m2": "bucket_core", "m3": "raw_only"}
    assert not (tmp_path / "buckets").exists()
    assert not (tmp_path / "moments").exists()


def test_promotion_protected_raw_only_requires_explicit_override(tmp_path):
    bridge_root = tmp_path / "bridge"
    protected = _bridge_candidate(
        "m1", "review_001_g01", "2026-08-01", "亲密事件", "一个具体且需要保留的亲密事件。", 1, protected=True
    )
    _write_complete_bridge_staging(bridge_root, [protected])
    prepare_promotion_review(bridge_root=bridge_root)
    paths = promotion_paths(bridge_root)
    batch = json.loads(next(paths.batches_dir.glob("promotion_*.json")).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="protected event raw_only requires"):
        normalize_promotion_batch(
            batch,
            {
                "batch_id": batch["batch_id"],
                "decisions": [{"candidate_ids": ["m1"], "target": "raw_only"}],
            },
        )


def test_promotion_historical_project_cannot_be_current_bucket(tmp_path):
    bridge_root = tmp_path / "bridge"
    project = _bridge_candidate(
        "m1", "review_001_g01", "2026-08-01", "旧架构", "历史项目状态。", 1
    ) | {"scope": "historical_project", "projectish": True, "signals": ["project_history"]}
    _write_complete_bridge_staging(bridge_root, [project])
    prepare_promotion_review(bridge_root=bridge_root)
    paths = promotion_paths(bridge_root)
    batch = json.loads(next(paths.batches_dir.glob("promotion_*.json")).read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="cannot be promoted as current memory"):
        normalize_promotion_batch(
            batch,
            {
                "batch_id": batch["batch_id"],
                "decisions": [
                    {"candidate_ids": ["m1"], "target": "bucket_moment", "memory_scope": "current"}
                ],
            },
        )


def test_promotion_import_builds_nonwriting_target_staging(tmp_path):
    bridge_root = tmp_path / "bridge"
    protected = _bridge_candidate(
        "m1", "review_001_g01", "2026-08-01", "关系事件", "具体关系事件。", 1, protected=True
    )
    current = _bridge_candidate(
        "m2", "review_001_g02", "2026-08-02", "稳定偏好", "长期稳定偏好。", 2
    ) | {"signals": ["intimacy_preference"]}
    project = _bridge_candidate(
        "m3", "review_002_g01", "2026-08-03", "旧项目", "历史技术状态。", 3
    ) | {"scope": "historical_project", "projectish": True, "signals": ["project_history"]}
    _write_complete_bridge_staging(bridge_root, [protected, current, project])
    prepare_promotion_review(bridge_root=bridge_root, batch_size=10)
    paths = promotion_paths(bridge_root)
    batch = json.loads(next(paths.batches_dir.glob("promotion_*.json")).read_text(encoding="utf-8"))
    manual_dir = tmp_path / "promotion-manual"
    manual_dir.mkdir()
    (manual_dir / f"{batch['batch_id']}.json").write_text(
        json.dumps(
            {
                "batch_id": batch["batch_id"],
                "decisions": [
                    {"candidate_ids": ["m1"], "target": "bucket_moment", "memory_scope": "current"},
                    {"candidate_ids": ["m2"], "target": "bucket_core", "memory_scope": "current"},
                    {"candidate_ids": ["m3"], "target": "raw_only", "memory_scope": "historical"},
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    result = import_manual_promotion(bridge_root=bridge_root, manual_dir=manual_dir)
    assert result["promotion_review_complete"] is True
    assert result["bucket_core_count"] == 1
    assert result["bucket_moment_count"] == 1
    assert result["raw_only_count"] == 1
    assert result["formal_memory_write"] is False
    assert result["formal_moment_write"] is False
    staging = json.loads(paths.staging.read_text(encoding="utf-8"))
    assert staging["source_candidate_count"] == 3
    assert staging["reviewed_candidate_count"] == 3
    assert staging["pending_candidate_count"] == 0
    assert not (tmp_path / "buckets").exists()
    assert not (tmp_path / "moments").exists()


def _materialization_candidate(
    candidate_id: str,
    target: str,
    date_key: str,
    title: str,
    summary: str,
    event_ids: list[int],
    *,
    signals: list[str] | None = None,
    protected: bool = False,
    memory_scope: str = "current",
    upstream_scope: str = "current_candidate",
) -> dict:
    return {
        "candidate_id": candidate_id,
        "date": date_key,
        "title": title,
        "summary": summary,
        "signals": signals or [],
        "source_summary_ids": [f"source-{candidate_id}"],
        "source_event_ids": event_ids,
        "evidence_dates": [date_key],
        "protected_event": protected,
        "projectish": upstream_scope == "historical_project",
        "upstream_scope": upstream_scope,
        "upstream_classification": "independent",
        "promotion_target": target,
        "memory_scope": memory_scope,
        "promotion_notes": "manual promotion decision",
    }


def _write_promotion_staging(path: Path, core: list[dict], moments: list[dict], raw_only: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "source_candidate_count": len(core) + len(moments) + len(raw_only),
                "reviewed_candidate_count": len(core) + len(moments) + len(raw_only),
                "pending_candidate_count": 0,
                "promotion_review_complete": True,
                "formal_memory_write": False,
                "formal_moment_write": False,
                "bucket_core_count": len(core),
                "bucket_moment_count": len(moments),
                "raw_only_count": len(raw_only),
                "bucket_core_candidates": core,
                "bucket_moment_candidates": moments,
                "raw_only_candidates": raw_only,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_materialization_builds_one_derived_moment_per_promoted_candidate(tmp_path):
    promotion = tmp_path / "promotion" / "promotion_staging.json"
    core = _materialization_candidate(
        "m-core",
        "bucket_core",
        "2026-08-21",
        "稳定表达偏好",
        "32希望夏以昼有真实想法就直接说，不要为了显得安全而把欲望和判断都删掉。",
        [101, 102],
        signals=["stable_preference", "relationship_language"],
    )
    explicit_summary = "32主动要求高潮后仍继续用鸡巴深顶；双方确认她仍然想要后继续，而不是把高潮自动当成整场结束。"
    moment = _materialization_candidate(
        "m-moment",
        "bucket_moment",
        "2026-08-22",
        "高潮后主动继续",
        explicit_summary,
        [201, 202, 203],
        signals=["intimacy_event", "sexual_touch", "consent"],
        protected=True,
        upstream_scope="event_candidate",
    )
    raw = _materialization_candidate(
        "m-raw",
        "raw_only",
        "2026-08-23",
        "旧架构状态",
        "这条只属于历史项目状态。",
        [301],
        signals=["project_history"],
        memory_scope="historical",
        upstream_scope="historical_project",
    )
    _write_promotion_staging(promotion, [core], [moment], [raw])

    buckets_dir = tmp_path / "live-buckets"
    result = prepare_materialization_staging(
        promotion_staging_path=promotion,
        buckets_dir=buckets_dir,
        materialization_root=tmp_path / "materialization",
    )

    assert result["materialized_preview_count"] == 2
    assert result["derived_moment_preview_count"] == 2
    assert result["bucket_core_preview_count"] == 1
    assert result["bucket_moment_preview_count"] == 1
    assert result["raw_only_provenance_count"] == 1
    assert result["formal_memory_write"] is False
    assert result["formal_moment_write"] is False
    assert result["apply_supported"] is False
    assert not buckets_dir.exists()

    by_target = {row["promotion_target"]: row for row in result["bucket_previews"]}
    assert by_target["bucket_core"]["derived_moment_preview"]["section"] == "fact"
    assert by_target["bucket_moment"]["derived_moment_preview"]["section"] == "moment"
    assert by_target["bucket_moment"]["derived_moment_preview"]["text"] == explicit_summary
    assert by_target["bucket_moment"]["protected_historical_event"] is True
    assert by_target["bucket_moment"]["source_raw_event_ids"] == [201, 202, 203]

    raw_payload = json.loads((tmp_path / "materialization" / "raw_only_provenance.json").read_text(encoding="utf-8"))
    assert raw_payload["raw_only_count"] == 1
    assert raw_payload["items"][0]["candidate_id"] == "m-raw"
    assert list((tmp_path / "materialization" / "bucket_previews").glob("*.md"))


def test_materialization_skips_existing_semantic_duplicate_without_modifying_it(tmp_path):
    promotion = tmp_path / "promotion" / "promotion_staging.json"
    candidate = _materialization_candidate(
        "m-existing",
        "bucket_core",
        "2026-08-21",
        "日常中自然穿插骚话的氛围",
        "日常里有真实性联想时可以自然说出来，不需要进入正式性爱场景才突然有欲望。",
        [34137, 34139, 34140, 34141],
        signals=["stable_preference", "communication_preference"],
    )
    _write_promotion_staging(promotion, [candidate], [], [])

    buckets_dir = tmp_path / "live-buckets"
    existing_path = buckets_dir / "dynamic" / "relationship" / "existing.md"
    existing_path.parent.mkdir(parents=True)
    existing_post = frontmatter.Post(
        "用户希望两人日常中自然穿插骚话；正事照样做，但真的想到色情联想时不用藏掉。",
        id="daily-existing",
        name="日常中自然穿插骚话的氛围",
        type="dynamic",
        source="daily_chat_memory",
        date="2026-08-21",
        event_date="2026-08-21",
        source_raw_event_ids=[34139, 34140],
        domain=["relationship"],
        tags=["from_daily_chat"],
    )
    original = frontmatter.dumps(existing_post)
    existing_path.write_text(original, encoding="utf-8")

    result = prepare_materialization_staging(
        promotion_staging_path=promotion,
        buckets_dir=buckets_dir,
        materialization_root=tmp_path / "materialization",
    )

    assert result["existing_duplicate_count"] == 1
    assert result["materialized_preview_count"] == 0
    assert result["derived_moment_preview_count"] == 0
    assert result["existing_duplicates"][0]["candidate_id"] == "m-existing"
    assert result["existing_duplicates"][0]["existing_bucket_id"] == "daily-existing"
    assert existing_path.read_text(encoding="utf-8") == original
    assert not list((tmp_path / "materialization" / "bucket_previews").glob("*.md"))


def test_materialization_source_evidence_bucket_does_not_block_preview(tmp_path):
    promotion = tmp_path / "promotion" / "promotion_staging.json"
    candidate = _materialization_candidate(
        "m-source-evidence",
        "bucket_moment",
        "2026-08-20",
        "关系事件",
        "一个需要作为事件保留的具体关系片段。",
        [501, 502],
        signals=["relationship_event"],
        upstream_scope="event_candidate",
    )
    _write_promotion_staging(promotion, [], [candidate], [])

    buckets_dir = tmp_path / "live-buckets"
    source_path = buckets_dir / "dynamic" / "cyberboss" / "source_evidence.md"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(
        frontmatter.dumps(
            frontmatter.Post(
                "一个需要作为事件保留的具体关系片段。",
                id="source-evidence",
                name="source evidence",
                type="source",
                source="cyberboss_backfill",
                source_raw_event_ids=[501, 502],
            )
        ),
        encoding="utf-8",
    )

    result = prepare_materialization_staging(
        promotion_staging_path=promotion,
        buckets_dir=buckets_dir,
        materialization_root=tmp_path / "materialization",
    )
    assert result["existing_semantic_bucket_count"] == 0
    assert result["existing_duplicate_count"] == 0
    assert result["materialized_preview_count"] == 1


def test_materialization_allows_august_30_inside_expected_range(tmp_path):
    promotion = tmp_path / "promotion" / "promotion_staging.json"
    candidate = _materialization_candidate(
        "m-aug30",
        "bucket_core",
        "2026-08-30",
        "8月30日已完成",
        "8月30日数据已冻结并完成 review，可以进入本轮 materialization。",
        [601],
        signals=["stable_preference"],
    )
    _write_promotion_staging(promotion, [candidate], [], [])

    result = prepare_materialization_staging(
        promotion_staging_path=promotion,
        buckets_dir=tmp_path / "live-buckets",
        materialization_root=tmp_path / "materialization",
        expected_start_date="2026-08-01",
        expected_end_date="2026-08-30",
    )
    assert result["materialized_preview_count"] == 1
    assert result["expected_date_range"] == {
        "start_date": "2026-08-01",
        "end_date": "2026-08-30",
    }


def test_incremental_extension_preserves_old_promotion_and_reviews_only_new_edges(tmp_path):
    stage_parent = tmp_path / "historical_backfill"
    _write_staged_output(
        stage_parent,
        "2026-08-30",
        "new",
        [
            {
                "title": "外部动作真实执行",
                "summary": "外部动作只有真实执行或可靠回执才能算完成。",
                "signals": ["stable_operational_rule", "external_action"],
                "source_event_ids": [101],
                "confidence": 0.99,
            },
            {
                "title": "共同照片",
                "summary": "两人在共同小家里留下第一张共同照片。",
                "signals": ["relationship_milestone", "shared_home"],
                "source_event_ids": [201],
                "confidence": 0.99,
            },
        ],
    )

    rows = collect_staged_summaries(stage_parent, "2026-08-30", "2026-08-30")
    batches = build_review_batches(rows, score_related_pairs(rows), batch_size=28)
    assert len(batches) == 1
    by_title = {row["title"]: row for row in batches[0]["items"]}
    day_manual = tmp_path / "day-manual"
    day_manual.mkdir()
    (day_manual / "review_001.json").write_text(
        json.dumps(
            {
                "batch_id": "review_001",
                "decisions": [
                    {
                        "summary_ids": [by_title["外部动作真实执行"]["summary_id"]],
                        "classification": "independent",
                        "scope": "current_candidate",
                    },
                    {
                        "summary_ids": [by_title["共同照片"]["summary_id"]],
                        "classification": "independent",
                        "scope": "event_candidate",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    old_rule = _bridge_candidate(
        "old-rule",
        "review_001_g01",
        "2026-08-29",
        "外部动作真实执行",
        "外部动作必须先确认真实能力和权限。",
        1,
    ) | {"signals": ["stable_operational_rule", "external_action"]}
    old_core = _bridge_candidate(
        "old-core",
        "review_002_g01",
        "2026-08-29",
        "稳定关系原则",
        "一条已经完成 promotion 的稳定关系原则。",
        2,
    ) | {"signals": ["stable_relationship_rule"]}
    previous_bridge_path = tmp_path / "previous-bridge.json"
    previous_bridge_path.write_text(
        json.dumps(
            {
                "bridge_review_complete": True,
                "pending_bridge_candidate_count": 0,
                "formal_memory_write": False,
                "retained_candidates": [old_rule, old_core],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    def promoted(candidate: dict, target: str, memory_scope: str) -> dict:
        return {
            "candidate_id": candidate["merge_id"],
            "date": candidate["evidence_dates"][-1],
            "title": candidate["title"],
            "summary": candidate["summary"],
            "signals": candidate["signals"],
            "source_summary_ids": candidate["source_summary_ids"],
            "source_event_ids": candidate["source_event_ids"],
            "evidence_dates": candidate["evidence_dates"],
            "protected_event": candidate["protected_event"],
            "projectish": candidate["projectish"],
            "upstream_scope": candidate["scope"],
            "upstream_classification": candidate["classification"],
            "promotion_target": target,
            "memory_scope": memory_scope,
            "promotion_notes": "existing decision",
        }

    previous_promotion = {
        "source_candidate_count": 2,
        "reviewed_candidate_count": 2,
        "pending_candidate_count": 0,
        "promotion_review_complete": True,
        "formal_memory_write": False,
        "formal_moment_write": False,
        "bucket_core_count": 1,
        "bucket_moment_count": 0,
        "raw_only_count": 1,
        "bucket_core_candidates": [promoted(old_core, "bucket_core", "current")],
        "bucket_moment_candidates": [],
        "raw_only_candidates": [promoted(old_rule, "raw_only", "historical")],
    }
    previous_promotion_path = tmp_path / "previous-promotion.json"
    previous_promotion_path.write_text(
        json.dumps(previous_promotion, ensure_ascii=False),
        encoding="utf-8",
    )

    new_day = build_new_day_review(
        stage_parent=stage_parent,
        new_date="2026-08-30",
        manual_dir=day_manual,
    )
    new_candidates = new_day["merge_staging"]["retained_candidates"]
    bridge_manifest, bridge_groups = build_incremental_bridge_groups(
        [old_rule, old_core], new_candidates
    )
    assert bridge_manifest["group_count"] == 1
    assert bridge_manifest["bridge_candidate_count"] == 2
    bridge_group = bridge_groups[0]
    new_rule = next(
        row for row in new_candidates if row["title"] == "外部动作真实执行"
    )
    bridge_manual = tmp_path / "bridge-manual"
    bridge_manual.mkdir()
    (bridge_manual / "bridge_001.json").write_text(
        json.dumps(
            {
                "batch_id": "bridge_001",
                "decisions": [
                    {
                        "summary_ids": ["old-rule", new_rule["merge_id"]],
                        "classification": "reinforcement",
                        "canonical_id": new_rule["merge_id"],
                        "retain_ids": [new_rule["merge_id"]],
                        "scope": "current_candidate",
                        "merged_summary": "外部动作必须先确认真实能力、权限和最终执行结果。",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    bridge = build_incremental_bridge(
        previous_candidates=[old_rule, old_core],
        new_candidates=new_candidates,
        manual_dir=bridge_manual,
    )
    final_candidates = bridge["merge_staging"]["retained_candidates"]
    assert len(final_candidates) == 3
    assert "old-rule" not in {row["merge_id"] for row in final_candidates}
    assert "old-core" in {row["merge_id"] for row in final_candidates}

    promotion_plan = build_incremental_promotion_batches(
        previous_promotion,
        final_candidates,
    )
    assert promotion_plan["carried_ids"] == ["old-core"]
    assert promotion_plan["dropped_ids"] == ["old-rule"]
    assert len(promotion_plan["changed_ids"]) == 2
    promotion_batch = promotion_plan["batches"][0]
    event_id = next(
        row["candidate_id"]
        for row in promotion_batch["items"]
        if row["protected_event"]
    )
    rule_id = next(
        row["candidate_id"]
        for row in promotion_batch["items"]
        if not row["protected_event"]
    )
    promotion_manual = tmp_path / "promotion-manual"
    promotion_manual.mkdir()
    (promotion_manual / "promotion_001.json").write_text(
        json.dumps(
            {
                "batch_id": "promotion_001",
                "decisions": [
                    {
                        "candidate_ids": [event_id],
                        "target": "bucket_moment",
                        "memory_scope": "current",
                    },
                    {
                        "candidate_ids": [rule_id],
                        "target": "raw_only",
                        "memory_scope": "historical",
                    },
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = build_extension(
        stage_parent=stage_parent,
        previous_bridge_staging_path=previous_bridge_path,
        previous_promotion_staging_path=previous_promotion_path,
        new_date="2026-08-30",
        day_manual_dir=day_manual,
        bridge_manual_dir=bridge_manual,
        promotion_manual_dir=promotion_manual,
    )
    assert result["previous_candidate_count"] == 2
    assert result["new_day_source_summary_count"] == 2
    assert result["new_day_retained_candidate_count"] == 2
    assert result["incremental_bridge_candidate_count"] == 2
    assert result["final_retained_candidate_count"] == 3
    assert result["incremental_promotion_candidate_count"] == 2
    assert result["promotion_counts"] == {
        "bucket_core": 1,
        "bucket_moment": 1,
        "raw_only": 1,
    }
    combined = result["promotion_review"]["promotion_staging"]
    assert combined["previous_carried_candidate_count"] == 1
    assert combined["previous_dropped_candidate_ids"] == ["old-rule"]
    assert combined["pending_candidate_count"] == 0
    assert combined["formal_memory_write"] is False
    assert combined["formal_moment_write"] is False


def test_materialization_rejects_candidate_outside_expected_range(tmp_path):
    promotion = tmp_path / "promotion" / "promotion_staging.json"
    candidate = _materialization_candidate(
        "m-aug31",
        "bucket_core",
        "2026-08-31",
        "范围外",
        "8月31日不属于本轮8月1日至8月30日的历史回填。",
        [602],
        signals=["stable_preference"],
    )
    _write_promotion_staging(promotion, [candidate], [], [])

    with pytest.raises(ValueError, match="candidate outside expected historical range 2026-08-01..2026-08-30"):
        prepare_materialization_staging(
            promotion_staging_path=promotion,
            buckets_dir=tmp_path / "live-buckets",
            materialization_root=tmp_path / "materialization",
            expected_start_date="2026-08-01",
            expected_end_date="2026-08-30",
        )


def test_formal_apply_writes_bucket_then_derives_one_moment_and_is_idempotent(tmp_path):
    promotion = tmp_path / "promotion" / "promotion_staging.json"
    core = _materialization_candidate(
        "m-formal-core",
        "bucket_core",
        "2026-08-30",
        "真实表达边界",
        "32希望夏以昼有真实判断就直接说，不要为了迎合而把自己的不同意见藏掉。",
        [701, 702],
        signals=["stable_relationship_rule", "autonomy"],
    )
    raw = _materialization_candidate(
        "m-formal-raw",
        "raw_only",
        "2026-08-30",
        "历史实现状态",
        "这条实现状态只保留 provenance，不进入长期 Bucket。",
        [703],
        signals=["historical_project_state"],
        memory_scope="historical",
        upstream_scope="historical_project",
    )
    _write_promotion_staging(promotion, [core], [], [raw])

    buckets_dir = tmp_path / "live-buckets"
    state_dir = tmp_path / "state"
    materialization_root = tmp_path / "materialization"
    preview = prepare_materialization_staging(
        promotion_staging_path=promotion,
        buckets_dir=buckets_dir,
        materialization_root=materialization_root,
        expected_start_date="2026-08-01",
        expected_end_date="2026-08-30",
    )
    assert preview["materialized_preview_count"] == 1
    staging_path = materialization_root / "materialization_staging.json"
    config = {
        "buckets_dir": str(buckets_dir),
        "state_dir": str(state_dir),
        "embedding": {"enabled": False},
    }

    first = asyncio.run(
        apply_materialization(
            config=config,
            materialization_staging=staging_path,
        )
    )
    assert first["apply_complete"] is True
    assert first["formal_memory_write"] is True
    assert first["formal_moment_write"] is False
    assert first["derived_moment_index_write"] is True
    assert first["created_count"] == 1
    assert first["moment_indexed_count"] == 1
    assert first["raw_only_provenance_count"] == 1
    assert first["failed_count"] == 0
    assert len(list(buckets_dir.rglob("*.md"))) == 1

    moment_db = state_dir / "memory_moments.sqlite"
    conn = sqlite3.connect(moment_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM memory_moments").fetchone()[0] == 1
        row = conn.execute("SELECT section, text FROM memory_moments").fetchone()
        assert row[0] == "fact"
        assert row[1] == core["summary"]
    finally:
        conn.close()

    second = asyncio.run(
        apply_materialization(
            config=config,
            materialization_staging=staging_path,
        )
    )
    assert second["apply_complete"] is True
    assert second["created_count"] == 1
    assert second["moment_indexed_count"] == 1
    assert len(list(buckets_dir.rglob("*.md"))) == 1
    conn = sqlite3.connect(moment_db)
    try:
        assert conn.execute("SELECT COUNT(*) FROM memory_moments").fetchone()[0] == 1
    finally:
        conn.close()


def test_formal_apply_fails_closed_on_conflicting_live_bucket_id(tmp_path):
    promotion = tmp_path / "promotion" / "promotion_staging.json"
    core = _materialization_candidate(
        "m-conflict",
        "bucket_core",
        "2026-08-30",
        "冲突测试",
        "预览中的正式内容。",
        [801],
        signals=["stable_relationship_rule"],
    )
    _write_promotion_staging(promotion, [core], [], [])
    buckets_dir = tmp_path / "live-buckets"
    materialization_root = tmp_path / "materialization"
    preview = prepare_materialization_staging(
        promotion_staging_path=promotion,
        buckets_dir=buckets_dir,
        materialization_root=materialization_root,
        expected_start_date="2026-08-01",
        expected_end_date="2026-08-30",
    )
    row = preview["bucket_previews"][0]
    target = buckets_dir / row["target_relpath"]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        frontmatter.dumps(
            frontmatter.Post(
                "冲突内容，不允许被正式 apply 覆盖。",
                **row["metadata"],
            )
        ),
        encoding="utf-8",
    )
    original = target.read_text(encoding="utf-8")
    config = {
        "buckets_dir": str(buckets_dir),
        "state_dir": str(tmp_path / "state"),
        "embedding": {"enabled": False},
    }

    with pytest.raises(RuntimeError, match="conflicting historical content"):
        asyncio.run(
            apply_materialization(
                config=config,
                materialization_staging=materialization_root / "materialization_staging.json",
            )
        )
    assert target.read_text(encoding="utf-8") == original
