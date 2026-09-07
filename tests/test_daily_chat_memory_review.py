"""Review lifecycle tests using synthetic dialogue, temporary state and mocked LLMs."""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from reflection_engine import ReflectionEngine


DATE = "2026-09-02"
CANDIDATE = {
    "kind": "stable_preference", "title": "安静的工作环境",
    "content": "用户明确表示工作时长期偏好安静的环境，不喜欢背景音乐。",
    "confidence": 0.9,
}


class ReadOnlyBuckets:
    async def list_all(self, **kwargs):
        return []

    async def get(self, *args):
        raise AssertionError("review/repair must not access the write path")

    async def create(self, **kwargs):
        raise AssertionError("review/repair must not create long-term memories")


class Events:
    def __init__(self, count=256):
        self.rows = [
            {"id": index + 1001, "role": "user", "text": "工作时我喜欢安静，不喜欢背景音乐。",
             "created_at": f"{DATE}T12:{index // 60:02d}:{index % 60:02d}+08:00",
             "session_id": "synthetic-session", "metadata": {"profile_id": "default"}}
            for index in range(count)
        ]

    def list_events_between(self, **kwargs):
        return list(self.rows)


@pytest.fixture
def engine(tmp_path):
    return ReflectionEngine({
        "state_dir": str(tmp_path / "state"),
        "identity": {"ai_name": "测试助手", "user_name": "用户"},
        "reflection": {"enabled": True, "daily_chat_memory_mode": "review",
                       "daily_chat_memory_summary_enabled": False,
                       "daily_chat_memory_review_min_confidence": 0.0},
    })


def model(monkeypatch, engine, payload, candidates=None):
    calls = []
    monkeypatch.setattr(engine, "_daily_chat_memory_model_client", lambda **kwargs: (object(), "fake", False))

    async def completion(*args, **kwargs):
        request = json.loads(kwargs["messages"][1]["content"])
        calls.append(request)
        if "candidate_memories" in request:
            if isinstance(payload, Exception):
                raise payload
            response = payload
        else:
            response = {"candidates": candidates if candidates is not None else [dict(CANDIDATE)]}
        content = response if isinstance(response, str) else json.dumps(response)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    monkeypatch.setattr(engine, "_daily_chat_memory_create_completion", completion)
    return calls


def run(engine, events, mode="review"):
    return asyncio.run(engine.run_daily_chat_memory(
        ReadOnlyBuckets(), conversation_turn_store=object(), raw_event_store=events,
        key=DATE, mode=mode, force=True,
    ))


@pytest.mark.parametrize("payload,error", [
    ({"alignments": "invalid"}, "invalid_payload"),
    ("truncated {", "invalid_payload"),
    ({"alignments": []}, "no_matching_evidence"),
    ({"alignments": [{"candidate_index": 0, "source_turn_refs": [9999]}]}, "no_matching_evidence"),
    (TimeoutError("synthetic timeout"), "alignment_failed"),
])
def test_review_failure_survives_storage_and_reload(engine, monkeypatch, payload, error):
    # IDs supplied during extraction must not survive a failed alignment.
    candidates = [{**CANDIDATE, "source_event_ids": list(range(1001, 1081)), "source_turn_ids": [17]}]
    model(monkeypatch, engine, payload, candidates)
    result = run(engine, Events())
    assert result["status"] == "pending"
    assert result["diagnostics"] == {"extracted": 1, "retained": 1, "filtered": 0, "aligned": 0, "needs_repair": 1}
    rows = engine.list_daily_chat_memory_pending()
    assert len(rows) == 1 and rows[0]["status"] == "pending"
    candidate = rows[0]["candidate"]
    assert candidate["provenance_status"] == "needs_repair"
    assert candidate["provenance_error"] == error
    assert candidate["source_turn_ids"] == candidate["source_event_ids"] == []
    assert len(candidate["provenance_context"]["input_event_ids"]) == 256
    assert engine._daily_chat_memory_materials_for_date(DATE) == []


def test_review_success_maps_only_selected_events(engine, monkeypatch):
    model(monkeypatch, engine, {"alignments": [{"candidate_index": 0, "source_turn_refs": [17, 18]}]})
    result = run(engine, Events())
    candidate = result["candidates"][0]
    assert candidate["provenance_status"] == "aligned"
    assert candidate["source_event_ids"] == [1017, 1018]
    assert candidate["source_turn_ids"] == []  # raw-event refs are not persistent turn IDs
    assert result["status"] == "pending" and result["added"] == 1
    assert engine.list_daily_chat_memory_pending()[0]["candidate"]["provenance_status"] == "aligned"


def test_daily_chat_memory_completion_disables_thinking_with_provider_contract(engine):
    options = engine._daily_chat_memory_completion_options(max_tokens=1600, temperature=0.0)
    assert options["response_format"] == {"type": "json_object"}
    assert options["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "enable_thinking" not in options["extra_body"]


def test_daily_chat_memory_fallback_client_also_disables_thinking(engine):
    calls = {}

    class Completions:
        async def create(self, **kwargs):
            calls.update(kwargs)
            return SimpleNamespace(choices=[])

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    asyncio.run(engine._daily_chat_memory_create_completion(
        client,
        model="fake",
        messages=[{"role": "user", "content": "{}"}],
        max_tokens=1600,
        temperature=0.0,
        use_daily_client=False,
    ))
    assert calls["response_format"] == {"type": "json_object"}
    assert calls["extra_body"] == {"thinking": {"type": "disabled"}}


@pytest.mark.parametrize("payload", [
    {"alignments": [{"candidate_index": 0, "source_turn_ids": [17, 18]}]},
    {"alignments": [{"candidate_index": 0, "turn_refs": [17, 18]}]},
    {"alignments": {"candidate_index": 0, "source_turn_ids": [17, 18]}},
    {"candidate_index": 0, "source_turn_ids": [17, 18]},
    {"candidate_index": 0, "turn_refs": [17, 18]},
])
def test_review_normalizes_provenance_transport_shape(engine, monkeypatch, payload):
    model(monkeypatch, engine, payload)
    candidate = run(engine, Events())["candidates"][0]
    assert candidate["provenance_status"] == "aligned"
    assert candidate["source_event_ids"] == [1017, 1018]


@pytest.mark.parametrize("grouped", [False, True])
def test_no_event_flood_from_256_turns_or_large_round(engine, monkeypatch, grouped):
    events = Events()
    if grouped:
        for event in events.rows:
            event["metadata"]["round_id"] = 1
        refs = [1]  # one round containing 256 raw events must not be truncated to "evidence"
    else:
        refs = list(range(1, 257))
    model(monkeypatch, engine, {"alignments": [{"candidate_index": 0, "source_turn_refs": refs}]})
    candidate = run(engine, events)["candidates"][0]
    assert candidate["provenance_status"] == "needs_repair"
    assert candidate["source_event_ids"] == candidate["source_turn_ids"] == []


def test_auto_still_rejects_alignment_failure(engine, monkeypatch):
    model(monkeypatch, engine, {"alignments": None})
    result = run(engine, Events(), mode="auto")
    assert result["status"] == "skipped" and result["reason"] == "no_candidates"
    assert engine._load_daily_chat_memory_pending() == []


def test_normalization_never_falls_back_to_window(engine):
    candidate = engine._normalize_daily_chat_memory_candidates(
        DATE, [CANDIDATE], [{"id": i, "raw_event_ids": [i + 1000]} for i in range(1, 257)],
        min_confidence=0, require_source_provenance=False,
    )[0]
    assert candidate["provenance_status"] == "needs_repair"
    assert candidate["source_event_ids"] == candidate["source_turn_ids"] == []


def test_repair_failure_then_success_keeps_identity_status_and_cursor(engine, monkeypatch):
    events = Events()
    model(monkeypatch, engine, {"alignments": None})
    candidate_id = run(engine, events)["candidates"][0]["id"]
    cursor = engine._load_daily_chat_memory_cursor()
    blocked = asyncio.run(engine.confirm_daily_chat_memory([candidate_id], ReadOnlyBuckets()))
    assert blocked["results"][0]["reason"] == "provenance_needs_repair"
    failed = asyncio.run(engine.repair_daily_chat_memory_provenance([candidate_id], raw_event_store=events))
    assert failed["needs_repair"] == 1
    # Newly ingested events are outside the saved extraction scope, even on the same day.
    events.rows.append({**events.rows[0], "id": 9999, "text": "must not enter repair input"})
    calls = model(monkeypatch, engine, {"alignments": [{"candidate_index": 0, "source_turn_refs": [17, 18]}]})
    repaired = asyncio.run(engine.repair_daily_chat_memory_provenance([candidate_id], raw_event_store=events))
    assert repaired["aligned"] == 1
    assert len(calls[0]["conversation_turns"]) == 256
    row = engine.list_daily_chat_memory_pending()[0]
    assert row["id"] == candidate_id and row["status"] == "pending"
    assert row["candidate"]["source_event_ids"] == [1017, 1018]
    assert "provenance_detail" not in row["candidate"]
    assert engine._load_daily_chat_memory_cursor() == cursor


def test_repair_aligned_candidate_clears_stale_diagnostic_without_realign(engine, monkeypatch):
    events = Events()
    model(monkeypatch, engine, {"alignments": [{"candidate_index": 0, "source_turn_refs": [17]}]})
    candidate_id = run(engine, events)["candidates"][0]["id"]
    items = engine._load_daily_chat_memory_pending()
    items[0]["candidate"]["provenance_detail"] = "empty_response"
    engine._save_daily_chat_memory_pending(items)

    async def should_not_realign(*args, **kwargs):
        raise AssertionError("already aligned candidate should not re-align just to clear stale diagnostics")

    monkeypatch.setattr(engine, "_align_daily_chat_memory_candidate_sources", should_not_realign)
    result = asyncio.run(engine.repair_daily_chat_memory_provenance([candidate_id], raw_event_store=events))
    assert result["aligned"] == 1
    row = engine.list_daily_chat_memory_pending()[0]
    assert row["candidate"]["provenance_status"] == "aligned"
    assert row["candidate"]["source_event_ids"] == [1017]
    assert "provenance_detail" not in row["candidate"]


def test_repair_does_not_overwrite_concurrent_rejection(engine, monkeypatch):
    model(monkeypatch, engine, {})
    events = Events()
    candidate_id = run(engine, events)["candidates"][0]["id"]

    async def align(key, candidates, turns):
        asyncio_result = await engine.confirm_daily_chat_memory([candidate_id], ReadOnlyBuckets(), action="reject")
        assert asyncio_result["rejected"] == 1
        return [{**candidates[0], "provenance_status": "aligned", "source_event_ids": [1017]}]

    monkeypatch.setattr(engine, "_align_daily_chat_memory_candidate_sources", align)
    result = asyncio.run(engine.repair_daily_chat_memory_provenance([candidate_id], raw_event_store=events))
    assert result["results"][0]["reason"] == "candidate_changed"
    assert engine._load_daily_chat_memory_pending()[0]["status"] == "rejected"


def test_repair_rejects_wrong_profile_and_missing_input(engine, monkeypatch):
    model(monkeypatch, engine, {})
    events = Events()
    candidate_id = run(engine, events)["candidates"][0]["id"]
    result = asyncio.run(engine.repair_daily_chat_memory_provenance(
        [candidate_id], raw_event_store=events, persona_engine=SimpleNamespace(profile_id="other"),
    ))
    assert result["results"][0]["reason"] == "repair_scope_unavailable"
    events.rows.pop()
    result = asyncio.run(engine.repair_daily_chat_memory_provenance([candidate_id], raw_event_store=events))
    assert result["results"][0]["reason"] == "repair_input_unavailable"
    assert len(engine.list_daily_chat_memory_pending()) == 1


def test_edit_invalidates_aligned_evidence(engine, monkeypatch):
    model(monkeypatch, engine, {"alignments": [{"candidate_index": 0, "source_turn_refs": [1]}]})
    candidate_id = run(engine, Events())["candidates"][0]["id"]
    result = asyncio.run(engine.confirm_daily_chat_memory(
        [candidate_id], ReadOnlyBuckets(), edits={candidate_id: {"content": "用户现在明确偏好在工作时播放背景音乐。"}},
    ))
    assert result["results"][0]["status"] == "blocked"
    candidate = engine.list_daily_chat_memory_pending()[0]["candidate"]
    assert candidate["provenance_status"] == "needs_repair"
    assert candidate["source_event_ids"] == []


def api_client(engine, events):
    # Execute the real route bodies without starting server.py's live services.
    source = Path(__file__).resolve().parents[1] / "server.py"
    names = {"api_daily_chat_memory_pending", "api_daily_chat_memory_repair"}
    functions = [node for node in ast.parse(source.read_text(encoding="utf-8")).body
                 if isinstance(node, ast.AsyncFunctionDef) and node.name in names]
    for function in functions:
        function.decorator_list = []
    namespace = {
        "reflection_engine": engine, "raw_event_store": events,
        "gateway_state_store": object(), "persona_engine": None,
        "_int_between": lambda value, default, low, high: max(low, min(high, int(value or default))),
        "_require_dashboard_auth": lambda request: None if request.headers.get("authorization") == "test"
        else JSONResponse({"error": "unauthorized"}, status_code=401),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return TestClient(Starlette(routes=[
        Route("/api/daily-chat-memory/pending", namespace["api_daily_chat_memory_pending"], methods=["GET"]),
        Route("/api/daily-chat-memory/repair", namespace["api_daily_chat_memory_repair"], methods=["POST"]),
    ]))


def test_pending_api_returns_needs_repair_and_repair_api_keeps_pending(engine, monkeypatch):
    model(monkeypatch, engine, {})
    events = Events()
    candidate_id = run(engine, events)["candidates"][0]["id"]
    with api_client(engine, events) as client:
        assert client.get("/api/daily-chat-memory/pending").status_code == 401
        response = client.get("/api/daily-chat-memory/pending", headers={"authorization": "test"})
        assert response.status_code == 200
        assert response.json()["items"][0]["candidate"]["provenance_status"] == "needs_repair"
        assert client.post("/api/daily-chat-memory/repair", json={"candidate_ids": [candidate_id]}).status_code == 401
        model(monkeypatch, engine, {"alignments": [{"candidate_index": 0, "source_turn_refs": [2]}]})
        response = client.post("/api/daily-chat-memory/repair", json={"candidate_ids": [candidate_id]}, headers={"authorization": "test"})
        assert response.status_code == 200 and response.json()["aligned"] == 1
        row = client.get("/api/daily-chat-memory/pending", headers={"authorization": "test"}).json()["items"][0]
        assert row["status"] == "pending" and row["candidate"]["source_event_ids"] == [1002]
