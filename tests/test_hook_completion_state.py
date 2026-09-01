from pathlib import Path

from gateway_state import GatewayStateStore
from recall_policy import RecallPolicy


def test_hook_completion_advances_round_and_records_recent_context_once(tmp_path: Path):
    store = GatewayStateStore(str(tmp_path / "gateway_state.db"))

    first = store.record_hook_completion(
        "main_relation",
        "canonical-1",
        ["mem_a", "mem_a", "mem_b"],
        recent_context_injected=True,
    )
    replay = store.record_hook_completion(
        "main_relation",
        "canonical-1",
        ["mem_other"],
        recent_context_injected=False,
    )

    assert first == {"recorded": True, "round_id": 1}
    assert replay == {"recorded": False, "round_id": 1}
    assert store.get_current_round("main_relation") == 1
    assert store.get_recent_bucket_ids("main_relation", 5) == {"mem_a", "mem_b"}
    assert store.get_last_recent_context_at("main_relation") is not None


def test_hook_completion_advances_even_when_no_memory_was_injected(tmp_path: Path):
    store = GatewayStateStore(str(tmp_path / "gateway_state.db"))

    result = store.record_hook_completion(
        "main_relation",
        "canonical-empty",
        [],
        recent_context_injected=False,
    )

    assert result == {"recorded": True, "round_id": 1}
    assert store.get_current_round("main_relation") == 1
    assert store.get_recent_bucket_ids("main_relation", 5) == set()
    assert store.get_last_recent_context_at("main_relation") is None


def test_response_style_request_is_vague_but_named_topic_remains_searchable():
    policy = RecallPolicy()

    light = policy.plan_query("🫠老公 说点轻松的")
    named = policy.plan_query("老公说点 AWM 的进度")

    assert light.skip_long_term_recall is True
    assert light.skip_reason == "auto_vague_query"
    assert named.skip_long_term_recall is False
