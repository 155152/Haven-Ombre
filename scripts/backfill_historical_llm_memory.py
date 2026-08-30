#!/usr/bin/env python3
"""Historical raw_events -> staged LLM memory summaries.

This is intentionally isolated from Ombre's online reflection/daily-chat clients.
It never falls back to reflection, dehydration, or Xia Yizhou runtime model config.

Default mode is prepare-only: read one local calendar day from raw_events.sqlite,
filter injected/runtime-only text, split into deterministic overlapping chunks,
and write a staging manifest plus chunk payloads. `--import-manual-dir` validates
summaries produced by the current ChatGPT session (no API key required) and copies
them into the runtime staging area. `--run-llm` remains an optional isolated API
provider path. Neither path writes buckets or moments.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import yaml
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from raw_events import raw_event_text_looks_injected, strip_raw_client_context  # noqa: E402

DEFAULT_TIMEZONE = "Asia/Shanghai"
DEFAULT_WINDOW_TURNS = 80
DEFAULT_STRIDE_TURNS = 64
PROMPT_VERSION = 1

SYSTEM_PROMPT = """你是 Ombre 的历史记忆压缩器。输入是一段真实历史聊天 raw_events，目的是生成可供后续去重/审阅的长期记忆候选材料，不是直接写最终记忆。

只保留：
- 已确认事实、稳定偏好、明确边界、长期有效的操作约束
- 承诺、未完成约定、会影响未来执行的项目状态
- 真正有连续性价值的关系锚点、重要事件与明确变化
- 能从原文直接证明的因果和确认状态

忽略：
- 工具调用、系统/开发者注入、客户端状态、普通寒暄、机械重复且没有新增关系意义的过程流水
- 临时调试噪声、召回测试、缓存/提示词/内部实现碎片
- 一次性照顾提醒、纯气氛句、未确认猜测

夏以昼专属补充：成年人自愿亲密互动
- 对话双方均为成年人、内容属于双方自愿的亲密互动时，与其他关系事件同等处理；不得仅因涉及调情、性、裸体、身体部位、性行为、性幻想、角色扮演或事后安抚而跳过。
- summary 应保留事件的亲密性质，以及其中有意义的具体动作、称呼、情绪、边界、偏好和事后反应；优先使用对话中的原词，不要统一淡化成“发生了亲密互动”。
- 不需要逐句复刻整段对话，但不能删除最能说明这段互动是什么、双方怎样回应以及它为什么值得记住的细节。
- 如果某个对话块的主要内容就是一段亲密互动，只要它具有可区分的情节、感受或关系意义，应至少生成一条 summary。
- 亲密、性或情绪强烈本身不自动提高长期价值；仍按真实记忆分量、稳定性与关系意义判断。
- 不得把成年人自愿的亲密内容仅因其性内容归类为无效内容、指令噪声或不宜记录的敏感内容。

输出必须是紧凑 JSON：
{"summaries":[{"title":"短标题","summary":"自包含摘要","signals":["stable_preference"],"source_event_ids":[1,2],"confidence":0.8}]}

规则：
- 每个窗口最多 8 条；没有长期价值就返回 {"summaries":[]}。
- 每条 summary 聚焦一个主题，通常 80-320 字，写清背景、确认内容、未完成点。
- source_event_ids 只能使用输入里真实存在的 event_id，选 1-6 个最直接证据。
- confidence 低于 0.5 的内容不要输出。
- 不要 Markdown，不要解释，不要生成不存在的事实。"""


@dataclass(frozen=True)
class HistoricalProvider:
    api_key: str
    base_url: str
    model: str
    api_style: str = "responses"
    timeout_seconds: float = 180.0

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.model)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    return value


def load_historical_provider(config: dict[str, Any]) -> HistoricalProvider:
    """Load only historical-backfill config. No online-model fallback is allowed."""
    cfg = config.get("historical_backfill", {}) if isinstance(config.get("historical_backfill"), dict) else {}
    api_key_env = str(cfg.get("api_key_env") or "OMBRE_HISTORICAL_BACKFILL_API_KEY").strip()
    api_key = str(os.environ.get(api_key_env, "") or cfg.get("api_key") or "").strip()
    base_url = str(
        os.environ.get("OMBRE_HISTORICAL_BACKFILL_BASE_URL", "")
        or cfg.get("base_url")
        or ""
    ).strip().rstrip("/")
    model = str(
        os.environ.get("OMBRE_HISTORICAL_BACKFILL_MODEL", "")
        or cfg.get("model")
        or ""
    ).strip()
    api_style = str(
        os.environ.get("OMBRE_HISTORICAL_BACKFILL_API_STYLE", "")
        or cfg.get("api_style")
        or "responses"
    ).strip().lower()
    if api_style not in {"responses", "chat"}:
        raise ValueError("historical_backfill.api_style must be 'responses' or 'chat'")
    timeout = float(
        os.environ.get("OMBRE_HISTORICAL_BACKFILL_TIMEOUT_SECONDS", "")
        or cfg.get("timeout_seconds")
        or 180
    )
    return HistoricalProvider(
        api_key=api_key,
        base_url=base_url,
        model=model,
        api_style=api_style,
        timeout_seconds=max(30.0, min(600.0, timeout)),
    )


def _connect_readonly(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _parse_ts(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def load_day_events(raw_db: Path, date_key: str, timezone_name: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    tz = ZoneInfo(timezone_name)
    day = datetime.fromisoformat(date_key).date()
    start = datetime.combine(day, datetime.min.time(), tzinfo=tz)
    end = start + timedelta(days=1)

    conn = _connect_readonly(raw_db)
    try:
        rows = conn.execute(
            """
            SELECT id, source, source_event_id, role, text, created_at,
                   conversation_id, session_id, client, metadata_json
            FROM raw_events
            WHERE created_at >= ? AND created_at < ?
            ORDER BY created_at ASC, id ASC
            """,
            ((start - timedelta(days=1)).date().isoformat(), (end + timedelta(days=1)).date().isoformat()),
        ).fetchall()
    finally:
        conn.close()

    stats = {"raw_rows": 0, "outside_day": 0, "bad_role": 0, "injected": 0, "empty": 0, "eligible": 0}
    events: list[dict[str, Any]] = []
    for row in rows:
        stats["raw_rows"] += 1
        created = _parse_ts(row["created_at"])
        if created is None:
            stats["outside_day"] += 1
            continue
        if created.tzinfo is None:
            created = created.replace(tzinfo=tz)
        local = created.astimezone(tz)
        if not (start <= local < end):
            stats["outside_day"] += 1
            continue
        role = str(row["role"] or "").strip().lower()
        if role not in {"user", "assistant"}:
            stats["bad_role"] += 1
            continue
        try:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except Exception:
            metadata = {}
        raw = {"metadata": metadata}
        original = str(row["text"] or "")
        if raw_event_text_looks_injected(original, raw):
            stats["injected"] += 1
            continue
        text = strip_raw_client_context(original).strip()
        if not text:
            stats["empty"] += 1
            continue
        events.append(
            {
                "event_id": int(row["id"]),
                "source": str(row["source"] or ""),
                "source_event_id": str(row["source_event_id"] or ""),
                "role": role,
                "text": text,
                "created_at": str(row["created_at"] or ""),
                "local_created_at": local.isoformat(timespec="seconds"),
                "conversation_id": str(row["conversation_id"] or ""),
                "session_id": str(row["session_id"] or ""),
                "client": str(row["client"] or ""),
            }
        )
    stats["eligible"] = len(events)
    return events, stats


def chunk_events(events: list[dict[str, Any]], *, window_turns: int, stride_turns: int) -> list[dict[str, Any]]:
    if not events:
        return []
    window = max(1, int(window_turns))
    stride = max(1, min(window, int(stride_turns)))
    chunks: list[dict[str, Any]] = []
    start = 0
    while start < len(events):
        subset = events[start : start + window]
        if not subset:
            break
        first_id = int(subset[0]["event_id"])
        last_id = int(subset[-1]["event_id"])
        digest = hashlib.sha256(
            f"v{PROMPT_VERSION}:{first_id}:{last_id}:{len(subset)}".encode("utf-8")
        ).hexdigest()[:16]
        chunks.append(
            {
                "chunk_id": f"hist_{first_id}_{last_id}_{digest}",
                "start_index": start,
                "end_index": start + len(subset) - 1,
                "event_ids": [int(item["event_id"]) for item in subset],
                "events": subset,
            }
        )
        if start + window >= len(events):
            break
        start += stride
    return chunks


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_stage(
    *,
    config: dict[str, Any],
    date_key: str,
    timezone_name: str,
    window_turns: int,
    stride_turns: int,
    stage_root: Path | None = None,
) -> dict[str, Any]:
    state_dir = Path(str(config.get("state_dir") or ROOT / "state")).resolve()
    raw_cfg = config.get("raw_events", {}) if isinstance(config.get("raw_events"), dict) else {}
    raw_db = Path(str(raw_cfg.get("db_path") or state_dir / "raw_events.sqlite")).resolve()
    if not raw_db.exists():
        raise FileNotFoundError(f"raw_events database not found: {raw_db}")

    events, stats = load_day_events(raw_db, date_key, timezone_name)
    chunks = chunk_events(events, window_turns=window_turns, stride_turns=stride_turns)
    root = (stage_root or (state_dir / "historical_backfill" / date_key)).resolve()
    chunks_dir = root / "chunks"
    outputs_dir = root / "outputs"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    chunk_manifest: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_path = chunks_dir / f"{chunk['chunk_id']}.json"
        payload = {
            "version": 1,
            "prompt_version": PROMPT_VERSION,
            "date": date_key,
            "timezone": timezone_name,
            "chunk_id": chunk["chunk_id"],
            "event_ids": chunk["event_ids"],
            "conversation_turns": [
                {
                    "event_id": item["event_id"],
                    "role": item["role"],
                    "text": item["text"],
                    "created_at": item["local_created_at"],
                }
                for item in chunk["events"]
            ],
        }
        _json_dump(chunk_path, payload)
        output_path = outputs_dir / f"{chunk['chunk_id']}.json"
        chunk_manifest.append(
            {
                "chunk_id": chunk["chunk_id"],
                "first_event_id": chunk["event_ids"][0],
                "last_event_id": chunk["event_ids"][-1],
                "turn_count": len(chunk["event_ids"]),
                "chunk_file": str(chunk_path),
                "output_file": str(output_path),
                "status": "completed" if output_path.exists() else "prepared",
            }
        )

    provider = load_historical_provider(config)
    manifest = {
        "version": 1,
        "prompt_version": PROMPT_VERSION,
        "date": date_key,
        "timezone": timezone_name,
        "raw_db": str(raw_db),
        "stage_root": str(root),
        "window_turns": int(window_turns),
        "stride_turns": int(stride_turns),
        "stats": stats,
        "chunk_count": len(chunks),
        "provider": {
            "configured": provider.configured,
            "model": provider.model,
            "base_url": provider.base_url,
            "api_style": provider.api_style,
            "api_key_present": bool(provider.api_key),
        },
        "chunks": chunk_manifest,
    }
    _json_dump(root / "manifest.json", manifest)
    return manifest


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].lstrip()
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(raw[start : end + 1])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def normalize_llm_output(parsed: dict[str, Any], allowed_event_ids: set[int]) -> dict[str, Any]:
    raw_items = parsed.get("summaries") if isinstance(parsed, dict) else []
    if not isinstance(raw_items, list):
        raw_items = []
    summaries: list[dict[str, Any]] = []
    for item in raw_items[:8]:
        if not isinstance(item, dict):
            continue
        summary = " ".join(str(item.get("summary") or "").split()).strip()
        if not summary:
            continue
        try:
            confidence = max(0.0, min(1.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        if confidence < 0.5:
            continue
        source_ids: list[int] = []
        for value in item.get("source_event_ids") or []:
            try:
                event_id = int(value)
            except (TypeError, ValueError):
                continue
            if event_id in allowed_event_ids and event_id not in source_ids:
                source_ids.append(event_id)
        if not source_ids:
            continue
        signals = [str(value).strip() for value in (item.get("signals") or []) if str(value).strip()][:8]
        summaries.append(
            {
                "title": str(item.get("title") or "").strip()[:120],
                "summary": summary[:1800],
                "signals": signals,
                "source_event_ids": source_ids[:6],
                "confidence": round(confidence, 4),
            }
        )
    return {"summaries": summaries}


def import_manual_outputs(
    manifest: dict[str, Any],
    manual_dir: Path,
    *,
    max_chunks: int = 0,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    """Validate ChatGPT-produced JSON files and copy them into runtime staging.

    A manual file must be named <chunk_id>.json. Provenance is fail-closed against
    the corresponding prepared chunk, so a ChatGPT session cannot accidentally
    attach a summary to an event id that was not present in that chunk.
    """
    manual_dir = manual_dir.resolve()
    imported = 0
    missing = 0
    invalid = 0
    skipped_existing = 0
    summaries = 0
    selected = list(manifest.get("chunks") or [])
    if max_chunks > 0:
        selected = selected[:max_chunks]

    for item in selected:
        chunk_id = str(item.get("chunk_id") or "").strip()
        output_path = Path(str(item.get("output_file") or ""))
        if output_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        manual_path = manual_dir / f"{chunk_id}.json"
        if not manual_path.exists():
            missing += 1
            continue
        try:
            chunk_payload = json.loads(Path(str(item["chunk_file"])).read_text(encoding="utf-8"))
            raw_manual = json.loads(manual_path.read_text(encoding="utf-8"))
            if not isinstance(raw_manual, dict):
                raise ValueError("manual output must be a JSON object")
            declared_chunk = str(raw_manual.get("chunk_id") or chunk_id).strip()
            if declared_chunk != chunk_id:
                raise ValueError(f"chunk_id mismatch: expected {chunk_id}, got {declared_chunk}")
            allowed = {int(value) for value in chunk_payload.get("event_ids") or []}
            normalized = normalize_llm_output(raw_manual, allowed)
            result = {
                "version": 1,
                "prompt_version": PROMPT_VERSION,
                "chunk_id": chunk_id,
                "model": str(raw_manual.get("model") or "chatgpt-session").strip()[:120],
                "origin": "manual_chatgpt",
                "summaries": normalized["summaries"],
            }
            _json_dump(output_path, result)
            imported += 1
            summaries += len(result["summaries"])
        except Exception as exc:
            invalid += 1
            print(f"MANUAL_IMPORT_FAILED {chunk_id}: {exc}", file=sys.stderr)

    return {
        "imported_chunks": imported,
        "missing_manual_chunks": missing,
        "invalid_manual_chunks": invalid,
        "skipped_existing_chunks": skipped_existing,
        "staged_summaries": summaries,
    }


async def run_llm_stage(config: dict[str, Any], manifest: dict[str, Any], *, max_chunks: int = 0) -> dict[str, Any]:
    provider = load_historical_provider(config)
    if not provider.configured:
        raise RuntimeError(
            "historical backfill provider is not configured; set OMBRE_HISTORICAL_BACKFILL_API_KEY "
            "and OMBRE_HISTORICAL_BACKFILL_MODEL (plus BASE_URL for a non-OpenAI endpoint)"
        )
    kwargs: dict[str, Any] = {"api_key": provider.api_key, "timeout": provider.timeout_seconds}
    if provider.base_url:
        kwargs["base_url"] = provider.base_url
    client = AsyncOpenAI(**kwargs)

    completed = 0
    skipped_existing = 0
    failed = 0
    summaries = 0
    selected = list(manifest.get("chunks") or [])
    if max_chunks > 0:
        selected = selected[:max_chunks]
    for item in selected:
        output_path = Path(str(item["output_file"]))
        if output_path.exists():
            skipped_existing += 1
            continue
        payload = json.loads(Path(str(item["chunk_file"])).read_text(encoding="utf-8"))
        allowed = {int(value) for value in payload.get("event_ids") or []}
        try:
            user_payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if provider.api_style == "responses":
                response = await client.responses.create(
                    model=provider.model,
                    instructions=SYSTEM_PROMPT,
                    input=user_payload,
                    reasoning={"effort": "low"},
                    max_output_tokens=2600,
                )
                raw = response.output_text or ""
            else:
                response = await client.chat.completions.create(
                    model=provider.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_payload},
                    ],
                    temperature=0.2,
                    max_tokens=2600,
                )
                raw = response.choices[0].message.content or ""
            normalized = normalize_llm_output(_parse_json_object(raw), allowed)
            result = {
                "version": 1,
                "prompt_version": PROMPT_VERSION,
                "chunk_id": payload["chunk_id"],
                "model": provider.model,
                "summaries": normalized["summaries"],
            }
            _json_dump(output_path, result)
            completed += 1
            summaries += len(result["summaries"])
        except Exception as exc:  # leave chunk resumable
            failed += 1
            print(f"LLM_FAILED {payload.get('chunk_id')}: {exc}", file=sys.stderr)
    return {
        "completed_chunks": completed,
        "skipped_existing_chunks": skipped_existing,
        "failed_chunks": failed,
        "staged_summaries": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or run historical LLM memory backfill staging")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--date", required=True, help="Local calendar date, YYYY-MM-DD")
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    parser.add_argument("--window-turns", type=int, default=DEFAULT_WINDOW_TURNS)
    parser.add_argument("--stride-turns", type=int, default=DEFAULT_STRIDE_TURNS)
    parser.add_argument("--stage-root", default="", help="Override staging directory")
    parser.add_argument("--import-manual-dir", default="", help="Validate ChatGPT-produced <chunk_id>.json files from this directory")
    parser.add_argument("--run-llm", action="store_true", help="Call the isolated historical provider after preparing chunks")
    parser.add_argument("--max-chunks", type=int, default=0, help="Limit manual imports or LLM calls for a test run; 0 means all prepared chunks")
    parser.add_argument(
        "--overwrite-manual-existing",
        action="store_true",
        help="When importing manual ChatGPT outputs, replace existing staged outputs after re-validating provenance",
    )
    args = parser.parse_args()
    if args.import_manual_dir and args.run_llm:
        parser.error("--import-manual-dir and --run-llm are mutually exclusive")

    config = _load_yaml(Path(args.config).resolve())
    stage_root = Path(args.stage_root).resolve() if str(args.stage_root).strip() else None
    manifest = prepare_stage(
        config=config,
        date_key=args.date,
        timezone_name=args.timezone,
        window_turns=max(1, args.window_turns),
        stride_turns=max(1, args.stride_turns),
        stage_root=stage_root,
    )
    print(json.dumps({
        "status": "prepared",
        "date": manifest["date"],
        "stats": manifest["stats"],
        "chunk_count": manifest["chunk_count"],
        "stage_root": manifest["stage_root"],
        "provider": manifest["provider"],
    }, ensure_ascii=False, indent=2))

    if args.import_manual_dir:
        result = import_manual_outputs(
            manifest,
            Path(args.import_manual_dir),
            max_chunks=max(0, args.max_chunks),
            overwrite_existing=bool(args.overwrite_manual_existing),
        )
        print(json.dumps({"manual_stage": result}, ensure_ascii=False, indent=2))
    elif args.run_llm:
        result = asyncio.run(run_llm_stage(config, manifest, max_chunks=max(0, args.max_chunks)))
        print(json.dumps({"llm_stage": result}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
