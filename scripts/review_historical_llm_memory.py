#!/usr/bin/env python3
"""Cross-day review staging for historical LLM memory backfill.

This module only reads per-day historical_backfill outputs and writes a separate
cross-day review/merge staging area. It never writes Ombre Buckets or Moments.

The review stage is deliberately fail-closed: similarity is used only to group
possible related summaries for manual review. No summary is deleted, merged,
corrected, superseded, or promoted to a current rule automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parents[1]
REVIEW_VERSION = 1
DEFAULT_BATCH_SIZE = 28
DEFAULT_RELATED_LIMIT = 8

REVIEW_CLASSIFICATIONS = {
    "independent",
    "duplicate",
    "reinforcement",
    "evolution",
    "correction",
    "superseded",
    "historical/project-only",
}

# These are event-shaped memories whose concrete event must not disappear into
# an abstract rule/preferences summary. Exact duplicate summaries of the same
# event may still be consolidated during manual review.
PROTECTED_EVENT_SIGNALS = {
    "intimacy_event",
    "relationship_event",
    "relationship_milestone",
    "intimacy_milestone",
    "family_milestone",
    "qixi_event",
    "verified_milestone",
}

PROJECT_SIGNAL_MARKERS = (
    "project",
    "architecture",
    "bug",
    "technical",
    "runtime",
    "stage_",
    "historical_project",
    "module_",
    "tool_",
)

# Manual-review aid only. Protected event summaries from overlapping/adjacent
# windows on the same day may describe different slices of one concrete scene.
# Keeping them in one review component prevents a canonical scene from being
# accidentally expanded into several durable Moments. This never auto-merges.
PROTECTED_SCENE_EVENT_ID_GAP = 24

_SPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^0-9A-Za-z\u3400-\u9fff]+")


@dataclass(frozen=True)
class ReviewPaths:
    root: Path
    source_manifest: Path
    batches_dir: Path
    decisions_dir: Path
    merge_staging: Path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError("config must be a mapping")
    return value


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"invalid date: {value!r}") from exc


def _date_keys(start_date: str, end_date: str) -> list[str]:
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be on or after start_date")
    rows: list[str] = []
    current = start
    while current <= end:
        rows.append(current.isoformat())
        current += timedelta(days=1)
    return rows


def _normalize_text(value: Any) -> str:
    text = _SPACE_RE.sub(" ", str(value or "")).strip().lower()
    return _PUNCT_RE.sub("", text)


def _ngrams(value: Any, n: int) -> set[str]:
    text = _normalize_text(value)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[idx : idx + n] for idx in range(len(text) - n + 1)}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _weighted_signal_jaccard(left: set[str], right: set[str], weights: dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    denominator = sum(weights.get(value, 1.0) for value in union)
    if denominator <= 0:
        return 0.0
    numerator = sum(weights.get(value, 1.0) for value in (left & right))
    return numerator / denominator


def _summary_id(date_key: str, chunk_id: str, index: int, summary: dict[str, Any]) -> str:
    source_ids = ",".join(str(int(value)) for value in summary.get("source_event_ids") or [])
    digest_input = "\n".join(
        [
            date_key,
            chunk_id,
            str(index),
            str(summary.get("title") or ""),
            str(summary.get("summary") or ""),
            source_ids,
        ]
    )
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:20]
    return f"hs_{date_key.replace('-', '')}_{digest}"


def _is_protected_event(signals: Iterable[str]) -> bool:
    values = {str(value).strip() for value in signals if str(value).strip()}
    return bool(values & PROTECTED_EVENT_SIGNALS)


def _is_projectish(signals: Iterable[str]) -> bool:
    for value in signals:
        signal = str(value or "").strip().lower()
        if signal and any(marker in signal for marker in PROJECT_SIGNAL_MARKERS):
            return True
    return False


def _resolve_day_root(stage_parent: Path, date_key: str) -> Path:
    direct = stage_parent / date_key
    if direct.is_dir():
        return direct

    nested = sorted(path for path in stage_parent.glob(f"*/{date_key}") if path.is_dir())
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        rendered = ", ".join(str(path) for path in nested)
        raise ValueError(f"ambiguous staged date {date_key}: {rendered}")
    return direct


def collect_staged_summaries(stage_parent: Path, start_date: str, end_date: str) -> list[dict[str, Any]]:
    """Collect validated per-day staging summaries for an exact inclusive range."""
    stage_parent = stage_parent.resolve()
    records: list[dict[str, Any]] = []
    for date_key in _date_keys(start_date, end_date):
        day_root = _resolve_day_root(stage_parent, date_key)
        outputs_dir = day_root / "outputs"
        if not outputs_dir.exists():
            raise FileNotFoundError(f"missing outputs directory for {date_key}: {outputs_dir}")
        for output_path in sorted(outputs_dir.glob("*.json")):
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"output must be object: {output_path}")
            chunk_id = str(payload.get("chunk_id") or output_path.stem).strip()
            raw_summaries = payload.get("summaries") or []
            if not isinstance(raw_summaries, list):
                raise ValueError(f"summaries must be list: {output_path}")
            for index, raw in enumerate(raw_summaries, start=1):
                if not isinstance(raw, dict):
                    raise ValueError(f"summary must be object: {output_path}#{index}")
                text = " ".join(str(raw.get("summary") or "").split()).strip()
                if not text:
                    raise ValueError(f"empty summary: {output_path}#{index}")
                source_event_ids: list[int] = []
                for value in raw.get("source_event_ids") or []:
                    event_id = int(value)
                    if event_id not in source_event_ids:
                        source_event_ids.append(event_id)
                if not source_event_ids:
                    raise ValueError(f"missing provenance: {output_path}#{index}")
                signals = [str(value).strip() for value in (raw.get("signals") or []) if str(value).strip()]
                try:
                    relative_source = str(output_path.relative_to(stage_parent)).replace("\\", "/")
                except ValueError:
                    relative_source = str(output_path)
                summary = {
                    "title": str(raw.get("title") or "").strip(),
                    "summary": text,
                    "signals": signals,
                    "source_event_ids": source_event_ids,
                    "confidence": float(raw.get("confidence") or 0.0),
                }
                records.append(
                    {
                        "summary_id": _summary_id(date_key, chunk_id, index, summary),
                        "date": date_key,
                        "chunk_id": chunk_id,
                        "source_file": relative_source,
                        "source_index": index,
                        **summary,
                        "protected_event": _is_protected_event(signals),
                        "projectish": _is_projectish(signals),
                    }
                )
    ids = [row["summary_id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("summary_id collision detected")
    return records


def _signal_weights(records: list[dict[str, Any]]) -> dict[str, float]:
    total = max(1, len(records))
    document_frequency: Counter[str] = Counter()
    for row in records:
        document_frequency.update(set(row.get("signals") or []))
    return {
        signal: 1.0 + math.log((total + 1.0) / (count + 1.0))
        for signal, count in document_frequency.items()
    }


def score_related_pairs(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return similarity suggestions only; never automatic merge decisions."""
    signal_weights = _signal_weights(records)
    prepared: list[dict[str, Any]] = []
    for row in records:
        prepared.append(
            {
                "row": row,
                "signals": set(row.get("signals") or []),
                "title_grams": _ngrams(row.get("title"), 2),
                "summary_grams": _ngrams(row.get("summary"), 3),
                "source_ids": set(int(value) for value in row.get("source_event_ids") or []),
            }
        )

    pairs: list[dict[str, Any]] = []
    for left_index in range(len(prepared)):
        left = prepared[left_index]
        for right_index in range(left_index + 1, len(prepared)):
            right = prepared[right_index]
            source_overlap = _jaccard(
                {str(value) for value in left["source_ids"]},
                {str(value) for value in right["source_ids"]},
            )
            title_similarity = _jaccard(left["title_grams"], right["title_grams"])
            signal_similarity = _weighted_signal_jaccard(left["signals"], right["signals"], signal_weights)
            protected_scene_neighbor = _protected_scene_neighbors(left["row"], right["row"])

            # Expensive summary comparison is only useful when another signal says
            # the pair might be related. Protected same-day adjacent spans are also
            # reviewed together so one scene is not split into several Moments.
            if (
                source_overlap <= 0
                and title_similarity < 0.08
                and signal_similarity < 0.10
                and not protected_scene_neighbor
            ):
                continue
            summary_similarity = _jaccard(left["summary_grams"], right["summary_grams"])
            exact_title = bool(_normalize_text(left["row"].get("title"))) and (
                _normalize_text(left["row"].get("title")) == _normalize_text(right["row"].get("title"))
            )
            score = max(
                source_overlap,
                0.62 * summary_similarity + 0.23 * signal_similarity + 0.15 * title_similarity,
                0.82 if exact_title else 0.0,
                0.34 if protected_scene_neighbor else 0.0,
            )
            if source_overlap > 0 or exact_title or score >= 0.18:
                pairs.append(
                    {
                        "left_id": left["row"]["summary_id"],
                        "right_id": right["row"]["summary_id"],
                        "left_date": left["row"]["date"],
                        "right_date": right["row"]["date"],
                        "same_day": left["row"]["date"] == right["row"]["date"],
                        "score": round(score, 4),
                        "source_overlap": round(source_overlap, 4),
                        "title_similarity": round(title_similarity, 4),
                        "summary_similarity": round(summary_similarity, 4),
                        "signal_similarity": round(signal_similarity, 4),
                    }
                )
    pairs.sort(key=lambda row: (-row["score"], row["left_id"], row["right_id"]))
    return pairs


def _strong_components(records: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> list[list[str]]:
    parent = {row["summary_id"]: row["summary_id"] for row in records}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for pair in pairs:
        strong = (
            pair["source_overlap"] > 0
            or pair["score"] >= 0.34
            or (pair["score"] >= 0.26 and pair["signal_similarity"] >= 0.28)
        )
        if strong:
            union(pair["left_id"], pair["right_id"])

    groups: dict[str, list[str]] = defaultdict(list)
    for summary_id in parent:
        groups[find(summary_id)].append(summary_id)
    ordered = [sorted(values) for values in groups.values()]
    ordered.sort(key=lambda values: (-len(values), values[0]))
    return ordered


def build_review_batches(
    records: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    related_limit: int = DEFAULT_RELATED_LIMIT,
) -> list[dict[str, Any]]:
    """Build review batches without splitting strong similarity components."""
    batch_size = max(1, int(batch_size))
    related_limit = max(1, int(related_limit))
    record_by_id = {row["summary_id"]: row for row in records}
    related: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        related[pair["left_id"]].append({"summary_id": pair["right_id"], "score": pair["score"]})
        related[pair["right_id"]].append({"summary_id": pair["left_id"], "score": pair["score"]})
    for summary_id in related:
        related[summary_id].sort(key=lambda row: (-row["score"], row["summary_id"]))

    components = _strong_components(records, pairs)
    components.sort(key=lambda ids: (min(record_by_id[value]["date"] for value in ids), -len(ids), ids[0]))

    batches: list[list[str]] = []
    current: list[str] = []
    for component in components:
        if current and len(current) + len(component) > batch_size:
            batches.append(current)
            current = []
        current.extend(component)
        if len(current) >= batch_size:
            batches.append(current)
            current = []
    if current:
        batches.append(current)

    result: list[dict[str, Any]] = []
    for batch_index, ids in enumerate(batches, start=1):
        rows: list[dict[str, Any]] = []
        for summary_id in ids:
            source = record_by_id[summary_id]
            rows.append(
                {
                    **source,
                    "related": related.get(summary_id, [])[:related_limit],
                }
            )
        rows.sort(key=lambda row: (row["date"], row["source_file"], row["source_index"]))
        result.append(
            {
                "version": REVIEW_VERSION,
                "batch_id": f"review_{batch_index:03d}",
                "item_count": len(rows),
                "items": rows,
            }
        )
    return result


def review_paths(stage_parent: Path, start_date: str, end_date: str, review_root: Path | None = None) -> ReviewPaths:
    if review_root is None:
        review_root = stage_parent / "cross_day_review" / f"{start_date}_to_{end_date}"
    root = review_root.resolve()
    return ReviewPaths(
        root=root,
        source_manifest=root / "source_manifest.json",
        batches_dir=root / "review_batches",
        decisions_dir=root / "decisions",
        merge_staging=root / "merge_staging.json",
    )


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def prepare_cross_day_review(
    *,
    stage_parent: Path,
    start_date: str,
    end_date: str,
    review_root: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    records = collect_staged_summaries(stage_parent, start_date, end_date)
    pairs = score_related_pairs(records)
    batches = build_review_batches(records, pairs, batch_size=batch_size)
    paths = review_paths(stage_parent, start_date, end_date, review_root)
    paths.batches_dir.mkdir(parents=True, exist_ok=True)
    paths.decisions_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": REVIEW_VERSION,
        "start_date": start_date,
        "end_date": end_date,
        "source_summary_count": len(records),
        "related_pair_count": len(pairs),
        "batch_count": len(batches),
        "protected_event_count": sum(1 for row in records if row["protected_event"]),
        "projectish_count": sum(1 for row in records if row["projectish"]),
        "review_root": str(paths.root),
        "formal_memory_write": False,
        "classification_values": sorted(REVIEW_CLASSIFICATIONS),
        "rules": {
            "later_evidence_priority": True,
            "protected_events_must_remain_concrete": True,
            "project_state_is_not_current_rule_by_default": True,
            "similarity_is_suggestion_only": True,
        },
        "records": records,
        "related_pairs": pairs,
    }
    _json_dump(paths.source_manifest, manifest)
    for batch in batches:
        _json_dump(paths.batches_dir / f"{batch['batch_id']}.json", batch)
    return manifest


def _load_review_batches(paths: ReviewPaths) -> list[dict[str, Any]]:
    if not paths.source_manifest.exists():
        raise FileNotFoundError(f"cross-day source manifest not found: {paths.source_manifest}")
    batches: list[dict[str, Any]] = []
    for path in sorted(paths.batches_dir.glob("review_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"review batch must be object: {path}")
        batches.append(payload)
    if not batches:
        raise FileNotFoundError(f"no review batches found: {paths.batches_dir}")
    return batches


def _event_span(row: dict[str, Any]) -> tuple[int, int] | None:
    event_ids = sorted({int(value) for value in row.get("source_event_ids") or []})
    if not event_ids:
        return None
    return event_ids[0], event_ids[-1]


def _protected_scene_neighbors(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if not left.get("protected_event") or not right.get("protected_event"):
        return False
    if str(left.get("date") or "") != str(right.get("date") or ""):
        return False
    left_events = {int(value) for value in left.get("source_event_ids") or []}
    right_events = {int(value) for value in right.get("source_event_ids") or []}
    if left_events & right_events:
        return True
    left_span = _event_span(left)
    right_span = _event_span(right)
    if not left_span or not right_span:
        return False
    if left_span[1] < right_span[0]:
        gap = right_span[0] - left_span[1]
    elif right_span[1] < left_span[0]:
        gap = left_span[0] - right_span[1]
    else:
        gap = 0
    return gap <= PROTECTED_SCENE_EVENT_ID_GAP


def _protected_duplicate_is_safe(items: list[dict[str, Any]], canonical_id: str) -> bool:
    protected = [row for row in items if row.get("protected_event")]
    if not protected:
        return True
    canonical = next(row for row in protected if row["summary_id"] == canonical_id)
    reachable = {canonical["summary_id"]}
    changed = True
    while changed:
        changed = False
        for left in protected:
            if left["summary_id"] not in reachable:
                continue
            for right in protected:
                if right["summary_id"] in reachable:
                    continue
                if _protected_scene_neighbors(left, right):
                    reachable.add(right["summary_id"])
                    changed = True
    return len(reachable) == len(protected)


def normalize_manual_review_batch(batch: dict[str, Any], raw_manual: dict[str, Any]) -> dict[str, Any]:
    batch_id = str(batch.get("batch_id") or "").strip()
    declared_batch = str(raw_manual.get("batch_id") or batch_id).strip()
    if declared_batch != batch_id:
        raise ValueError(f"batch_id mismatch: expected {batch_id}, got {declared_batch}")
    batch_items = list(batch.get("items") or [])
    item_by_id = {str(row.get("summary_id") or ""): row for row in batch_items}
    expected_ids = set(item_by_id)
    if "" in expected_ids:
        raise ValueError(f"batch contains empty summary_id: {batch_id}")

    raw_decisions = raw_manual.get("decisions") or []
    if not isinstance(raw_decisions, list):
        raise ValueError("decisions must be a list")
    normalized: list[dict[str, Any]] = []
    covered: set[str] = set()

    for index, decision in enumerate(raw_decisions, start=1):
        if not isinstance(decision, dict):
            raise ValueError(f"decision #{index} must be an object")
        classification = str(decision.get("classification") or "").strip()
        if classification not in REVIEW_CLASSIFICATIONS:
            raise ValueError(f"invalid classification in decision #{index}: {classification!r}")
        summary_ids = [str(value).strip() for value in (decision.get("summary_ids") or []) if str(value).strip()]
        if not summary_ids:
            raise ValueError(f"decision #{index} has no summary_ids")
        if len(summary_ids) != len(set(summary_ids)):
            raise ValueError(f"decision #{index} repeats a summary_id")
        unknown = set(summary_ids) - expected_ids
        if unknown:
            raise ValueError(f"decision #{index} references unknown ids: {sorted(unknown)}")
        overlap = set(summary_ids) & covered
        if overlap:
            raise ValueError(f"summary ids reviewed more than once: {sorted(overlap)}")
        covered.update(summary_ids)
        items = [item_by_id[value] for value in summary_ids]

        canonical_id = str(decision.get("canonical_id") or "").strip()
        if classification == "independent":
            if len(summary_ids) != 1:
                raise ValueError("independent decision must contain exactly one summary")
            canonical_id = canonical_id or summary_ids[0]
        elif classification in {"duplicate", "reinforcement", "correction", "superseded"}:
            if not canonical_id:
                raise ValueError(f"{classification} decision requires canonical_id")
        elif canonical_id and canonical_id not in summary_ids:
            raise ValueError("canonical_id must be one of summary_ids")
        if canonical_id and canonical_id not in summary_ids:
            raise ValueError("canonical_id must be one of summary_ids")

        if classification in {"correction", "superseded"}:
            latest_date = max(str(row.get("date") or "") for row in items)
            canonical_date = str(item_by_id[canonical_id].get("date") or "")
            if canonical_date != latest_date:
                raise ValueError(
                    f"{classification} canonical must use latest-date evidence: {canonical_date} != {latest_date}"
                )

        if classification == "duplicate" and any(row.get("protected_event") for row in items):
            if not canonical_id or not item_by_id[canonical_id].get("protected_event"):
                raise ValueError("protected event duplicate must keep a protected event as canonical")
            if not _protected_duplicate_is_safe(items, canonical_id):
                raise ValueError(
                    "protected event summaries without shared provenance cannot be collapsed as duplicate; "
                    "use reinforcement/evolution and preserve the concrete events"
                )

        explicit_retain = [str(value).strip() for value in (decision.get("retain_ids") or []) if str(value).strip()]
        if explicit_retain:
            if len(explicit_retain) != len(set(explicit_retain)):
                raise ValueError(f"decision #{index} repeats retain_ids")
            if set(explicit_retain) - set(summary_ids):
                raise ValueError(f"decision #{index} retain_ids must be within summary_ids")
            retain_ids = explicit_retain
        elif classification in {"duplicate", "reinforcement", "correction", "superseded"}:
            retain_ids = [canonical_id]
        else:
            retain_ids = list(summary_ids)

        # Concrete relationship/intimacy events may reinforce or evolve a rule,
        # but the event itself must survive unless this is a provenance-overlap
        # duplicate of the same event.
        if classification != "duplicate":
            for row in items:
                if row.get("protected_event") and row["summary_id"] not in retain_ids:
                    retain_ids.append(row["summary_id"])
        if canonical_id and canonical_id not in retain_ids:
            retain_ids.insert(0, canonical_id)

        scope = str(decision.get("scope") or "").strip()
        if classification == "historical/project-only":
            scope = "historical_project"
        elif not scope:
            if any(row.get("projectish") for row in items):
                raise ValueError(
                    f"decision #{index} includes projectish summaries; set scope to current_candidate or historical_project"
                )
            scope = "event_candidate" if any(row.get("protected_event") for row in items) else "current_candidate"
        if scope not in {"current_candidate", "historical_project", "event_candidate"}:
            raise ValueError(f"invalid scope in decision #{index}: {scope!r}")
        if any(row.get("protected_event") for row in items) and scope == "historical_project" and not any(
            row.get("projectish") for row in items
        ):
            raise ValueError("a protected relationship/intimacy event cannot be hidden as project-only history")

        merged_summary = " ".join(str(decision.get("merged_summary") or "").split()).strip()
        if len(merged_summary) > 2400:
            raise ValueError("merged_summary is too long")
        normalized.append(
            {
                "classification": classification,
                "summary_ids": summary_ids,
                "canonical_id": canonical_id,
                "retain_ids": retain_ids,
                "scope": scope,
                "merged_summary": merged_summary,
                "notes": " ".join(str(decision.get("notes") or "").split()).strip()[:1200],
            }
        )

    missing = expected_ids - covered
    if missing:
        raise ValueError(f"manual batch is incomplete; missing {len(missing)} summary ids")
    return {
        "version": REVIEW_VERSION,
        "batch_id": batch_id,
        "origin": "manual_chatgpt_cross_day_review",
        "decisions": normalized,
    }


def build_merge_staging(manifest: dict[str, Any], reviewed_batches: list[dict[str, Any]]) -> dict[str, Any]:
    record_by_id = {str(row.get("summary_id") or ""): row for row in (manifest.get("records") or [])}
    reviewed_ids: set[str] = set()
    decision_groups: list[dict[str, Any]] = []
    retained_candidates: list[dict[str, Any]] = []

    for reviewed in reviewed_batches:
        batch_id = str(reviewed.get("batch_id") or "")
        for group_index, decision in enumerate(reviewed.get("decisions") or [], start=1):
            source_ids = list(decision.get("summary_ids") or [])
            reviewed_ids.update(source_ids)
            group_id = f"{batch_id}_g{group_index:02d}"
            decision_groups.append({"group_id": group_id, **decision})
            canonical_id = str(decision.get("canonical_id") or "")
            retain_ids = list(decision.get("retain_ids") or [])
            collapse_to_canonical = bool(canonical_id and len(source_ids) > 1 and retain_ids == [canonical_id])
            for retain_id in retain_ids:
                source = record_by_id[retain_id]
                is_canonical = bool(canonical_id and retain_id == canonical_id)
                merged_ids = source_ids if is_canonical and collapse_to_canonical else [retain_id]
                merged_rows = [record_by_id[value] for value in merged_ids]
                event_ids: list[int] = []
                signals: list[str] = []
                evidence_dates: list[str] = []
                for row in merged_rows:
                    if row["date"] not in evidence_dates:
                        evidence_dates.append(row["date"])
                    for value in row.get("source_event_ids") or []:
                        event_id = int(value)
                        if event_id not in event_ids:
                            event_ids.append(event_id)
                    for value in row.get("signals") or []:
                        signal = str(value)
                        if signal and signal not in signals:
                            signals.append(signal)
                retained_candidates.append(
                    {
                        "merge_id": f"hm_{hashlib.sha256((group_id + ':' + retain_id).encode('utf-8')).hexdigest()[:20]}",
                        "group_id": group_id,
                        "classification": decision["classification"],
                        "scope": decision["scope"],
                        "source_summary_id": retain_id,
                        "canonical_source_id": canonical_id,
                        "title": source.get("title") or "",
                        "summary": decision.get("merged_summary") if is_canonical and decision.get("merged_summary") else source.get("summary") or "",
                        "signals": signals if is_canonical else list(source.get("signals") or []),
                        "source_summary_ids": merged_ids,
                        "source_event_ids": event_ids if is_canonical else list(source.get("source_event_ids") or []),
                        "evidence_dates": sorted(evidence_dates) if is_canonical else [source.get("date")],
                        "protected_event": bool(source.get("protected_event")),
                        "projectish": any(row.get("projectish") for row in merged_rows) if is_canonical else bool(source.get("projectish")),
                    }
                )

    all_ids = set(record_by_id)
    pending_ids = sorted(all_ids - reviewed_ids, key=lambda value: (record_by_id[value]["date"], value))
    retained_candidates.sort(key=lambda row: (row["evidence_dates"][-1], row["merge_id"]))
    return {
        "version": REVIEW_VERSION,
        "start_date": manifest.get("start_date"),
        "end_date": manifest.get("end_date"),
        "source_summary_count": len(record_by_id),
        "reviewed_summary_count": len(reviewed_ids),
        "pending_summary_count": len(pending_ids),
        "review_complete": not pending_ids,
        "formal_memory_write": False,
        "decision_groups": decision_groups,
        "retained_candidates": retained_candidates,
        "pending_summary_ids": pending_ids,
    }


def import_manual_review_outputs(
    paths: ReviewPaths,
    manual_dir: Path,
    *,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    manual_dir = manual_dir.resolve()
    manifest = json.loads(paths.source_manifest.read_text(encoding="utf-8"))
    batches = _load_review_batches(paths)
    imported = 0
    missing = 0
    invalid = 0
    skipped_existing = 0

    for batch in batches:
        batch_id = str(batch.get("batch_id") or "")
        manual_path = manual_dir / f"{batch_id}.json"
        decision_path = paths.decisions_dir / f"{batch_id}.json"
        if decision_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        if not manual_path.exists():
            missing += 1
            continue
        try:
            raw_manual = json.loads(manual_path.read_text(encoding="utf-8"))
            if not isinstance(raw_manual, dict):
                raise ValueError("manual review output must be a JSON object")
            normalized = normalize_manual_review_batch(batch, raw_manual)
            _json_dump(decision_path, normalized)
            imported += 1
        except Exception as exc:
            invalid += 1
            print(f"CROSS_DAY_REVIEW_IMPORT_FAILED {batch_id}: {exc}", file=sys.stderr)

    reviewed_batches: list[dict[str, Any]] = []
    for decision_path in sorted(paths.decisions_dir.glob("review_*.json")):
        reviewed_batches.append(json.loads(decision_path.read_text(encoding="utf-8")))
    merge_staging = build_merge_staging(manifest, reviewed_batches)
    _json_dump(paths.merge_staging, merge_staging)
    return {
        "imported_batches": imported,
        "missing_manual_batches": missing,
        "invalid_manual_batches": invalid,
        "skipped_existing_batches": skipped_existing,
        "reviewed_summary_count": merge_staging["reviewed_summary_count"],
        "pending_summary_count": merge_staging["pending_summary_count"],
        "retained_candidate_count": len(merge_staging["retained_candidates"]),
        "review_complete": merge_staging["review_complete"],
        "merge_staging": str(paths.merge_staging),
        "formal_memory_write": False,
    }


def _stage_parent_from_config(config: dict[str, Any]) -> Path:
    state_dir = Path(str(config.get("state_dir") or ROOT / "state")).resolve()
    return state_dir / "historical_backfill"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare cross-day historical summary review staging")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--stage-parent", default="", help="Parent containing YYYY-MM-DD historical_backfill directories")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--review-root", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--prepare", action="store_true", help="Write source manifest and review batches only")
    parser.add_argument("--import-manual-dir", default="", help="Validate manual review_<NNN>.json decisions and build merge staging")
    parser.add_argument("--overwrite-manual-existing", action="store_true", help="Replace existing imported decisions after validation")
    args = parser.parse_args()
    if bool(args.prepare) == bool(str(args.import_manual_dir).strip()):
        parser.error("choose exactly one of --prepare or --import-manual-dir")

    config = _load_yaml(Path(args.config).resolve())
    stage_parent = Path(args.stage_parent).resolve() if str(args.stage_parent).strip() else _stage_parent_from_config(config)
    review_root = Path(args.review_root).resolve() if str(args.review_root).strip() else None
    paths = review_paths(stage_parent, args.start_date, args.end_date, review_root)

    if args.prepare:
        manifest = prepare_cross_day_review(
            stage_parent=stage_parent,
            start_date=args.start_date,
            end_date=args.end_date,
            review_root=review_root,
            batch_size=max(1, args.batch_size),
        )
        result = {
            "status": "review_prepared",
            "start_date": manifest["start_date"],
            "end_date": manifest["end_date"],
            "source_summary_count": manifest["source_summary_count"],
            "related_pair_count": manifest["related_pair_count"],
            "batch_count": manifest["batch_count"],
            "protected_event_count": manifest["protected_event_count"],
            "projectish_count": manifest["projectish_count"],
            "review_root": manifest["review_root"],
            "formal_memory_write": manifest["formal_memory_write"],
        }
    else:
        imported = import_manual_review_outputs(
            paths,
            Path(args.import_manual_dir),
            overwrite_existing=bool(args.overwrite_manual_existing),
        )
        result = {"status": "manual_review_imported", **imported}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
