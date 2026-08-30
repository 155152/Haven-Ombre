#!/usr/bin/env python3
"""Build non-writing Bucket Markdown previews for historical Ombre backfill.

This stage consumes a *completed* historical promotion review and materializes
only preview artifacts. It does not call BucketManager.create(), does not touch
the live buckets directory, and does not write Memory Moments.

Semantics:
- bucket_core   -> one durable Bucket preview with exactly one ``## Fact`` body
- bucket_moment -> one durable Bucket preview with exactly one ``## Moment`` body
- raw_only      -> provenance JSON only; no Bucket preview

Memory Moments remain derived from Bucket Markdown through
``memory_moments.parse_bucket_moments``. The preview validates that every
promoted candidate would derive exactly one Moment, preventing materialization
from expanding one canonical event into several Moments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import frontmatter
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memory_moments import parse_bucket_moments  # noqa: E402
from utils import sanitize_name  # noqa: E402

MATERIALIZATION_VERSION = 1
PROMOTED_TARGETS = ("bucket_core", "bucket_moment")
TARGET_SECTION = {"bucket_core": "fact", "bucket_moment": "moment"}
TARGET_HEADING = {"bucket_core": "Fact", "bucket_moment": "Moment"}
EXCLUDED_EXISTING_TYPES = frozenset({"source", "feel"})


@dataclass(frozen=True)
class MaterializationPaths:
    root: Path
    manifest: Path
    previews_dir: Path
    raw_only_provenance: Path


def materialization_paths(promotion_staging_path: Path, materialization_root: Path | None = None) -> MaterializationPaths:
    promotion_staging_path = promotion_staging_path.resolve()
    root = (materialization_root or (promotion_staging_path.parent / "materialization_preview")).resolve()
    return MaterializationPaths(
        root=root,
        manifest=root / "materialization_staging.json",
        previews_dir=root / "bucket_previews",
        raw_only_provenance=root / "raw_only_provenance.json",
    )


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _normalized_text(value: Any) -> str:
    text = str(value or "").lower()
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _char_ngrams(value: Any, n: int = 2) -> set[str]:
    text = _normalized_text(value)
    if not text:
        return set()
    if len(text) <= n:
        return {text}
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def _jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _unique_strings(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _unique_ints(values: Iterable[Any]) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    for value in values:
        number = int(value)
        if number in seen:
            continue
        seen.add(number)
        result.append(number)
    return result


def _candidate_id(candidate: dict[str, Any]) -> str:
    value = str(candidate.get("candidate_id") or "").strip()
    if not value:
        raise ValueError("promotion candidate has empty candidate_id")
    return value


def _bucket_id_for_candidate(candidate_id: str) -> str:
    digest = hashlib.sha256(candidate_id.encode("utf-8")).hexdigest()[:16]
    return f"historical_{digest}"


def _candidate_title(candidate: dict[str, Any]) -> str:
    raw = str(candidate.get("title") or "").strip()
    if not raw:
        raw = f"{candidate.get('date') or 'historical'} historical memory"
    title = sanitize_name(raw)
    if not title or title == "unnamed":
        raise ValueError(f"candidate title cannot be materialized: {_candidate_id(candidate)}")
    return title


def _candidate_summary(candidate: dict[str, Any]) -> str:
    summary = str(candidate.get("summary") or "").strip()
    if not summary:
        raise ValueError(f"promotion candidate has empty summary: {_candidate_id(candidate)}")
    return summary


def _candidate_dates(candidate: dict[str, Any]) -> list[str]:
    candidate_id = _candidate_id(candidate)
    evidence_dates = _unique_strings(candidate.get("evidence_dates") or [])
    if not evidence_dates:
        raise ValueError(f"promotion candidate has no evidence_dates: {candidate_id}")
    date_key = str(candidate.get("date") or "").strip()
    if not date_key:
        raise ValueError(f"promotion candidate has no date: {candidate_id}")
    if date_key != max(evidence_dates):
        raise ValueError(f"candidate date must equal latest evidence date: {candidate_id}")
    return evidence_dates


def _validate_candidate_date_range(
    candidate: dict[str, Any],
    *,
    expected_start_date: str,
    expected_end_date: str,
) -> None:
    if not expected_start_date and not expected_end_date:
        return
    if not expected_start_date or not expected_end_date:
        raise ValueError("expected_start_date and expected_end_date must be provided together")
    if expected_start_date > expected_end_date:
        raise ValueError("expected_start_date must not be after expected_end_date")
    candidate_id = _candidate_id(candidate)
    evidence_dates = _candidate_dates(candidate)
    dates = [str(candidate.get("date") or "").strip(), *evidence_dates]
    outside = sorted({value for value in dates if value < expected_start_date or value > expected_end_date})
    if outside:
        raise ValueError(
            f"candidate outside expected historical range {expected_start_date}..{expected_end_date}: "
            f"{candidate_id} ({outside[0]})"
        )


def _candidate_provenance(candidate: dict[str, Any]) -> tuple[list[str], list[int]]:
    candidate_id = _candidate_id(candidate)
    source_summary_ids = _unique_strings(candidate.get("source_summary_ids") or [])
    source_raw_event_ids = _unique_ints(candidate.get("source_event_ids") or [])
    if not source_summary_ids:
        raise ValueError(f"promotion candidate has no source_summary_ids: {candidate_id}")
    if not source_raw_event_ids:
        raise ValueError(f"promotion candidate has no source_event_ids: {candidate_id}")
    return source_summary_ids, source_raw_event_ids


def _candidate_domain(candidate: dict[str, Any]) -> list[str]:
    signals = [str(value or "").strip().lower() for value in candidate.get("signals") or []]
    haystack = " ".join(signals)
    if any(marker in haystack for marker in ("intimacy", "sexual", "sex_", "body_", "aftercare", "dirty_talk", "qixi")):
        return ["intimacy", "relationship"]
    if any(
        marker in haystack
        for marker in (
            "collaboration",
            "user_burden",
            "operational_rule",
            "workflow",
            "review_role",
            "technical_boundary",
        )
    ):
        return ["collaboration"]
    return ["relationship"]


def _candidate_tags(candidate: dict[str, Any], target: str) -> list[str]:
    tags = ["historical_backfill", "historical_materialized", target]
    tags.extend(str(value or "").strip() for value in candidate.get("signals") or [])
    if candidate.get("protected_event"):
        tags.append("protected_historical_event")
    return _unique_strings(tags)


def _bucket_metadata(candidate: dict[str, Any], target: str) -> dict[str, Any]:
    candidate_id = _candidate_id(candidate)
    evidence_dates = _candidate_dates(candidate)
    source_summary_ids, source_raw_event_ids = _candidate_provenance(candidate)
    date_key = str(candidate["date"])
    metadata: dict[str, Any] = {
        "id": _bucket_id_for_candidate(candidate_id),
        "name": _candidate_title(candidate),
        "tags": _candidate_tags(candidate, target),
        "domain": _candidate_domain(candidate),
        "valence": 0.5,
        "arousal": 0.3,
        "importance": 5,
        "type": "dynamic",
        "created": date_key,
        "last_active": date_key,
        "updated_at": date_key,
        "activation_count": 0,
        "source": "historical_backfill",
        "date": date_key,
        "from_historical_backfill": True,
        "historical_backfill_version": MATERIALIZATION_VERSION,
        "historical_candidate_id": candidate_id,
        "historical_promotion_target": target,
        "historical_memory_scope": str(candidate.get("memory_scope") or "current"),
        "historical_upstream_scope": str(candidate.get("upstream_scope") or ""),
        "historical_evidence_dates": evidence_dates,
        "source_summary_ids": source_summary_ids,
        "source_raw_event_ids": source_raw_event_ids,
        "promotion_notes": str(candidate.get("promotion_notes") or "").strip(),
        "protected_historical_event": bool(candidate.get("protected_event")),
    }
    if target == "bucket_moment":
        metadata["event_date"] = date_key
    return metadata


def _bucket_content(candidate: dict[str, Any], target: str) -> str:
    return f"## {TARGET_HEADING[target]}\n\n{_candidate_summary(candidate)}"


def build_bucket_preview(candidate: dict[str, Any], target: str) -> dict[str, Any]:
    if target not in PROMOTED_TARGETS:
        raise ValueError(f"invalid materialization target: {target!r}")
    if str(candidate.get("promotion_target") or target) != target:
        raise ValueError(f"promotion target mismatch for {_candidate_id(candidate)}")
    if str(candidate.get("memory_scope") or "current") != "current":
        raise ValueError(f"promoted bucket candidate must be current memory: {_candidate_id(candidate)}")
    if str(candidate.get("upstream_scope") or "") == "historical_project":
        raise ValueError(f"historical project state cannot be materialized as current bucket: {_candidate_id(candidate)}")

    metadata = _bucket_metadata(candidate, target)
    content = _bucket_content(candidate, target)
    bucket = {"id": metadata["id"], "metadata": metadata, "content": content}
    moments = parse_bucket_moments(bucket)
    expected_section = TARGET_SECTION[target]
    if len(moments) != 1:
        raise ValueError(f"materialized bucket must derive exactly one Moment: {_candidate_id(candidate)}")
    moment = moments[0]
    if str(moment.get("section") or "") != expected_section:
        raise ValueError(f"derived Moment section mismatch for {_candidate_id(candidate)}")
    if str(moment.get("text") or "") != _candidate_summary(candidate):
        raise ValueError(f"derived Moment text changed candidate summary: {_candidate_id(candidate)}")

    primary_domain = sanitize_name(metadata["domain"][0]) if metadata.get("domain") else "未分类"
    filename = f"{metadata['name']}_{metadata['id']}.md"
    target_relpath = str(Path("dynamic") / primary_domain / filename).replace("\\", "/")
    markdown = frontmatter.dumps(frontmatter.Post(content, **metadata))
    return {
        "candidate_id": _candidate_id(candidate),
        "promotion_target": target,
        "bucket_id": metadata["id"],
        "name": metadata["name"],
        "date": metadata["date"],
        "evidence_dates": list(metadata["historical_evidence_dates"]),
        "source_summary_ids": list(metadata["source_summary_ids"]),
        "source_raw_event_ids": list(metadata["source_raw_event_ids"]),
        "protected_historical_event": bool(metadata["protected_historical_event"]),
        "target_relpath": target_relpath,
        "metadata": metadata,
        "content": content,
        "markdown": markdown,
        "derived_moment_preview": {
            "moment_id": str(moment.get("moment_id") or ""),
            "section": str(moment.get("section") or ""),
            "source": str(moment.get("source") or ""),
            "source_id": str(moment.get("source_id") or ""),
            "text": str(moment.get("text") or ""),
        },
    }


def _fast_source_evidence_path(path: Path) -> bool:
    return path.name.startswith("source_") and path.parent.name == "cyberboss"


def load_existing_semantic_buckets(buckets_dir: Path) -> list[dict[str, Any]]:
    buckets_dir = buckets_dir.resolve()
    if not buckets_dir.exists():
        return []
    result: list[dict[str, Any]] = []
    for path in sorted(buckets_dir.rglob("*.md")):
        if _fast_source_evidence_path(path):
            continue
        post = frontmatter.load(path)
        metadata = dict(post.metadata)
        bucket_type = str(metadata.get("type") or "dynamic").strip().lower()
        if bucket_type in EXCLUDED_EXISTING_TYPES:
            continue
        result.append(
            {
                "id": str(metadata.get("id") or ""),
                "name": str(metadata.get("name") or metadata.get("title") or ""),
                "type": bucket_type,
                "date": str(metadata.get("event_date") or metadata.get("date") or ""),
                "source": str(metadata.get("source") or ""),
                "source_raw_event_ids": _unique_ints(metadata.get("source_raw_event_ids") or []),
                "historical_candidate_id": str(metadata.get("historical_candidate_id") or ""),
                "content": str(post.content or ""),
                "path": str(path),
            }
        )
    return result


def _existing_duplicate_match(candidate: dict[str, Any], preview: dict[str, Any], existing_buckets: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidate_id = _candidate_id(candidate)
    candidate_events = set(preview["source_raw_event_ids"])
    candidate_text = f"{preview['name']} {preview['content']}"
    candidate_grams = _char_ngrams(candidate_text, 2)
    best: dict[str, Any] | None = None

    for existing in existing_buckets:
        if existing.get("historical_candidate_id") == candidate_id:
            return {
                "existing_bucket_id": existing.get("id"),
                "existing_bucket_path": existing.get("path"),
                "reason": "same_historical_candidate_id",
                "event_overlap": 1.0,
                "text_similarity": 1.0,
            }

        existing_events = set(existing.get("source_raw_event_ids") or [])
        intersection = candidate_events & existing_events
        candidate_coverage = len(intersection) / len(candidate_events) if candidate_events else 0.0
        existing_coverage = len(intersection) / len(existing_events) if existing_events else 0.0
        text_similarity = _jaccard(candidate_grams, _char_ngrams(f"{existing.get('name')} {existing.get('content')}", 2))
        same_date = bool(preview.get("date")) and preview.get("date") == existing.get("date")
        exact_text = _normalized_text(_candidate_summary(candidate)) == _normalized_text(existing.get("content"))

        duplicate = exact_text or (
            bool(intersection)
            and same_date
            and max(candidate_coverage, existing_coverage) >= 0.75
            and text_similarity >= 0.08
        ) or (
            bool(intersection)
            and same_date
            and candidate_coverage >= 0.5
            and existing_coverage >= 1.0
            and text_similarity >= 0.12
        )
        if not duplicate:
            continue
        score = 0.55 * max(candidate_coverage, existing_coverage) + 0.25 * min(candidate_coverage, existing_coverage) + 0.20 * text_similarity
        match = {
            "existing_bucket_id": existing.get("id"),
            "existing_bucket_path": existing.get("path"),
            "reason": "same_raw_event_provenance_and_semantics" if intersection else "same_summary_text",
            "shared_source_raw_event_ids": sorted(intersection),
            "candidate_event_coverage": round(candidate_coverage, 4),
            "existing_event_coverage": round(existing_coverage, 4),
            "text_similarity": round(text_similarity, 4),
            "score": round(score, 4),
        }
        if best is None or float(match["score"]) > float(best.get("score") or 0.0):
            best = match
    return best


def _staging_duplicate_warnings(previews: list[dict[str, Any]]) -> list[dict[str, Any]]:
    warnings: list[dict[str, Any]] = []
    for left_index in range(len(previews)):
        left = previews[left_index]
        left_events = set(left.get("source_raw_event_ids") or [])
        left_text = _normalized_text(left.get("content"))
        for right in previews[left_index + 1 :]:
            right_events = set(right.get("source_raw_event_ids") or [])
            shared = left_events & right_events
            if not shared:
                continue
            exact_text = left_text == _normalized_text(right.get("content"))
            same_target = left.get("promotion_target") == right.get("promotion_target")
            event_jaccard = _jaccard(left_events, right_events)
            if exact_text or (same_target and event_jaccard >= 0.95):
                warnings.append(
                    {
                        "left_candidate_id": left["candidate_id"],
                        "right_candidate_id": right["candidate_id"],
                        "promotion_target": left.get("promotion_target") if same_target else "mixed",
                        "shared_source_raw_event_ids": sorted(shared),
                        "event_jaccard": round(event_jaccard, 4),
                        "exact_text": exact_text,
                        "blocking": True,
                    }
                )
            elif same_target and left.get("promotion_target") == "bucket_moment":
                warnings.append(
                    {
                        "left_candidate_id": left["candidate_id"],
                        "right_candidate_id": right["candidate_id"],
                        "promotion_target": "bucket_moment",
                        "shared_source_raw_event_ids": sorted(shared),
                        "event_jaccard": round(event_jaccard, 4),
                        "exact_text": False,
                        "blocking": False,
                    }
                )
    return warnings


def _raw_only_provenance(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence_dates = _candidate_dates(candidate)
    source_summary_ids, source_raw_event_ids = _candidate_provenance(candidate)
    if str(candidate.get("promotion_target") or "raw_only") != "raw_only":
        raise ValueError(f"raw-only provenance target mismatch: {_candidate_id(candidate)}")
    return {
        "candidate_id": _candidate_id(candidate),
        "date": str(candidate.get("date") or ""),
        "evidence_dates": evidence_dates,
        "source_summary_ids": source_summary_ids,
        "source_raw_event_ids": source_raw_event_ids,
        "memory_scope": str(candidate.get("memory_scope") or ""),
        "upstream_scope": str(candidate.get("upstream_scope") or ""),
        "protected_historical_event": bool(candidate.get("protected_event")),
        "promotion_notes": str(candidate.get("promotion_notes") or "").strip(),
    }


def plan_materialization_staging(
    *,
    promotion: dict[str, Any],
    buckets_dir: Path,
    expected_start_date: str = "",
    expected_end_date: str = "",
) -> dict[str, Any]:
    """Build a read-only materialization plan without writing preview artifacts."""
    if not promotion.get("promotion_review_complete") or int(promotion.get("pending_candidate_count") or 0) != 0:
        raise ValueError("promotion review must be complete before materialization")
    if promotion.get("formal_memory_write") is not False or promotion.get("formal_moment_write") is not False:
        raise ValueError("promotion staging must remain non-writing")

    counts = {
        "bucket_core": len(promotion.get("bucket_core_candidates") or []),
        "bucket_moment": len(promotion.get("bucket_moment_candidates") or []),
        "raw_only": len(promotion.get("raw_only_candidates") or []),
    }
    if counts["bucket_core"] != int(promotion.get("bucket_core_count") or 0):
        raise ValueError("bucket_core_count mismatch in promotion staging")
    if counts["bucket_moment"] != int(promotion.get("bucket_moment_count") or 0):
        raise ValueError("bucket_moment_count mismatch in promotion staging")
    if counts["raw_only"] != int(promotion.get("raw_only_count") or 0):
        raise ValueError("raw_only_count mismatch in promotion staging")

    expected_start_date = str(expected_start_date or "").strip()
    expected_end_date = str(expected_end_date or "").strip()
    all_candidates = [
        row
        for target in (*PROMOTED_TARGETS, "raw_only")
        for row in promotion.get(f"{target}_candidates") or []
    ]
    for candidate in all_candidates:
        _validate_candidate_date_range(
            candidate,
            expected_start_date=expected_start_date,
            expected_end_date=expected_end_date,
        )

    existing_buckets = load_existing_semantic_buckets(buckets_dir)
    all_promoted_previews: list[dict[str, Any]] = []
    materialized_previews: list[dict[str, Any]] = []
    existing_duplicates: list[dict[str, Any]] = []
    for target in PROMOTED_TARGETS:
        for candidate in promotion.get(f"{target}_candidates") or []:
            preview = build_bucket_preview(candidate, target)
            all_promoted_previews.append(preview)
            duplicate = _existing_duplicate_match(candidate, preview, existing_buckets)
            if duplicate:
                existing_duplicates.append(
                    {
                        "candidate_id": preview["candidate_id"],
                        "promotion_target": target,
                        "date": preview["date"],
                        "evidence_dates": list(preview["evidence_dates"]),
                        "source_summary_ids": list(preview["source_summary_ids"]),
                        "source_raw_event_ids": list(preview["source_raw_event_ids"]),
                        "protected_historical_event": bool(preview["protected_historical_event"]),
                        **duplicate,
                    }
                )
                continue
            materialized_previews.append(preview)

    raw_only = [_raw_only_provenance(row) for row in promotion.get("raw_only_candidates") or []]
    warnings = _staging_duplicate_warnings(all_promoted_previews)
    blocking_warnings = [row for row in warnings if row.get("blocking")]
    target_paths = [row["target_relpath"] for row in materialized_previews]
    if len(target_paths) != len(set(target_paths)):
        raise ValueError("materialized target path collision")
    bucket_ids = [row["bucket_id"] for row in materialized_previews]
    if len(bucket_ids) != len(set(bucket_ids)):
        raise ValueError("materialized bucket id collision")

    core_previews = [row for row in materialized_previews if row["promotion_target"] == "bucket_core"]
    moment_previews = [row for row in materialized_previews if row["promotion_target"] == "bucket_moment"]
    return {
        "expected_date_range": {
            "start_date": expected_start_date,
            "end_date": expected_end_date,
        },
        "source_candidate_count": int(promotion.get("source_candidate_count") or sum(counts.values())),
        "promotion_counts": counts,
        "promoted_candidate_count": counts["bucket_core"] + counts["bucket_moment"],
        "existing_semantic_bucket_count": len(existing_buckets),
        "existing_duplicate_count": len(existing_duplicates),
        "materialized_preview_count": len(materialized_previews),
        "bucket_core_preview_count": len(core_previews),
        "bucket_moment_preview_count": len(moment_previews),
        "derived_moment_preview_count": len(materialized_previews),
        "raw_only_provenance_count": len(raw_only),
        "staging_duplicate_warning_count": len(warnings),
        "blocking_duplicate_warning_count": len(blocking_warnings),
        "materialization_ready": not blocking_warnings,
        "existing_duplicates": existing_duplicates,
        "staging_duplicate_warnings": warnings,
        "materialized_previews": materialized_previews,
        "raw_only_provenance": raw_only,
    }


def prepare_materialization_staging(
    *,
    promotion_staging_path: Path,
    buckets_dir: Path,
    materialization_root: Path | None = None,
    expected_start_date: str = "",
    expected_end_date: str = "",
) -> dict[str, Any]:
    promotion_staging_path = promotion_staging_path.resolve()
    if not promotion_staging_path.exists():
        raise FileNotFoundError(f"promotion staging not found: {promotion_staging_path}")
    promotion = json.loads(promotion_staging_path.read_text(encoding="utf-8"))
    if not promotion.get("promotion_review_complete") or int(promotion.get("pending_candidate_count") or 0) != 0:
        raise ValueError("promotion review must be complete before materialization")
    if promotion.get("formal_memory_write") is not False or promotion.get("formal_moment_write") is not False:
        raise ValueError("promotion staging must remain non-writing")

    counts = {
        "bucket_core": len(promotion.get("bucket_core_candidates") or []),
        "bucket_moment": len(promotion.get("bucket_moment_candidates") or []),
        "raw_only": len(promotion.get("raw_only_candidates") or []),
    }
    if counts["bucket_core"] != int(promotion.get("bucket_core_count") or 0):
        raise ValueError("bucket_core_count mismatch in promotion staging")
    if counts["bucket_moment"] != int(promotion.get("bucket_moment_count") or 0):
        raise ValueError("bucket_moment_count mismatch in promotion staging")
    if counts["raw_only"] != int(promotion.get("raw_only_count") or 0):
        raise ValueError("raw_only_count mismatch in promotion staging")

    expected_start_date = str(expected_start_date or "").strip()
    expected_end_date = str(expected_end_date or "").strip()
    all_candidates = [
        row
        for target in (*PROMOTED_TARGETS, "raw_only")
        for row in promotion.get(f"{target}_candidates") or []
    ]
    for candidate in all_candidates:
        _validate_candidate_date_range(
            candidate,
            expected_start_date=expected_start_date,
            expected_end_date=expected_end_date,
        )

    existing_buckets = load_existing_semantic_buckets(buckets_dir)
    all_promoted_previews: list[dict[str, Any]] = []
    materialized_previews: list[dict[str, Any]] = []
    existing_duplicates: list[dict[str, Any]] = []
    paths = materialization_paths(promotion_staging_path, materialization_root)
    paths.previews_dir.mkdir(parents=True, exist_ok=True)
    for stale_preview in paths.previews_dir.glob("*.md"):
        stale_preview.unlink()

    for target in PROMOTED_TARGETS:
        for candidate in promotion.get(f"{target}_candidates") or []:
            preview = build_bucket_preview(candidate, target)
            all_promoted_previews.append(preview)
            duplicate = _existing_duplicate_match(candidate, preview, existing_buckets)
            if duplicate:
                existing_duplicates.append(
                    {
                        "candidate_id": preview["candidate_id"],
                        "promotion_target": target,
                        "date": preview["date"],
                        "evidence_dates": list(preview["evidence_dates"]),
                        "source_summary_ids": list(preview["source_summary_ids"]),
                        "source_raw_event_ids": list(preview["source_raw_event_ids"]),
                        "protected_historical_event": bool(preview["protected_historical_event"]),
                        **duplicate,
                    }
                )
                continue
            materialized_previews.append(preview)
            preview_path = paths.previews_dir / f"{preview['bucket_id']}.md"
            preview_path.write_text(preview["markdown"], encoding="utf-8")

    raw_only = [_raw_only_provenance(row) for row in promotion.get("raw_only_candidates") or []]
    _json_dump(
        paths.raw_only_provenance,
        {
            "version": MATERIALIZATION_VERSION,
            "formal_memory_write": False,
            "formal_moment_write": False,
            "raw_only_count": len(raw_only),
            "items": raw_only,
        },
    )

    warnings = _staging_duplicate_warnings(all_promoted_previews)
    blocking_warnings = [row for row in warnings if row.get("blocking")]
    target_paths = [row["target_relpath"] for row in materialized_previews]
    if len(target_paths) != len(set(target_paths)):
        raise ValueError("materialized target path collision")
    bucket_ids = [row["bucket_id"] for row in materialized_previews]
    if len(bucket_ids) != len(set(bucket_ids)):
        raise ValueError("materialized bucket id collision")

    core_previews = [row for row in materialized_previews if row["promotion_target"] == "bucket_core"]
    moment_previews = [row for row in materialized_previews if row["promotion_target"] == "bucket_moment"]
    manifest = {
        "version": MATERIALIZATION_VERSION,
        "promotion_staging": str(promotion_staging_path),
        "buckets_dir_read_only": str(buckets_dir.resolve()),
        "formal_memory_write": False,
        "formal_moment_write": False,
        "apply_supported": False,
        "moment_semantics": "Memory Moments are derived from Bucket Markdown; materialization writes no Moment index.",
        "expected_date_range": {
            "start_date": expected_start_date,
            "end_date": expected_end_date,
        },
        "source_candidate_count": int(promotion.get("source_candidate_count") or sum(counts.values())),
        "promotion_counts": counts,
        "promoted_candidate_count": counts["bucket_core"] + counts["bucket_moment"],
        "existing_semantic_bucket_count": len(existing_buckets),
        "existing_duplicate_count": len(existing_duplicates),
        "materialized_preview_count": len(materialized_previews),
        "bucket_core_preview_count": len(core_previews),
        "bucket_moment_preview_count": len(moment_previews),
        "derived_moment_preview_count": len(materialized_previews),
        "raw_only_provenance_count": len(raw_only),
        "staging_duplicate_warning_count": len(warnings),
        "blocking_duplicate_warning_count": len(blocking_warnings),
        "materialization_ready": not blocking_warnings,
        "rules": {
            "one_promoted_candidate_per_bucket_preview": True,
            "one_derived_moment_per_bucket_preview": True,
            "raw_only_never_gets_bucket_preview": True,
            "source_evidence_buckets_do_not_block_semantic_materialization": True,
            "existing_semantic_duplicates_are_skipped_not_modified": True,
            "historical_project_state_cannot_be_current_bucket": True,
            "protected_intimacy_text_is_not_sanitized_or_summarized_again": True,
            "formal_bucket_write": False,
            "formal_moment_write": False,
        },
        "existing_duplicates": existing_duplicates,
        "staging_duplicate_warnings": warnings,
        "bucket_previews": [
            {key: value for key, value in row.items() if key != "markdown"}
            for row in materialized_previews
        ],
        "raw_only_provenance_file": str(paths.raw_only_provenance),
        "bucket_previews_dir": str(paths.previews_dir),
    }
    _json_dump(paths.manifest, manifest)
    return manifest


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError("config must be a YAML object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Build non-writing historical Bucket materialization previews")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    parser.add_argument("--promotion-staging", required=True)
    parser.add_argument("--materialization-root", default="")
    parser.add_argument("--start-date", default="")
    parser.add_argument("--end-date", default="")
    args = parser.parse_args()

    config = _load_config(Path(args.config).resolve())
    buckets_dir = Path(str(config.get("buckets_dir") or "")).resolve()
    if not str(config.get("buckets_dir") or "").strip():
        raise ValueError("config.buckets_dir is required for duplicate preflight")
    root = Path(args.materialization_root).resolve() if str(args.materialization_root).strip() else None
    manifest = prepare_materialization_staging(
        promotion_staging_path=Path(args.promotion_staging),
        buckets_dir=buckets_dir,
        materialization_root=root,
        expected_start_date=args.start_date,
        expected_end_date=args.end_date,
    )
    result = {
        "status": "historical_materialization_preview_ready",
        "promotion_counts": manifest["promotion_counts"],
        "expected_date_range": manifest["expected_date_range"],
        "existing_duplicate_count": manifest["existing_duplicate_count"],
        "materialized_preview_count": manifest["materialized_preview_count"],
        "bucket_core_preview_count": manifest["bucket_core_preview_count"],
        "bucket_moment_preview_count": manifest["bucket_moment_preview_count"],
        "derived_moment_preview_count": manifest["derived_moment_preview_count"],
        "raw_only_provenance_count": manifest["raw_only_provenance_count"],
        "blocking_duplicate_warning_count": manifest["blocking_duplicate_warning_count"],
        "materialization_ready": manifest["materialization_ready"],
        "materialization_staging": str(materialization_paths(Path(args.promotion_staging), root).manifest),
        "formal_memory_write": False,
        "formal_moment_write": False,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
