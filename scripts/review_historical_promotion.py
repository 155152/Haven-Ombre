#!/usr/bin/env python3
"""Promotion review staging for historical Ombre backfill.

This third pass reads the completed bridge-review merge staging and classifies
retained candidates for *review only* into three semantic targets:

- bucket_core: stable long-term facts, preferences, boundaries, relationship or
  identity knowledge that belongs in a durable Bucket body.
- bucket_moment: concrete important events that should remain event-shaped
  inside a durable Bucket; Ombre derives Memory Moments from Bucket content.
- raw_only: provenance remains in raw/staging history and is not promoted to a
  durable Bucket.

No Bucket Markdown files or Memory Moment indexes are written by this module.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROMOTION_VERSION = 1
DEFAULT_BATCH_SIZE = 28
PROMOTION_TARGETS = {"bucket_core", "bucket_moment", "raw_only"}
MEMORY_SCOPES = {"current", "historical"}


@dataclass(frozen=True)
class PromotionPaths:
    root: Path
    source_manifest: Path
    batches_dir: Path
    decisions_dir: Path
    staging: Path


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def promotion_paths(bridge_root: Path, promotion_root: Path | None = None) -> PromotionPaths:
    root = (promotion_root or (bridge_root / "promotion_review")).resolve()
    return PromotionPaths(
        root=root,
        source_manifest=root / "source_manifest.json",
        batches_dir=root / "review_batches",
        decisions_dir=root / "decisions",
        staging=root / "promotion_staging.json",
    )


def _suggest_target(candidate: dict[str, Any]) -> tuple[str, str]:
    scope = str(candidate.get("scope") or "")
    signals = {str(value or "").strip().lower() for value in candidate.get("signals") or []}
    protected = bool(candidate.get("protected_event"))

    if scope == "historical_project":
        return "raw_only", "historical project state should not become a current rule by default"
    if protected:
        return "bucket_moment", "protected concrete event should remain event-shaped in durable memory"

    moment_markers = {
        "relationship_milestone",
        "family_milestone",
        "intimacy_milestone",
        "qixi_event",
        "relationship_event",
        "intimacy_event",
        "verified_milestone",
    }
    if signals & moment_markers:
        return "bucket_moment", "event or milestone signal is better preserved as a concrete bucket moment"

    core_fragments = (
        "preference",
        "boundary",
        "stable_",
        "relationship_insight",
        "relationship_growth",
        "identity_continuity",
        "relationship_symbol",
        "relationship_language",
        "long_term",
        "autonomy_rule",
        "shared_date",
        "consent",
    )
    if any(any(fragment in signal for fragment in core_fragments) for signal in signals):
        return "bucket_core", "stable preference/boundary/relationship/identity signal suggests durable core memory"

    return "raw_only", "no strong durable-memory marker; manual review may still promote if meaning warrants it"


def _candidate_row(candidate: dict[str, Any]) -> dict[str, Any]:
    merge_id = str(candidate.get("merge_id") or "").strip()
    if not merge_id:
        raise ValueError("retained candidate has empty merge_id")
    evidence_dates = [str(value) for value in candidate.get("evidence_dates") or [] if str(value)]
    if not evidence_dates:
        raise ValueError(f"retained candidate has no evidence_dates: {merge_id}")
    target, rationale = _suggest_target(candidate)
    inherited_scope = str(candidate.get("scope") or "")
    memory_scope = "historical" if inherited_scope == "historical_project" else "current"
    return {
        "candidate_id": merge_id,
        "date": max(evidence_dates),
        "title": str(candidate.get("title") or ""),
        "summary": str(candidate.get("summary") or ""),
        "signals": list(candidate.get("signals") or []),
        "source_summary_ids": list(candidate.get("source_summary_ids") or []),
        "source_event_ids": [int(value) for value in candidate.get("source_event_ids") or []],
        "evidence_dates": evidence_dates,
        "protected_event": bool(candidate.get("protected_event")),
        "projectish": bool(candidate.get("projectish")),
        "upstream_scope": inherited_scope,
        "upstream_classification": str(candidate.get("classification") or ""),
        "suggested_target": target,
        "suggested_memory_scope": memory_scope,
        "suggestion_reason": rationale,
    }


def prepare_promotion_review(
    *, bridge_root: Path,
    promotion_root: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    bridge_root = bridge_root.resolve()
    bridge_staging_path = bridge_root / "merge_staging.json"
    if not bridge_staging_path.exists():
        raise FileNotFoundError(f"bridge merge staging not found: {bridge_staging_path}")
    bridge = json.loads(bridge_staging_path.read_text(encoding="utf-8"))
    if not bridge.get("bridge_review_complete") or int(bridge.get("pending_bridge_candidate_count") or 0) != 0:
        raise ValueError("bridge review must be complete before promotion review")
    if bridge.get("formal_memory_write") is not False:
        raise ValueError("bridge staging must remain non-writing")

    candidates = [_candidate_row(row) for row in bridge.get("retained_candidates") or []]
    if not candidates:
        raise ValueError("bridge staging has no retained candidates")
    ids = [row["candidate_id"] for row in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate promotion candidate_id")

    # Keep review chronological. No automatic promotion decision is made here;
    # suggestions only reduce reviewer search cost.
    candidates.sort(key=lambda row: (row["date"], row["candidate_id"]))
    batch_size = max(1, int(batch_size))
    paths = promotion_paths(bridge_root, promotion_root)
    paths.batches_dir.mkdir(parents=True, exist_ok=True)
    paths.decisions_dir.mkdir(parents=True, exist_ok=True)

    batches: list[dict[str, Any]] = []
    for index, start in enumerate(range(0, len(candidates), batch_size), start=1):
        items = candidates[start : start + batch_size]
        payload = {
            "version": PROMOTION_VERSION,
            "batch_id": f"promotion_{index:03d}",
            "item_count": len(items),
            "items": items,
        }
        batches.append(payload)
        _json_dump(paths.batches_dir / f"{payload['batch_id']}.json", payload)

    suggestion_counts = Counter(row["suggested_target"] for row in candidates)
    scope_counts = Counter(row["upstream_scope"] for row in candidates)
    manifest = {
        "version": PROMOTION_VERSION,
        "bridge_root": str(bridge_root),
        "source_candidate_count": len(candidates),
        "batch_count": len(batches),
        "batch_size": batch_size,
        "formal_memory_write": False,
        "moment_semantics": "Memory Moments are derived from durable Bucket content; bucket_moment is a review role, not a direct write target.",
        "allowed_targets": sorted(PROMOTION_TARGETS),
        "allowed_memory_scopes": sorted(MEMORY_SCOPES),
        "suggestion_counts": dict(sorted(suggestion_counts.items())),
        "upstream_scope_counts": dict(sorted(scope_counts.items())),
        "rules": {
            "suggestions_are_not_decisions": True,
            "protected_raw_only_requires_explicit_override": True,
            "historical_project_cannot_be_current_bucket": True,
            "formal_bucket_write": False,
            "formal_moment_write": False,
        },
    }
    _json_dump(paths.source_manifest, manifest)
    return manifest


def _load_batches(paths: PromotionPaths) -> list[dict[str, Any]]:
    if not paths.source_manifest.exists():
        raise FileNotFoundError(f"promotion manifest not found: {paths.source_manifest}")
    result: list[dict[str, Any]] = []
    for path in sorted(paths.batches_dir.glob("promotion_*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"promotion batch must be object: {path}")
        result.append(value)
    if not result:
        raise FileNotFoundError(f"no promotion batches found: {paths.batches_dir}")
    return result


def normalize_promotion_batch(batch: dict[str, Any], raw: dict[str, Any]) -> dict[str, Any]:
    batch_id = str(batch.get("batch_id") or "")
    if str(raw.get("batch_id") or batch_id) != batch_id:
        raise ValueError(f"batch_id mismatch: expected {batch_id}")
    item_by_id = {str(row.get("candidate_id") or ""): row for row in batch.get("items") or []}
    expected = set(item_by_id)
    decisions = raw.get("decisions") or []
    if not isinstance(decisions, list):
        raise ValueError("decisions must be a list")

    normalized: list[dict[str, Any]] = []
    covered: set[str] = set()
    for index, decision in enumerate(decisions, start=1):
        if not isinstance(decision, dict):
            raise ValueError(f"decision #{index} must be object")
        candidate_ids = [str(value).strip() for value in decision.get("candidate_ids") or [] if str(value).strip()]
        if not candidate_ids:
            raise ValueError(f"decision #{index} has no candidate_ids")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"decision #{index} repeats candidate_ids")
        unknown = set(candidate_ids) - expected
        if unknown:
            raise ValueError(f"decision #{index} references unknown ids: {sorted(unknown)}")
        overlap = covered & set(candidate_ids)
        if overlap:
            raise ValueError(f"candidate ids reviewed twice: {sorted(overlap)}")
        covered.update(candidate_ids)

        target = str(decision.get("target") or "").strip()
        if target not in PROMOTION_TARGETS:
            raise ValueError(f"invalid target in decision #{index}: {target!r}")
        memory_scope = str(decision.get("memory_scope") or "").strip()
        if not memory_scope:
            memory_scope = "historical" if all(item_by_id[value]["upstream_scope"] == "historical_project" for value in candidate_ids) else "current"
        if memory_scope not in MEMORY_SCOPES:
            raise ValueError(f"invalid memory_scope in decision #{index}: {memory_scope!r}")

        items = [item_by_id[value] for value in candidate_ids]
        if target != "raw_only" and any(row["upstream_scope"] == "historical_project" for row in items) and memory_scope != "historical":
            raise ValueError("historical_project candidate cannot be promoted as current memory")
        if target == "bucket_core" and memory_scope == "historical":
            raise ValueError("historical material may be raw_only or bucket_moment, not current-style bucket_core")
        if target == "raw_only" and any(row.get("protected_event") for row in items):
            if decision.get("allow_protected_raw_only") is not True:
                raise ValueError("protected event raw_only requires allow_protected_raw_only=true")
            reason = " ".join(str(decision.get("notes") or "").split()).strip()
            if not reason:
                raise ValueError("protected event raw_only requires an explicit notes reason")

        normalized.append(
            {
                "candidate_ids": candidate_ids,
                "target": target,
                "memory_scope": memory_scope,
                "allow_protected_raw_only": bool(decision.get("allow_protected_raw_only")),
                "notes": " ".join(str(decision.get("notes") or "").split()).strip()[:1200],
            }
        )

    missing = expected - covered
    if missing:
        raise ValueError(f"manual promotion batch incomplete; missing {len(missing)} candidate ids")
    return {
        "version": PROMOTION_VERSION,
        "batch_id": batch_id,
        "origin": "manual_chatgpt_historical_promotion_review",
        "decisions": normalized,
    }


def build_promotion_staging(batches: list[dict[str, Any]], reviewed: list[dict[str, Any]]) -> dict[str, Any]:
    item_by_id = {
        str(item.get("candidate_id") or ""): item
        for batch in batches
        for item in batch.get("items") or []
    }
    reviewed_ids: set[str] = set()
    bucket_core: list[dict[str, Any]] = []
    bucket_moment: list[dict[str, Any]] = []
    raw_only: list[dict[str, Any]] = []
    decision_groups: list[dict[str, Any]] = []

    target_lists = {
        "bucket_core": bucket_core,
        "bucket_moment": bucket_moment,
        "raw_only": raw_only,
    }
    for result in reviewed:
        batch_id = str(result.get("batch_id") or "")
        for index, decision in enumerate(result.get("decisions") or [], start=1):
            decision_groups.append({"group_id": f"{batch_id}_g{index:02d}", **decision})
            for candidate_id in decision.get("candidate_ids") or []:
                reviewed_ids.add(candidate_id)
                row = dict(item_by_id[candidate_id])
                row["promotion_target"] = decision["target"]
                row["memory_scope"] = decision["memory_scope"]
                row["promotion_notes"] = decision.get("notes") or ""
                target_lists[decision["target"]].append(row)

    pending = sorted(set(item_by_id) - reviewed_ids, key=lambda value: (item_by_id[value]["date"], value))
    for rows in target_lists.values():
        rows.sort(key=lambda row: (row["date"], row["candidate_id"]))
    return {
        "version": PROMOTION_VERSION,
        "source_candidate_count": len(item_by_id),
        "reviewed_candidate_count": len(reviewed_ids),
        "pending_candidate_count": len(pending),
        "promotion_review_complete": not pending,
        "formal_memory_write": False,
        "formal_moment_write": False,
        "bucket_core_count": len(bucket_core),
        "bucket_moment_count": len(bucket_moment),
        "raw_only_count": len(raw_only),
        "decision_groups": decision_groups,
        "bucket_core_candidates": bucket_core,
        "bucket_moment_candidates": bucket_moment,
        "raw_only_candidates": raw_only,
        "pending_candidate_ids": pending,
    }


def import_manual_promotion(
    *,
    bridge_root: Path,
    manual_dir: Path,
    promotion_root: Path | None = None,
    overwrite_existing: bool = False,
) -> dict[str, Any]:
    paths = promotion_paths(bridge_root.resolve(), promotion_root)
    batches = _load_batches(paths)
    manual_dir = manual_dir.resolve()
    imported = missing = invalid = skipped = 0

    for batch in batches:
        batch_id = str(batch.get("batch_id") or "")
        manual_path = manual_dir / f"{batch_id}.json"
        decision_path = paths.decisions_dir / f"{batch_id}.json"
        if decision_path.exists() and not overwrite_existing:
            skipped += 1
            continue
        if not manual_path.exists():
            missing += 1
            continue
        try:
            raw = json.loads(manual_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("manual promotion output must be JSON object")
            normalized = normalize_promotion_batch(batch, raw)
            _json_dump(decision_path, normalized)
            imported += 1
        except Exception as exc:
            invalid += 1
            print(f"PROMOTION_REVIEW_IMPORT_FAILED {batch_id}: {exc}")

    reviewed = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(paths.decisions_dir.glob("promotion_*.json"))
    ]
    staging = build_promotion_staging(batches, reviewed)
    _json_dump(paths.staging, staging)
    return {
        "imported_batches": imported,
        "missing_manual_batches": missing,
        "invalid_manual_batches": invalid,
        "skipped_existing_batches": skipped,
        "reviewed_candidate_count": staging["reviewed_candidate_count"],
        "pending_candidate_count": staging["pending_candidate_count"],
        "bucket_core_count": staging["bucket_core_count"],
        "bucket_moment_count": staging["bucket_moment_count"],
        "raw_only_count": staging["raw_only_count"],
        "promotion_review_complete": staging["promotion_review_complete"],
        "promotion_staging": str(paths.staging),
        "formal_memory_write": False,
        "formal_moment_write": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare or import historical promotion review staging")
    parser.add_argument("--bridge-root", required=True)
    parser.add_argument("--promotion-root", default="")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--import-manual-dir", default="")
    parser.add_argument("--overwrite-manual-existing", action="store_true")
    args = parser.parse_args()
    if bool(args.prepare) == bool(str(args.import_manual_dir).strip()):
        parser.error("choose exactly one of --prepare or --import-manual-dir")

    bridge_root = Path(args.bridge_root).resolve()
    promotion_root = Path(args.promotion_root).resolve() if str(args.promotion_root).strip() else None
    if args.prepare:
        manifest = prepare_promotion_review(
            bridge_root=bridge_root,
            promotion_root=promotion_root,
            batch_size=max(1, args.batch_size),
        )
        result = {
            "status": "promotion_review_prepared",
            "source_candidate_count": manifest["source_candidate_count"],
            "batch_count": manifest["batch_count"],
            "suggestion_counts": manifest["suggestion_counts"],
            "upstream_scope_counts": manifest["upstream_scope_counts"],
            "promotion_root": str(promotion_paths(bridge_root, promotion_root).root),
            "formal_memory_write": False,
            "formal_moment_write": False,
        }
    else:
        result = {
            "status": "promotion_manual_imported",
            **import_manual_promotion(
                bridge_root=bridge_root,
                manual_dir=Path(args.import_manual_dir),
                promotion_root=promotion_root,
                overwrite_existing=bool(args.overwrite_manual_existing),
            ),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
