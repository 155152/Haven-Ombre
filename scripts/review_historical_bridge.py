#!/usr/bin/env python3
"""Second-pass bridge review for cross-day historical memory staging.

The first cross-day review intentionally validates decisions inside bounded review
batches. This bridge pass only revisits retained candidates that still have a
similarity edge to a candidate originating from a different first-pass batch.
It never writes Ombre Buckets or Moments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from .review_historical_llm_memory import REVIEW_VERSION, normalize_manual_review_batch, score_related_pairs
except ImportError:  # pragma: no cover - direct script execution
    from review_historical_llm_memory import REVIEW_VERSION, normalize_manual_review_batch, score_related_pairs

BRIDGE_VERSION = 1


@dataclass(frozen=True)
class BridgePaths:
    root: Path
    source_manifest: Path
    groups_dir: Path
    decisions_dir: Path
    merge_staging: Path


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bridge_paths(cross_day_root: Path, bridge_root: Path | None = None) -> BridgePaths:
    root = (bridge_root or (cross_day_root / "bridge_review")).resolve()
    return BridgePaths(
        root=root,
        source_manifest=root / "source_manifest.json",
        groups_dir=root / "review_groups",
        decisions_dir=root / "decisions",
        merge_staging=root / "merge_staging.json",
    )


def _first_pass_batch(candidate: dict[str, Any]) -> str:
    group_id = str(candidate.get("group_id") or "")
    return group_id.split("_g", 1)[0] if "_g" in group_id else group_id


def _candidate_record(candidate: dict[str, Any]) -> dict[str, Any]:
    evidence_dates = [str(value) for value in candidate.get("evidence_dates") or [] if str(value)]
    if not evidence_dates:
        raise ValueError(f"candidate has no evidence_dates: {candidate.get('merge_id')}")
    return {
        "summary_id": str(candidate.get("merge_id") or ""),
        "date": max(evidence_dates),
        "title": str(candidate.get("title") or ""),
        "summary": str(candidate.get("summary") or ""),
        "signals": list(candidate.get("signals") or []),
        "source_event_ids": [int(value) for value in candidate.get("source_event_ids") or []],
        "protected_event": bool(candidate.get("protected_event")),
        "projectish": bool(candidate.get("projectish")),
        "upstream_group_id": str(candidate.get("group_id") or ""),
        "upstream_scope": str(candidate.get("scope") or ""),
        "upstream_classification": str(candidate.get("classification") or ""),
        "upstream_source_summary_ids": list(candidate.get("source_summary_ids") or []),
        "upstream_evidence_dates": evidence_dates,
    }


def _all_edge_components(candidate_ids: set[str], pairs: list[dict[str, Any]]) -> list[list[str]]:
    parent = {value: value for value in candidate_ids}

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
        union(str(pair["left_id"]), str(pair["right_id"]))

    groups: dict[str, list[str]] = defaultdict(list)
    for value in candidate_ids:
        groups[find(value)].append(value)
    result = [sorted(values) for values in groups.values()]
    result.sort(key=lambda values: (-len(values), values[0]))
    return result


def prepare_bridge_review(*, cross_day_root: Path, bridge_root: Path | None = None) -> dict[str, Any]:
    cross_day_root = cross_day_root.resolve()
    upstream_path = cross_day_root / "merge_staging.json"
    if not upstream_path.exists():
        raise FileNotFoundError(f"first-pass merge staging not found: {upstream_path}")
    upstream = json.loads(upstream_path.read_text(encoding="utf-8"))
    if not upstream.get("review_complete") or int(upstream.get("pending_summary_count") or 0) != 0:
        raise ValueError("first-pass review must be complete before bridge review")
    candidates = list(upstream.get("retained_candidates") or [])
    if not candidates:
        raise ValueError("first-pass merge staging has no retained candidates")

    by_id = {str(row.get("merge_id") or ""): row for row in candidates}
    if "" in by_id or len(by_id) != len(candidates):
        raise ValueError("invalid or duplicate first-pass merge_id")
    records = [_candidate_record(row) for row in candidates]
    all_pairs = score_related_pairs(records)
    cross_pairs = [
        pair
        for pair in all_pairs
        if _first_pass_batch(by_id[str(pair["left_id"])]) != _first_pass_batch(by_id[str(pair["right_id"])])
    ]
    bridge_ids = {value for pair in cross_pairs for value in (str(pair["left_id"]), str(pair["right_id"]))}
    components = _all_edge_components(bridge_ids, cross_pairs) if bridge_ids else []

    related: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in cross_pairs:
        related[str(pair["left_id"])].append({"summary_id": str(pair["right_id"]), "score": pair["score"]})
        related[str(pair["right_id"])].append({"summary_id": str(pair["left_id"]), "score": pair["score"]})
    for value in related:
        related[value].sort(key=lambda row: (-float(row["score"]), str(row["summary_id"])))

    record_by_id = {row["summary_id"]: row for row in records}
    paths = bridge_paths(cross_day_root, bridge_root)
    paths.groups_dir.mkdir(parents=True, exist_ok=True)
    paths.decisions_dir.mkdir(parents=True, exist_ok=True)

    groups: list[dict[str, Any]] = []
    for index, component in enumerate(
        sorted(components, key=lambda ids: (min(record_by_id[v]["date"] for v in ids), ids[0])), start=1
    ):
        component_set = set(component)
        items = []
        for value in sorted(component, key=lambda v: (record_by_id[v]["date"], v)):
            item = dict(record_by_id[value])
            item["related"] = [row for row in related.get(value, []) if row["summary_id"] in component_set]
            items.append(item)
        group = {
            "version": BRIDGE_VERSION,
            "batch_id": f"bridge_{index:03d}",
            "item_count": len(items),
            "items": items,
        }
        groups.append(group)
        _json_dump(paths.groups_dir / f"{group['batch_id']}.json", group)

    manifest = {
        "version": BRIDGE_VERSION,
        "review_version": REVIEW_VERSION,
        "cross_day_root": str(cross_day_root),
        "upstream_retained_candidate_count": len(candidates),
        "cross_batch_related_pair_count": len(cross_pairs),
        "bridge_candidate_count": len(bridge_ids),
        "unaffected_candidate_count": len(candidates) - len(bridge_ids),
        "group_count": len(groups),
        "largest_group_size": max((group["item_count"] for group in groups), default=0),
        "formal_memory_write": False,
        "rules": {
            "only_cross_first_pass_batch_edges": True,
            "all_edge_components_stay_together": True,
            "similarity_is_suggestion_only": True,
            "later_evidence_priority": True,
            "protected_events_must_remain_concrete": True,
            "project_state_is_not_current_rule_by_default": True,
        },
        "bridge_candidate_ids": sorted(bridge_ids),
        "cross_batch_related_pairs": cross_pairs,
    }
    _json_dump(paths.source_manifest, manifest)
    return manifest


def _load_bridge_groups(paths: BridgePaths) -> list[dict[str, Any]]:
    if not paths.source_manifest.exists():
        raise FileNotFoundError(f"bridge source manifest not found: {paths.source_manifest}")
    result: list[dict[str, Any]] = []
    for path in sorted(paths.groups_dir.glob("bridge_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"bridge group must be object: {path}")
        result.append(payload)
    return result


def _merge_unique(rows: list[dict[str, Any]], field: str) -> list[Any]:
    result: list[Any] = []
    for row in rows:
        for value in row.get(field) or []:
            if value not in result:
                result.append(value)
    return result


def build_bridge_merge_staging(
    upstream: dict[str, Any], bridge_manifest: dict[str, Any], reviewed_groups: list[dict[str, Any]]
) -> dict[str, Any]:
    upstream_candidates = list(upstream.get("retained_candidates") or [])
    upstream_by_id = {str(row.get("merge_id") or ""): row for row in upstream_candidates}
    reviewed_ids: set[str] = set()
    decision_groups: list[dict[str, Any]] = []
    final_candidates: list[dict[str, Any]] = []

    for reviewed in reviewed_groups:
        batch_id = str(reviewed.get("batch_id") or "")
        for group_index, decision in enumerate(reviewed.get("decisions") or [], start=1):
            source_ids = list(decision.get("summary_ids") or [])
            reviewed_ids.update(source_ids)
            decision_group_id = f"{batch_id}_g{group_index:02d}"
            decision_groups.append({"group_id": decision_group_id, **decision})
            canonical_id = str(decision.get("canonical_id") or "")
            retain_ids = list(decision.get("retain_ids") or [])
            collapse_to_canonical = bool(canonical_id and len(source_ids) > 1 and retain_ids == [canonical_id])
            for retain_id in retain_ids:
                source = upstream_by_id[retain_id]
                is_canonical = bool(canonical_id and retain_id == canonical_id)
                merge_ids = source_ids if is_canonical and collapse_to_canonical else [retain_id]
                rows = [upstream_by_id[value] for value in merge_ids]
                source_summary_ids = _merge_unique(rows, "source_summary_ids")
                source_event_ids = [int(value) for value in _merge_unique(rows, "source_event_ids")]
                evidence_dates = sorted(str(value) for value in _merge_unique(rows, "evidence_dates"))
                signals = [str(value) for value in _merge_unique(rows, "signals")]
                upstream_classifications: list[str] = []
                for row in rows:
                    value = str(row.get("classification") or "")
                    if value and value not in upstream_classifications:
                        upstream_classifications.append(value)
                final_candidates.append(
                    {
                        "merge_id": f"bhm_{hashlib.sha256((decision_group_id + ':' + retain_id).encode('utf-8')).hexdigest()[:20]}",
                        "group_id": decision_group_id,
                        "classification": decision["classification"],
                        "scope": decision["scope"],
                        "source_summary_id": source.get("source_summary_id") or "",
                        "canonical_source_id": source.get("canonical_source_id") or "",
                        "title": source.get("title") or "",
                        "summary": decision.get("merged_summary") if is_canonical and decision.get("merged_summary") else source.get("summary") or "",
                        "signals": signals if is_canonical else list(source.get("signals") or []),
                        "source_summary_ids": source_summary_ids if is_canonical else list(source.get("source_summary_ids") or []),
                        "source_event_ids": source_event_ids if is_canonical else list(source.get("source_event_ids") or []),
                        "evidence_dates": evidence_dates if is_canonical else list(source.get("evidence_dates") or []),
                        "protected_event": any(bool(row.get("protected_event")) for row in rows) if is_canonical else bool(source.get("protected_event")),
                        "projectish": any(bool(row.get("projectish")) for row in rows) if is_canonical else bool(source.get("projectish")),
                        "upstream_merge_ids": merge_ids if is_canonical else [retain_id],
                        "upstream_classifications": upstream_classifications if is_canonical else [str(source.get("classification") or "")],
                    }
                )

    bridge_ids = {str(value) for value in bridge_manifest.get("bridge_candidate_ids") or []}
    for candidate in upstream_candidates:
        merge_id = str(candidate.get("merge_id") or "")
        if merge_id not in bridge_ids:
            carry = dict(candidate)
            carry["bridge_status"] = "unaffected_carry_through"
            carry["upstream_merge_ids"] = [merge_id]
            final_candidates.append(carry)

    pending_ids = sorted(bridge_ids - reviewed_ids)
    final_candidates.sort(key=lambda row: ((row.get("evidence_dates") or [""])[-1], str(row.get("merge_id") or "")))
    return {
        "version": BRIDGE_VERSION,
        "upstream_retained_candidate_count": len(upstream_candidates),
        "bridge_candidate_count": len(bridge_ids),
        "reviewed_bridge_candidate_count": len(reviewed_ids),
        "pending_bridge_candidate_count": len(pending_ids),
        "unaffected_candidate_count": len(upstream_candidates) - len(bridge_ids),
        "bridge_review_complete": not pending_ids,
        "formal_memory_write": False,
        "decision_groups": decision_groups,
        "retained_candidates": final_candidates,
        "pending_bridge_candidate_ids": pending_ids,
    }


def import_bridge_manual_outputs(
    *, cross_day_root: Path, manual_dir: Path, bridge_root: Path | None = None, overwrite_existing: bool = False
) -> dict[str, Any]:
    cross_day_root = cross_day_root.resolve()
    paths = bridge_paths(cross_day_root, bridge_root)
    bridge_manifest = json.loads(paths.source_manifest.read_text(encoding="utf-8"))
    upstream = json.loads((cross_day_root / "merge_staging.json").read_text(encoding="utf-8"))
    groups = _load_bridge_groups(paths)
    manual_dir = manual_dir.resolve()
    imported = 0
    missing = 0
    invalid = 0
    skipped_existing = 0

    for group in groups:
        batch_id = str(group.get("batch_id") or "")
        manual_path = manual_dir / f"{batch_id}.json"
        decision_path = paths.decisions_dir / f"{batch_id}.json"
        if decision_path.exists() and not overwrite_existing:
            skipped_existing += 1
            continue
        if not manual_path.exists():
            missing += 1
            continue
        try:
            raw = json.loads(manual_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("manual bridge output must be a JSON object")
            normalized = normalize_manual_review_batch(group, raw)
            normalized["origin"] = "manual_chatgpt_cross_batch_bridge"
            _json_dump(decision_path, normalized)
            imported += 1
        except Exception as exc:
            invalid += 1
            print(f"CROSS_BATCH_BRIDGE_IMPORT_FAILED {batch_id}: {exc}")

    reviewed_groups = [
        json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths.decisions_dir.glob("bridge_*.json"))
    ]
    staging = build_bridge_merge_staging(upstream, bridge_manifest, reviewed_groups)
    _json_dump(paths.merge_staging, staging)
    return {
        "imported_groups": imported,
        "missing_manual_groups": missing,
        "invalid_manual_groups": invalid,
        "skipped_existing_groups": skipped_existing,
        "reviewed_bridge_candidate_count": staging["reviewed_bridge_candidate_count"],
        "pending_bridge_candidate_count": staging["pending_bridge_candidate_count"],
        "final_retained_candidate_count": len(staging["retained_candidates"]),
        "bridge_review_complete": staging["bridge_review_complete"],
        "merge_staging": str(paths.merge_staging),
        "formal_memory_write": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Second-pass cross-batch bridge review for historical memory staging")
    parser.add_argument("--cross-day-root", required=True)
    parser.add_argument("--bridge-root", default="")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--import-manual-dir", default="")
    parser.add_argument("--overwrite-manual-existing", action="store_true")
    args = parser.parse_args()
    if bool(args.prepare) == bool(str(args.import_manual_dir).strip()):
        parser.error("choose exactly one of --prepare or --import-manual-dir")

    cross_day_root = Path(args.cross_day_root)
    bridge_root = Path(args.bridge_root).resolve() if str(args.bridge_root).strip() else None
    if args.prepare:
        manifest = prepare_bridge_review(cross_day_root=cross_day_root, bridge_root=bridge_root)
        result = {
            "status": "bridge_review_prepared",
            "upstream_retained_candidate_count": manifest["upstream_retained_candidate_count"],
            "cross_batch_related_pair_count": manifest["cross_batch_related_pair_count"],
            "bridge_candidate_count": manifest["bridge_candidate_count"],
            "unaffected_candidate_count": manifest["unaffected_candidate_count"],
            "group_count": manifest["group_count"],
            "largest_group_size": manifest["largest_group_size"],
            "formal_memory_write": False,
        }
    else:
        result = {
            "status": "bridge_manual_imported",
            **import_bridge_manual_outputs(
                cross_day_root=cross_day_root,
                manual_dir=Path(args.import_manual_dir),
                bridge_root=bridge_root,
                overwrite_existing=bool(args.overwrite_manual_existing),
            ),
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
